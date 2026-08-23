from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    duration_ms: int
    started_at: str | None = None
    finished_at: str | None = None
    timeout_seconds: int | None = None
    deadline_overrun_ms: int = 0
    deadline_overrun: bool = False
    host_pause_suspected: bool = False

    def audit_dict(self) -> dict[str, object]:
        return asdict(self)


class CommandError(RuntimeError):
    def __init__(self, result: CommandResult):
        self.result = result
        detail = result.stderr.strip() or result.stdout.strip() or "no command output"
        super().__init__(
            f"Command exited {result.returncode}: {' '.join(result.argv)}: {detail}"
        )


class CommandRunner:
    @staticmethod
    def _windows_descendant_pids(root_pid: int) -> list[int]:
        """Snapshot descendants before taskkill can orphan them."""

        import ctypes
        from ctypes import wintypes

        class PROCESSENTRY32W(ctypes.Structure):
            _fields_ = [
                ("dwSize", wintypes.DWORD),
                ("cntUsage", wintypes.DWORD),
                ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.c_void_p),
                ("th32ModuleID", wintypes.DWORD),
                ("cntThreads", wintypes.DWORD),
                ("th32ParentProcessID", wintypes.DWORD),
                ("pcPriClassBase", wintypes.LONG),
                ("dwFlags", wintypes.DWORD),
                ("szExeFile", wintypes.WCHAR * 260),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        kernel32.Process32FirstW.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESSENTRY32W),
        ]
        kernel32.Process32FirstW.restype = wintypes.BOOL
        kernel32.Process32NextW.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESSENTRY32W),
        ]
        kernel32.Process32NextW.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL

        snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
        invalid_handle = ctypes.c_void_p(-1).value
        if not snapshot or snapshot == invalid_handle:
            return []
        children: dict[int, list[int]] = {}
        try:
            entry = PROCESSENTRY32W()
            entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
            more = bool(kernel32.Process32FirstW(snapshot, ctypes.byref(entry)))
            while more:
                children.setdefault(int(entry.th32ParentProcessID), []).append(
                    int(entry.th32ProcessID)
                )
                more = bool(kernel32.Process32NextW(snapshot, ctypes.byref(entry)))
        finally:
            kernel32.CloseHandle(snapshot)

        descendants: list[int] = []
        pending = list(children.get(root_pid, []))
        while pending:
            pid = pending.pop()
            descendants.append(pid)
            pending.extend(children.get(pid, []))
        return descendants

    @staticmethod
    def _windows_force_terminate(pids: list[int]) -> list[str]:
        """Terminate snapshotted descendants that survived taskkill /T."""

        import ctypes
        from ctypes import wintypes

        notes: list[str] = []
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.TerminateProcess.restype = wintypes.BOOL
        kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        for pid in reversed(pids):
            handle = kernel32.OpenProcess(0x0001 | 0x00100000, False, pid)
            if not handle:
                continue
            try:
                if not kernel32.TerminateProcess(handle, 1):
                    notes.append(
                        f"TerminateProcess({pid}) failed with {ctypes.get_last_error()}"
                    )
                else:
                    kernel32.WaitForSingleObject(handle, 5000)
            finally:
                kernel32.CloseHandle(handle)
        return notes

    @staticmethod
    def _terminate_process_tree(process: subprocess.Popen[str]) -> list[str]:
        """Best-effort bounded termination for the complete spawned process tree."""

        notes: list[str] = []
        if process.poll() is not None:
            return notes
        if os.name == "nt":
            descendants = CommandRunner._windows_descendant_pids(process.pid)
            system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
            taskkill = system_root / "System32" / "taskkill.exe"
            try:
                killer = subprocess.Popen(
                    [
                        str(taskkill),
                        "/PID",
                        str(process.pid),
                        "/T",
                        "/F",
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    shell=False,
                )
                try:
                    taskkill_code = killer.wait(timeout=10)
                    if taskkill_code:
                        notes.append(f"taskkill exited {taskkill_code}")
                except subprocess.TimeoutExpired:
                    killer.kill()
                    killer.wait(timeout=5)
                    notes.append("taskkill itself exceeded 10s")
            except (OSError, subprocess.SubprocessError) as exc:
                notes.append(f"taskkill failed: {type(exc).__name__}: {exc}")
            notes.extend(CommandRunner._windows_force_terminate(descendants))
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as exc:
                notes.append(f"process-group kill failed: {type(exc).__name__}: {exc}")

        if process.poll() is None:
            try:
                process.kill()
            except OSError as exc:
                notes.append(f"direct-process kill failed: {type(exc).__name__}: {exc}")
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            notes.append("direct process did not exit within 10s after termination")
        return notes

    def run(
        self,
        argv: Sequence[str | Path],
        *,
        timeout: int,
        check: bool = True,
        env: Mapping[str, str] | None = None,
        input_text: str | None = None,
        cwd: str | Path | None = None,
    ) -> CommandResult:
        rendered = tuple(str(item) for item in argv)
        if not rendered:
            raise ValueError("Command argv cannot be empty")
        child_env = os.environ.copy()
        if env:
            child_env.update(env)
        started = time.monotonic()
        started_at = datetime.now(timezone.utc).isoformat()
        timed_out = False
        termination_notes: list[str] = []

        def result_metadata() -> dict[str, object]:
            duration_ms = round((time.monotonic() - started) * 1000)
            overrun_ms = max(duration_ms - (timeout * 1000), 0)
            # A few seconds of termination/OS scheduling overhead is normal.
            # A command that returns normally far beyond its stated timeout is
            # not: on Windows this can happen when the host sleeps or the Docker
            # VM pauses while the OS wait timeout is suspended.
            deadline_overrun = overrun_ms > 5_000
            return {
                "duration_ms": duration_ms,
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "timeout_seconds": timeout,
                "deadline_overrun_ms": overrun_ms,
                "deadline_overrun": deadline_overrun,
                "host_pause_suspected": (
                    deadline_overrun and not timed_out and overrun_ms > 60_000
                ),
            }
        try:
            with (
                tempfile.TemporaryFile() as stdout_file,
                tempfile.TemporaryFile() as stderr_file,
            ):
                popen_kwargs: dict[str, object] = {
                    "stdin": subprocess.PIPE if input_text is not None else None,
                    "stdout": stdout_file,
                    "stderr": stderr_file,
                    "text": True,
                    "encoding": "utf-8",
                    "errors": "replace",
                    "shell": False,
                    "env": child_env,
                    "cwd": str(cwd) if cwd is not None else None,
                }
                if os.name == "nt":
                    popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
                else:
                    popen_kwargs["start_new_session"] = True
                process = subprocess.Popen(list(rendered), **popen_kwargs)
                try:
                    process.communicate(input=input_text, timeout=timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    termination_notes = self._terminate_process_tree(process)
                stdout_file.flush()
                stderr_file.flush()
                stdout_file.seek(0)
                stderr_file.seek(0)
                stdout = stdout_file.read().decode("utf-8", errors="replace")
                stderr = stderr_file.read().decode("utf-8", errors="replace")
        except FileNotFoundError as exc:
            result = CommandResult(
                argv=rendered,
                returncode=127,
                stdout="",
                stderr=f"command was not found: {rendered[0]}",
                **result_metadata(),
            )
            if check:
                raise CommandError(result) from exc
            return result
        if timed_out:
            timeout_message = f"timed out after {timeout}s; spawned process tree terminated"
            if termination_notes:
                timeout_message += "; " + "; ".join(termination_notes)
            result = CommandResult(
                argv=rendered,
                returncode=124,
                stdout=stdout,
                stderr=(
                    f"{stderr.rstrip()}\n{timeout_message}"
                    if stderr
                    else timeout_message
                ),
                **result_metadata(),
            )
            if check:
                raise CommandError(result)
            return result
        result = CommandResult(
            argv=rendered,
            returncode=int(process.returncode),
            stdout=stdout,
            stderr=stderr,
            **result_metadata(),
        )
        if check and result.returncode:
            raise CommandError(result)
        return result
