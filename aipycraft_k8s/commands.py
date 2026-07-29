from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    duration_ms: int

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
    def _terminate_process_tree(process: subprocess.Popen[str]) -> list[str]:
        """Best-effort bounded termination for the complete spawned process tree."""

        notes: list[str] = []
        if process.poll() is not None:
            return notes
        if os.name == "nt":
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
        timed_out = False
        termination_notes: list[str] = []
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
                duration_ms=round((time.monotonic() - started) * 1000),
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
                duration_ms=round((time.monotonic() - started) * 1000),
            )
            if check:
                raise CommandError(result)
            return result
        result = CommandResult(
            argv=rendered,
            returncode=int(process.returncode),
            stdout=stdout,
            stderr=stderr,
            duration_ms=round((time.monotonic() - started) * 1000),
        )
        if check and result.returncode:
            raise CommandError(result)
        return result
