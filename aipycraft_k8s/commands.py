from __future__ import annotations

import os
import subprocess
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
    def run(
        self,
        argv: Sequence[str | Path],
        *,
        timeout: int,
        check: bool = True,
        env: Mapping[str, str] | None = None,
    ) -> CommandResult:
        rendered = tuple(str(item) for item in argv)
        if not rendered:
            raise ValueError("Command argv cannot be empty")
        child_env = os.environ.copy()
        if env:
            child_env.update(env)
        started = time.monotonic()
        try:
            completed = subprocess.run(
                list(rendered),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
                shell=False,
                env=child_env,
            )
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
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            result = CommandResult(
                argv=rendered,
                returncode=124,
                stdout=stdout,
                stderr=stderr or f"timed out after {timeout}s",
                duration_ms=round((time.monotonic() - started) * 1000),
            )
            if check:
                raise CommandError(result) from exc
            return result
        result = CommandResult(
            argv=rendered,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_ms=round((time.monotonic() - started) * 1000),
        )
        if check and result.returncode:
            raise CommandError(result)
        return result
