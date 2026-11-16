# wumpus_engine/gateway/subprocess_launcher.py
from __future__ import annotations

import os
import pathlib
import queue
import subprocess
import sys
import threading
from typing import Any
from wumpus_engine.config.schema import SandboxLevel


class SubprocessLauncher:
    """Spawns and manages isolated child process running `python -s -P -B -m wumpus_sdk.host`."""

    def __init__(
        self,
        entry_point: str,
        source_dir: pathlib.Path | str | None = None,
        isolation_level: SandboxLevel = SandboxLevel.PROCESS,
        python_executable: str | None = None,
    ) -> None:
        self.entry_point = entry_point
        self.source_dir = pathlib.Path(source_dir).resolve() if source_dir else None
        self.isolation_level = isolation_level
        self.python_executable = python_executable or sys.executable
        self.proc: subprocess.Popen[str] | None = None
        self._stdout_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._stderr_lines: list[str] = []

    def launch(self) -> None:
        cmd = [
            self.python_executable,
            "-s",
            "-P",
            "-B",
            "-m",
            "wumpus_sdk.host",
            self.entry_point,
        ]

        # Construct scrubbed PYTHONPATH: project root + agent source_dir (excluding wumpus_engine)
        project_root = pathlib.Path(__file__).resolve().parents[2]
        pythonpath_dirs = [str(project_root)]
        if self.source_dir:
            pythonpath_dirs.append(str(self.source_dir))

        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_dirs)

        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )

        t_out = threading.Thread(target=self._stdout_reader, daemon=True)
        t_err = threading.Thread(target=self._stderr_reader, daemon=True)
        t_out.start()
        t_err.start()

    def _stdout_reader(self) -> None:
        if self.proc is None or self.proc.stdout is None:
            return
        try:
            for line in self.proc.stdout:
                self._stdout_queue.put(("LINE", line))
        except Exception as exc:
            self._stdout_queue.put(("ERROR", exc))
        finally:
            self._stdout_queue.put(("EOF", None))

    def _stderr_reader(self) -> None:
        if self.proc is None or self.proc.stderr is None:
            return
        try:
            for line in self.proc.stderr:
                self._stderr_lines.append(line)
        except Exception:
            pass

    def write_line(self, line: str) -> int:
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("Process not running")
        payload = line + "\n"
        self.proc.stdin.write(payload)
        self.proc.stdin.flush()
        return len(payload.encode("utf-8"))

    def read_line(self, timeout_s: float) -> tuple[str | None, int]:
        try:
            event, data = self._stdout_queue.get(timeout=timeout_s)
        except queue.Empty:
            raise TimeoutError(f"Read timed out after {timeout_s}s")

        if event == "LINE":
            raw_line: str = data
            return raw_line.rstrip("\r\n"), len(raw_line.encode("utf-8"))
        if event == "EOF":
            return None, 0
        if event == "ERROR":
            raise RuntimeError(f"Error reading stdout: {data}")
        return None, 0

    def get_stderr_tail(self, max_chars: int = 4096) -> str:
        return "".join(self._stderr_lines)[-max_chars:]

    def terminate(self) -> None:
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=0.5)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass