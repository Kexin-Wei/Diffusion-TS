from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


DTS = Path(__file__).resolve().parent.parent
PY = DTS / ".venv" / "bin" / "python"
LOG_DIR = DTS / "logs"
PID_DIR = LOG_DIR / "pids"


def ensure_dirs() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PID_DIR.mkdir(parents=True, exist_ok=True)


def launch_bg(name: str, cmd: list[str], gpu: int) -> int | None:
    """Run cmd detached, one GPU visible via CUDA_VISIBLE_DEVICES.
    Writes logs/{name}.log and logs/pids/{name}.pid. Returns pid or None."""
    log = LOG_DIR / f"{name}.log"
    pidfile = PID_DIR / f"{name}.pid"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    print(f"{name} -> GPU {gpu}  (log: {log})")
    try:
        logf = open(log, "wb")
        proc = subprocess.Popen(
            cmd, cwd=DTS, env=env,
            stdout=logf, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except OSError as e:
        print(f"  ! launch failed for {name}: {e}", file=sys.stderr)
        return None
    pidfile.write_text(f"{proc.pid}\n")
    print(f"       pid={proc.pid}")
    return proc.pid
