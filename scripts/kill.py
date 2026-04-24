"""Shared helpers for the train/test orchestrators.

Run directly to kill every Diffusion-TS job (train + test):
  python scripts/_jobs.py

Layout:
  Diffusion-TS/
    .venv/bin/python         <- PY
    logs/<name>.log          <- stdout+stderr of each job
    logs/pids/<name>.pid     <- process-group id used by kill_all
"""
from __future__ import annotations

import os
import signal
import sys
from pathlib import Path


from setup import LOG_DIR, PID_DIR, ensure_dirs

    
def kill_all(pidfiles: list[Path]) -> int:
    if not pidfiles:
        print("nothing to kill")
        return 0
    killed = 0
    for pidfile in sorted(pidfiles):
        name = pidfile.stem
        try:
            pid = int(pidfile.read_text().strip())
        except ValueError:
            print(f"  ! bad pid file: {pidfile}", file=sys.stderr)
            pidfile.unlink(missing_ok=True)
            continue
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            print(f"SIGTERM -> {name} (pid {pid})")
            killed += 1
        except ProcessLookupError:
            print(f"{name} (pid {pid}) already gone")
        pidfile.unlink(missing_ok=True)
    print(f"killed {killed} process group(s)")
    return 0


if __name__ == "__main__":
    import sys
    ensure_dirs()
    sys.exit(kill_all(list(PID_DIR.glob("*.pid"))))
