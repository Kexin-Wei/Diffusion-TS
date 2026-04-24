#!/usr/bin/env python3
"""Train Diffusion-TS on the standard benchmark datasets.

Run from the Diffusion-TS directory:
  python scripts/train_diffusion_ts.py              # parallel, one dataset per GPU
  python scripts/train_diffusion_ts.py --only etth  # single dataset

Kill running jobs: python scripts/_jobs.py
"""
from __future__ import annotations

import os
import sys
import argparse
import time
import torch
import numpy as np

from setup import LOG_DIR, PY, check_setup, ensure_dirs, launch_bg

GPU_OF: dict[str, int] = {
    "sines": 0,
    "stocks": 2,
    "etth": 4,
    "energy": 5,
    "fmri": 6,
    "mujoco": 7,
}


def train_cmd(cfg: str) -> list[str]:
    return [
        str(PY), "main.py",
        "--name", cfg,
        "--config_file", f"Config/{cfg}.yaml",
        "--gpu", "0",
        "--train",
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", metavar="DATASET", help="train a single dataset")
    args = ap.parse_args()

    ensure_dirs()

    runs = [args.only] if args.only else list(GPU_OF)
    rc = check_setup(runs, GPU_OF)
    if rc != 0:
        return rc

    failures: list[str] = []
    for cfg in runs:
        if launch_bg(cfg, train_cmd(cfg), GPU_OF[cfg]) is None:
            failures.append(cfg)
        time.sleep(0.5)  # stagger CUDA init

    print()
    print(f"monitor:  tail -f {LOG_DIR}/*.log")
    print("kill:     python scripts/_jobs.py")

    if failures:
        print(f"\nfailures: {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
