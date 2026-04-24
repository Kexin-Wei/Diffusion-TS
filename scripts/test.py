#!/usr/bin/env python3
"""Test Diffusion-TS imputation on the standard benchmark datasets.

For each dataset, one bash job sweeps every missing ratio sequentially and
renames outputs so they don't overwrite each other:
  OUTPUT/<name>/ddpm_infill_<name>_m{70,80,90}.npy

Paper params (Yuan & Qiao 2024, §B.3 + Table 6) come from the config's
test_dataset block: coefficient=1e-2, step_size=0.05, sampling_steps from
config.

Run from the Diffusion-TS directory:
  python scripts/test_diffusion_ts.py                # parallel, one dataset per GPU
  python scripts/test_diffusion_ts.py --only etth    # single dataset
  python scripts/test_diffusion_ts.py --ratios 0.7 0.8 0.9
  python scripts/test_diffusion_ts.py --milestone 10

Kill running jobs: python scripts/_jobs.py
"""
from __future__ import annotations

import argparse
import sys
import time

from setup import LOG_DIR, PY, check_setup, ensure_dirs, launch_bg

GPU_OF: dict[str, int] = {
    "sines": 0,
    "stocks": 3,
    "etth": 4,
    "energy": 5,
    "fmri": 6,
    "mujoco": 2,
}
MISSING_RATIOS: tuple[float, ...] = (0.7, 0.8, 0.9)
MILESTONE_DEFAULT = 10


def test_cmd(cfg: str, ratios: list[float], milestone: int) -> list[str]:
    py = str(PY)
    raw = f"OUTPUT/{cfg}/ddpm_infill_{cfg}.npy"
    parts: list[str] = []
    for r in ratios:
        tag = f"m{int(round(r * 100))}"
        parts.append(
            f"{py} main.py --name {cfg} --config_file Config/{cfg}.yaml "
            f"--gpu 0 --sample 1 --mode infill "
            f"--missing_ratio {r} --milestone {milestone}"
        )
        parts.append(f"mv {raw} OUTPUT/{cfg}/ddpm_infill_{cfg}_{tag}.npy")
    return ["bash", "-c", " && ".join(parts)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", metavar="DATASET", help="test a single dataset")
    ap.add_argument(
        "--ratios", type=float, nargs="+", default=list(MISSING_RATIOS),
        metavar="R", help="missing ratios to sweep (default: 0.7 0.8 0.9)",
    )
    ap.add_argument(
        "--milestone", type=int, default=MILESTONE_DEFAULT,
        help="checkpoint milestone to load (default: 10 = final)",
    )
    args = ap.parse_args()

    ensure_dirs()

    runs = [args.only] if args.only else list(GPU_OF)
    rc = check_setup(runs, GPU_OF)
    if rc != 0:
        return rc

    failures: list[str] = []
    for cfg in runs:
        name = f"test_{cfg}"
        cmd = test_cmd(cfg, args.ratios, args.milestone)
        if launch_bg(name, cmd, GPU_OF[cfg]) is None:
            failures.append(cfg)
        time.sleep(0.5)

    print()
    print(f"monitor:  tail -f {LOG_DIR}/test_*.log")
    print("kill:     python scripts/_jobs.py")

    if failures:
        print(f"\nfailures: {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
