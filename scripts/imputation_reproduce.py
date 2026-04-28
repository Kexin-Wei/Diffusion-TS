#!/usr/bin/env python3
"""Parallel multi-GPU caller for `impute_one`.

Spawns one OS process per GPU, each running its slice of IMPUTE_RUNS sequentially.
The single-dataset imputation pipeline lives in `imputation_one.py`; this file
owns only the sweep definition and the cross-GPU orchestration.

Mirrors the train_reproduce.py shape — same mp.spawn pattern, same logs/ scheme.

Usage (runs every entry in IMPUTE_RUNS sequentially per GPU, in parallel across GPUs):
  ./Diffusion-TS/.venv/bin/python scripts/imputation_reproduce.py

To subset the sweep, edit IMPUTE_RUNS below.
"""
from __future__ import annotations

import multiprocessing as mp
import sys
from enum import Enum
from pathlib import Path

from imputation_one import (
    impute_one,
    MILESTONE_DEFAULT,
    MISSING_RATIOS_DEFAULT,
    SEED_DEFAULT,
)


class CONFIGS(Enum):
    SINES = "sines"
    ENERGY = "energy"
    STOCKS = "stocks"
    ETT_H = "etth"
    FMRI = "fmri"
    MUJOCO = "mujoco"


# (cfg, ratios) per GPU. Override ratios per-job when a dataset needs a
# different sweep; default is MISSING_RATIOS_DEFAULT (0.7, 0.8, 0.9).
IMPUTE_RUNS: dict[int, list[tuple[CONFIGS, list[float]]]] = {
    0: [(CONFIGS.SINES, list(MISSING_RATIOS_DEFAULT))],
    2: [(CONFIGS.MUJOCO, list(MISSING_RATIOS_DEFAULT))],
    3: [(CONFIGS.STOCKS, list(MISSING_RATIOS_DEFAULT))],
    4: [(CONFIGS.ETT_H, list(MISSING_RATIOS_DEFAULT))],
    5: [(CONFIGS.ENERGY, list(MISSING_RATIOS_DEFAULT))],
    6: [(CONFIGS.FMRI, list(MISSING_RATIOS_DEFAULT))],
}


def _run_worker(gpu: int) -> None:
    """Worker process: run every (cfg, ratios) in IMPUTE_RUNS[gpu] on this GPU.

    One worker owns one GPU end-to-end. Sequential inside the worker;
    parallelism only happens across workers (one OS process per GPU).
    Redirects stdout/stderr to logs/imputation_gpu{N}.log.
    """
    log_path = Path("logs") / f"imputation_gpu{gpu}.log"
    log_path.parent.mkdir(exist_ok=True)
    sys.stdout = sys.stderr = open(log_path, "w", buffering=1)  # line-buffered

    for cfg, ratios in IMPUTE_RUNS[gpu]:
        print(f"[gpu={gpu}] starting {cfg.value} ratios={ratios}", flush=True)
        impute_one(
            cfg=cfg.value,
            gpu=gpu,
            seed=SEED_DEFAULT,
            milestone=MILESTONE_DEFAULT,
            ratios=ratios,
        )


def run_parallel() -> int:
    """Parent: spawn one mp.Process per GPU with the 'spawn' start method.

    Same rationale as train_reproduce.py — fresh CUDA context per child,
    isolated crashes, fork bomb structurally impossible thanks to __main__ guard.
    """
    ctx = mp.get_context("spawn")
    procs = {gpu: ctx.Process(target=_run_worker, args=(gpu,)) for gpu in IMPUTE_RUNS}
    for gpu, p in procs.items():
        p.start()
        print(f"[parent] gpu={gpu} pid={p.pid} → logs/imputation_gpu{gpu}.log")

    failed: list[int] = []
    for gpu, p in procs.items():
        p.join()
        status = "OK" if p.exitcode == 0 else f"FAIL (rc={p.exitcode})"
        print(f"[parent] gpu={gpu} done: {status}")
        if p.exitcode != 0:
            failed.append(gpu)

    if failed:
        print(f"[parent] failed GPUs: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run_parallel())
