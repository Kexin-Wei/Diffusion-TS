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
    SEQ_LENGTH_DEFAULT,
)


class CONFIGS(Enum):
    SINES = "sines"
    ENERGY = "energy"
    STOCKS = "stocks"
    ETT_H = "etth"
    FMRI = "fmri"
    MUJOCO = "mujoco"


# (cfg, seq_length, ratios) per GPU. seq_length must match the trained
# checkpoint (see train_reproduce.py); ratios default is MISSING_RATIOS_DEFAULT.
IMPUTE_RUNS: dict[int, list[tuple[CONFIGS, int, list[float]]]] = {
    0: [(CONFIGS.SINES, SEQ_LENGTH_DEFAULT, list(MISSING_RATIOS_DEFAULT))],
    2: [(CONFIGS.MUJOCO, SEQ_LENGTH_DEFAULT, list(MISSING_RATIOS_DEFAULT))],
    3: [(CONFIGS.STOCKS, SEQ_LENGTH_DEFAULT, list(MISSING_RATIOS_DEFAULT))],
    4: [(CONFIGS.ETT_H, SEQ_LENGTH_DEFAULT, list(MISSING_RATIOS_DEFAULT))],
    5: [(CONFIGS.ENERGY, SEQ_LENGTH_DEFAULT, list(MISSING_RATIOS_DEFAULT))],
    6: [(CONFIGS.FMRI, SEQ_LENGTH_DEFAULT, list(MISSING_RATIOS_DEFAULT))],
}


def _run_worker(gpu: int) -> None:
    """Worker process: run every (cfg, seq_length, ratios) in IMPUTE_RUNS[gpu]
    on this GPU.

    One worker owns one GPU end-to-end. Sequential inside the worker;
    parallelism only happens across workers (one OS process per GPU).
    Redirects stdout/stderr to logs/imputation_gpu{N}.log.
    """
    log_path = Path("logs") / f"imputation_gpu{gpu}.log"
    log_path.parent.mkdir(exist_ok=True)
    sys.stdout = sys.stderr = open(log_path, "w", buffering=1)  # line-buffered

    for cfg, seq_length, ratios in IMPUTE_RUNS[gpu]:
        print(
            f"[gpu={gpu}] starting {cfg.value} seq_length={seq_length} ratios={ratios}",
            flush=True,
        )
        impute_one(
            cfg=cfg.value,
            gpu=gpu,
            seed=SEED_DEFAULT,
            milestone=MILESTONE_DEFAULT,
            ratios=ratios,
            seq_length=seq_length,
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
