#!/usr/bin/env python3
"""Parallel multi-GPU caller for `forecast_one`.

Spawns one OS process per GPU, each running its slice of FORECAST_RUNS sequentially.
The single-dataset forecasting pipeline lives in `forecasting_one.py`; this file
owns only the sweep definition and the cross-GPU orchestration.

Mirrors the train_reproduce.py shape — same mp.spawn pattern, same logs/ scheme.

Usage (runs every entry in FORECAST_RUNS sequentially per GPU, in parallel across GPUs):
  ./Diffusion-TS/.venv/bin/python scripts/forecasting_reproduce.py

To subset the sweep, edit FORECAST_RUNS below.
"""
from __future__ import annotations

import multiprocessing as mp
import sys
from enum import Enum
from pathlib import Path

from forecasting_one import (
    forecast_one,
    MILESTONE_DEFAULT,
    PRED_LENS_DEFAULT,
    SEED_DEFAULT,
)


class CONFIGS(Enum):
    SINES = "sines"
    ENERGY = "energy"
    STOCKS = "stocks"
    ETT_H = "etth"
    FMRI = "fmri"
    MUJOCO = "mujoco"


# (cfg, pred_lens) per GPU. Override pred_lens per-job when paper Table specifies
# multiple horizons for a dataset; default is PRED_LENS_DEFAULT.
FORECAST_RUNS: dict[int, list[tuple[CONFIGS, list[int]]]] = {
    0: [(CONFIGS.SINES, list(PRED_LENS_DEFAULT))],
    2: [(CONFIGS.MUJOCO, list(PRED_LENS_DEFAULT))],
    3: [(CONFIGS.STOCKS, list(PRED_LENS_DEFAULT))],
    4: [(CONFIGS.ETT_H, list(PRED_LENS_DEFAULT))],
    5: [(CONFIGS.ENERGY, list(PRED_LENS_DEFAULT))],
    6: [(CONFIGS.FMRI, list(PRED_LENS_DEFAULT))],
}


def _run_worker(gpu: int) -> None:
    """Worker process: run every (cfg, pred_lens) in FORECAST_RUNS[gpu] on this GPU.

    One worker owns one GPU end-to-end. Sequential inside the worker;
    parallelism only happens across workers (one OS process per GPU).
    Redirects stdout/stderr to logs/forecasting_gpu{N}.log.
    """
    log_path = Path("logs") / f"forecasting_gpu{gpu}.log"
    log_path.parent.mkdir(exist_ok=True)
    sys.stdout = sys.stderr = open(log_path, "w", buffering=1)  # line-buffered

    for cfg, pred_lens in FORECAST_RUNS[gpu]:
        print(f"[gpu={gpu}] starting {cfg.value} pred_lens={pred_lens}", flush=True)
        forecast_one(
            cfg=cfg.value,
            gpu=gpu,
            seed=SEED_DEFAULT,
            milestone=MILESTONE_DEFAULT,
            pred_lens=pred_lens,
        )


def run_parallel() -> int:
    """Parent: spawn one mp.Process per GPU with the 'spawn' start method.

    Same rationale as train_reproduce.py — fresh CUDA context per child,
    isolated crashes, fork bomb structurally impossible thanks to __main__ guard.
    """
    ctx = mp.get_context("spawn")
    procs = {gpu: ctx.Process(target=_run_worker, args=(gpu,)) for gpu in FORECAST_RUNS}
    for gpu, p in procs.items():
        p.start()
        print(f"[parent] gpu={gpu} pid={p.pid} → logs/forecasting_gpu{gpu}.log")

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
