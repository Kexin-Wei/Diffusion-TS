#!/usr/bin/env python3
"""Parallel multi-GPU caller for `train_one`.

Spawns one OS process per GPU, each running its slice of PAPER_RUNS sequentially.
The single-dataset training pipeline lives in `train_one.py`; this file owns
only the sweep definition and the cross-GPU orchestration.

Usage (runs every entry in PAPER_RUNS sequentially per GPU, in parallel across GPUs):
  ./Diffusion-TS/.venv/bin/python scripts/train.py

To subset the sweep, edit PAPER_RUNS below.
"""
from __future__ import annotations

import multiprocessing as mp
import sys
from enum import Enum
from pathlib import Path

from train_one import train_one


class CONFIGS(Enum):
    SINES = "sines"
    ENERGY = "energy"
    STOCKS = "stocks"
    ETT_H = "etth"
    FMRI = "fmri"
    MUJOCO = "mujoco"


SEED_DEFAULT = 12345
LOG_FREQUENCY = 100
TENSORBOARD = False  # flip to True to enable TensorBoard scalar logging

# Smoke test: one short run per GPU to verify the pipeline works end-to-end.
PAPER_RUNS: dict[int, list[tuple[CONFIGS, int]]] = {
    5: [(CONFIGS.ENERGY, 24)],
    6: [(CONFIGS.MUJOCO, 24)],
    7: [(CONFIGS.ETT_H, 24)],
}

# Paper sweep — sequence lengths follow train_one.py's docstring. Allocation is
# longest-job-first (LPT) across GPUs 5/6/7: ENERGY/256 anchors GPU 5,
# heavy ETT_H runs cluster on GPU 7.
# PAPER_RUNS: dict[int, list[tuple[CONFIGS, int]]] = {
#     5: [
#         (CONFIGS.ENERGY, 256),
#         (CONFIGS.ENERGY, 24),
#         (CONFIGS.MUJOCO, 24),
#         (CONFIGS.STOCKS, 24),
#     ],
#     6: [
#         (CONFIGS.ENERGY, 128),
#         (CONFIGS.MUJOCO, 100),
#         (CONFIGS.FMRI, 24),
#         (CONFIGS.ETT_H, 64),
#     ],
#     7: [
#         (CONFIGS.ETT_H, 256),
#         (CONFIGS.ENERGY, 64),
#         (CONFIGS.ETT_H, 128),
#         (CONFIGS.ETT_H, 24),
#         (CONFIGS.SINES, 24),
#     ],
# }


def _run_worker(gpu: int) -> None:
    """Worker process: run every (cfg, seq_length) in PAPER_RUNS[gpu] on this GPU.

    One worker owns one GPU end-to-end. Sequential inside the worker;
    parallelism only happens across workers (one OS process per GPU).
    Redirects stdout/stderr to logs/gpu{N}.log so the terminal stays readable.
    """
    log_path = Path("logs") / f"gpu{gpu}.log"
    log_path.parent.mkdir(exist_ok=True)
    sys.stdout = sys.stderr = open(log_path, "w", buffering=1)  # line-buffered

    for cfg, seq_length in PAPER_RUNS[gpu]:
        print(f"[gpu={gpu}] starting {cfg.value} seq={seq_length}", flush=True)
        train_one(
            cfg.value,
            gpu,
            seq_length=seq_length,
            seed=SEED_DEFAULT,
            log_frequency=LOG_FREQUENCY,
            tensorboard=TENSORBOARD,
        )


def run_parallel() -> int:
    """Parent: spawn one mp.Process per GPU with the 'spawn' start method.

    Why mp.Process + spawn (not subprocess re-invoking this file, not fork):
    - 'spawn' boots a fresh interpreter and re-imports the module with
      __name__ == '__mp_main__', so the __main__ guard naturally protects
      the child from running parent code → fork bomb structurally impossible.
    - Fresh interpreter per child = isolated CUDA context, so a crash on
      one GPU doesn't take the others down.
    - 'fork' would inherit CUDA state from the parent, which CUDA forbids.
    """
    ctx = mp.get_context("spawn")
    procs = {gpu: ctx.Process(target=_run_worker, args=(gpu,)) for gpu in PAPER_RUNS}
    for gpu, p in procs.items():
        p.start()
        print(f"[parent] gpu={gpu} pid={p.pid} → logs/gpu{gpu}.log")

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
