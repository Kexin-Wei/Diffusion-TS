#!/usr/bin/env python3
"""Diffusion-TS trainer on simglucose 3-meal data (threemeal_24h.npy).

Paper hyperparameter ranges (Yuan & Qiao 2024 — "limited hyperparameter tuning"):
  batch_size:        [32, 64, 128]              → dataloader.batch_size
  n_heads:           [4, 8]                     → model.params.n_heads
  d_model:           [32, 64, 96, 128]          → model.params.d_model     (basic dim)
  timesteps:         [50, 200, 500, 1000]       → model.params.timesteps   (diffusion steps)
  guidance strength: [1, 1e-1, 5e-2, 1e-2, 1e-3] → test_dataset.coefficient
        (sampling-time only — irrelevant for the training loop here; sweep it later
         when running imputation/forecasting against a frozen checkpoint.)

Defaults in Config/simglucose_3m.yaml sit inside the paper grid:
  batch_size=64, n_heads=4, d_model=96, timesteps=1000.

Workflow (graduated validation):
  1. SMOKE=True  → Config/simglucose_3m_smoke.yaml, max_epochs=200, ~minutes.
  2. Inspect logs/loss curve; flip SMOKE=False for the full 15k-step run.
  3. To sweep hyperparameters, fork this script into a multi-GPU orchestrator
     patterned after scripts/train_reproduce.py (one cfg per (gpu, override) pair).

Usage (must run from Diffusion-TS/ with the local Py3.8 venv):
  cd Diffusion-TS && .venv/bin/python scripts/train_simglucose.py
"""
from __future__ import annotations

from train_one import train_one


SMOKE = False  # flip to False after the smoke run looks healthy
GPU = 0


if __name__ == "__main__":
    cfg = "simglucose_3m_smoke" if SMOKE else "simglucose_3m"
    train_one(
        cfg=cfg,
        gpu=GPU,
        seq_length=24,
        seed=12345,
        log_frequency=100,
        tensorboard=False,
    )
