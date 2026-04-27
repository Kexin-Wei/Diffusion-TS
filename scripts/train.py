#!/usr/bin/env python3
"""Train Diffusion-TS on benchmark datasets, in-process.

Dismantled version of `python main.py --train …` — inlines main.py's setup
AND unfolds Trainer.__init__ + Trainer.train() so the full training pipeline
is visible top-to-bottom in one file. Only `instantiate_from_config(config['model'])`
stays opaque (the diffusion model itself).

Usage (runs every entry in PAPER_RUNS sequentially):
  ./Diffusion-TS/.venv/bin/python scripts/train.py

To subset the sweep, edit PAPER_RUNS below.

Per-dataset sequence lengths (Yuan & Qiao 2024 — Diffusion-TS paper):
  sines, stocks, fmri   → 24                  (Table 1, unconditional generation)
  etth                  → 24, 64, 128, 256    (Table 1 + Table 3 long-term)
  energy                → 24, 64, 128, 256    (Table 1 + Table 3 long-term)
  mujoco                → 24, 100             (Table 1 + Table 4 imputation)

READING (source this script unfolds):
  main.py:54-95                  — top-level dispatch
  engine/solver.py:25-55, 97-144 — Trainer.__init__ + Trainer.train()
  engine/solver.py:57-66         — Trainer.save dict shape
  Data/build_dataloader.py:5-24  — build_dataloader → {'dataloader', 'dataset'}
  Config/<name>.yaml             — solver + dataloader blocks
"""
from __future__ import annotations

import multiprocessing as mp
import sys
import time
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import torch
from ema_pytorch import EMA
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from engine.logger import Logger
from Data.build_dataloader import build_dataloader
from Utils.io_utils import (
    load_yaml_config,
    seed_everything,
    instantiate_from_config,
    get_model_parameters_info,
)


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

# SMOKE TEST: one short run per GPU, validates parent/worker wiring + log routing.
# Restore the full paper sweep below once this passes.
PAPER_RUNS: dict[int, list[tuple[CONFIGS, int]]] = {
    5: [(CONFIGS.SINES, 24)],
    6: [(CONFIGS.STOCKS, 24)],
    7: [(CONFIGS.MUJOCO, 24)],
}
# Full paper sweep (uncomment when smoke test passes; smoke trim above takes precedence):
# PAPER_RUNS: dict[int, list[tuple[CONFIGS, int]]] = {
#     5: [(CONFIGS.ENERGY, 256), (CONFIGS.ENERGY, 24), (CONFIGS.MUJOCO, 24), (CONFIGS.STOCKS, 24)],
#     6: [(CONFIGS.ENERGY, 128), (CONFIGS.MUJOCO, 100), (CONFIGS.FMRI, 24), (CONFIGS.ETT_H, 64)],
#     7: [(CONFIGS.ETT_H, 256), (CONFIGS.ENERGY, 64), (CONFIGS.ETT_H, 128), (CONFIGS.ETT_H, 24), (CONFIGS.SINES, 24)],
# }2 m


def cycle(dl):
    """Yield batches forever so the step-based training loop never exhausts.

    Unlike `itertools.cycle`, this re-iterates the DataLoader each pass instead
    of caching every batch in memory, and lets shuffling/workers reset normally.
    """
    while True:
        for data in dl:
            yield data


def train_one(
    cfg: str,
    gpu: int,
    seq_length: int = None,
    seed: int = SEED_DEFAULT,
    tensorboard: bool = False,
    output: str = "OUTPUT",
) -> None:
    """Train one dataset end-to-end — inlines main.py setup + unfolds Trainer."""

    config_file = f"Config/{cfg}.yaml"
    config = load_yaml_config(config_file)

    # Resolve seq_length up front (caller override else YAML default), then propagate.
    seq_length = (
        seq_length
        if seq_length is not None
        else config["model"]["params"]["seq_length"]
    )
    config["model"]["params"]["seq_length"] = seq_length
    config["dataloader"]["train_dataset"]["params"]["window"] = seq_length
    config["dataloader"]["test_dataset"]["params"]["window"] = seq_length

    # Centralise checkpoints under ./checkpoints/<cfg>_seq_<N>/ — folder name is
    # always self-describing, no matter whether seq_length came from YAML or caller.
    results_folder = Path("checkpoints") / f"{cfg}_seq_{seq_length}"
    config["solver"]["results_folder"] = results_folder

    save_dir = Path(output).joinpath(f"{cfg}_seq_{seq_length}")
    args = SimpleNamespace(
        name=cfg,
        config_file=config_file,
        output=output,
        save_dir=save_dir,
        tensorboard=tensorboard,
        seed=seed,
        gpu=gpu,
        cudnn_deterministic=False,
    )

    seed_everything(seed)
    torch.cuda.set_device(gpu)

    logger = Logger(args)
    logger.save_config(config)

    model = instantiate_from_config(config["model"]).cuda()

    dataloader_bundle = build_dataloader(config, args)

    # ============ UNFOLDED Trainer.__init__ ============

    base_lr = config["solver"].get("base_lr", 1.0e-4)
    opt = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=base_lr,
        betas=[0.9, 0.96],
    )

    ema_decay = config["solver"]["ema"]["decay"]
    ema_update_every = config["solver"]["ema"]["update_interval"]
    device = model.betas.device
    ema = EMA(model, beta=ema_decay, update_every=ema_update_every).to(device)

    # Inject runtime optimizer into scheduler config — YAML can't reference live objects.
    sc_cfg = config["solver"]["scheduler"]
    sc_cfg["params"]["optimizer"] = opt
    sch = instantiate_from_config(sc_cfg)

    logger.log_info(str(get_model_parameters_info(model)))

    results_folder.mkdir(parents=True, exist_ok=True)

    # `max_epochs` is a misnomer — it's the number of optimizer steps, not epochs.
    train_num_steps = config["solver"]["max_epochs"]
    grad_accum = config["solver"]["gradient_accumulate_every"]
    save_cycle = config["solver"]["save_cycle"]

    # ============ UNFOLDED Trainer.train() ============

    step = 0
    milestone = 0
    dl_iter = cycle(dataloader_bundle["dataloader"])
    tic = time.time()
    logger.log_info(f"{cfg}: start training...", check_primary=False)

    with tqdm(initial=step, total=train_num_steps) as pbar:
        while step < train_num_steps:

            # Gradient accumulation: backward() per mini-batch (not once at the end)
            # so each graph can be freed immediately — that's the memory win.
            # `/ grad_accum` keeps the effective LR correct; `.item()` detaches the log.
            total_loss = 0.0
            for _ in range(grad_accum):
                data = next(dl_iter).to(device)
                loss = model(data, target=data)
                loss = loss / grad_accum
                loss.backward()
                total_loss += loss.item()

            pbar.set_description(f"loss: {total_loss:.6f}")

            clip_grad_norm_(
                model.parameters(), 1.0
            )  # cap global grad norm at 1.0 to avoid spike-driven divergence
            opt.step()  # apply Adam update using the accumulated grads
            sch.step(
                total_loss
            )  # ReduceLROnPlateau-style: needs the loss to detect plateaus
            opt.zero_grad()  # wipe .grad so the next backward starts clean
            step += 1  # one optimizer step completed

            ema.update()  # mix live weights into EMA shadow (honours update_every internally)

            with torch.no_grad():
                if step != 0 and step % save_cycle == 0:
                    # Bump milestone BEFORE saving so the filename matches Trainer.load.
                    milestone += 1
                    ckpt_path = results_folder / f"checkpoint-{milestone}.pt"
                    logger.log_info(f"Saving checkpoint at step {step} to {ckpt_path}")
                    torch.save(
                        {
                            "step": step,
                            "model": model.state_dict(),
                            "ema": ema.state_dict(),
                            "opt": opt.state_dict(),
                        },
                        ckpt_path,
                    )

                if step % LOG_FREQUENCY == 0:
                    logger.add_scalar(
                        tag="train/loss",
                        scalar_value=total_loss,
                        global_step=step,
                    )

            pbar.update(1)

    print("training complete")
    logger.log_info(f"Training done, time: {time.time() - tic:.2f}")


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
