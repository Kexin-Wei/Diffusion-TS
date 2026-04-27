#!/usr/bin/env python3
"""Run Diffusion-TS imputation on one benchmark dataset, in-process.

Dismantled version of `python main.py --sample 1 --mode infill …` —
inlines main.py's setup AND unfolds Trainer.__init__ + Trainer.load +
Trainer.restore so the full conditional-sampling pipeline is visible
top-to-bottom. The diffusion model itself stays opaque
(`instantiate_from_config(config['model'])`).

Run from the Diffusion-TS directory:
  python scripts/imputation.py --only etth                       # default ratios
  python scripts/imputation.py --only etth --ratios 0.7 0.8 0.9
  python scripts/imputation.py --only etth --milestone 10
  python scripts/imputation.py --only etth --gpu 0

In-process can only run one dataset per invocation. Output:
  OUTPUT/<cfg>/ddpm_infill_<cfg>_m{70,80,90}.npy

READING (the source this script unfolds):
  main.py:54-95              — top-level dispatch
  main.py:77-87              — sample==1 conditional branch (load → restore → save)
  Data/build_dataloader.py:26-48 — build_dataloader_cond:
                                injects missing_ratio for mode='infill'
                                or predict_length for mode='predict'
  engine/solver.py:25-55     — Trainer.__init__: builds EMA wrapper
                                (we need ema.ema_model to sample)
  engine/solver.py:77-86     — Trainer.load: torch.load + state_dict restore
  engine/solver.py:162-188   — Trainer.restore: per-batch sample_infill loop
  Models/interpretable_diffusion/* — sample_infill / fast_sample_infill (opaque)
  Config/etth.yaml:dataloader.test_dataset.{coefficient,step_size,sampling_steps}
                                 — Yuan & Qiao 2024 §B.3 + Table 6
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# Make the Diffusion-TS root importable when run directly.
_DTS = Path(__file__).resolve().parent.parent
if str(_DTS) not in sys.path:
    sys.path.insert(0, str(_DTS))

import numpy as np
import torch
from ema_pytorch import EMA

from engine.logger import Logger
from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one
from Utils.io_utils import (
    load_yaml_config,
    seed_everything,
    instantiate_from_config,
)


GPU_OF: dict[str, int] = {
    "sines": 0, "stocks": 3, "etth": 4,
    "energy": 5, "fmri": 6, "mujoco": 2,
}
MISSING_RATIOS_DEFAULT: tuple[float, ...] = (0.7, 0.8, 0.9)
MILESTONE_DEFAULT = 10
SEED_DEFAULT = 12345


def build_cond_dataloader(config: dict, args: SimpleNamespace):
    """Unfolded Data.build_dataloader.build_dataloader_cond — only the
    mode='infill' branch (build_dataloader.py:26-48). Returns
    (dataloader, dataset)."""
    # STEP A: pick batch size + inject save_dir into the dataset params.
    batch_size = config['dataloader']['sample_size']
    config['dataloader']['test_dataset']['params']['output_dir'] = args.save_dir

    # STEP B: dispatch on mode — for infill we set missing_ratio; the
    # dataset's __init__ uses it to build a random mask (build_dataloader.py:29-30).
    config['dataloader']['test_dataset']['params']['missing_ratio'] = args.missing_ratio

    # STEP C: instantiate dataset, wrap in DataLoader (no shuffle so
    # outputs line up with reals; drop_last=False so we don't lose tail samples).
    dataset = instantiate_from_config(config['dataloader']['test_dataset'])
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=None,
        drop_last=False,
    )
    return dataloader, dataset


def impute_one(
    cfg: str,
    gpu: int,
    seed: int,
    milestone: int,
    ratios: list[float],
    output: str = "OUTPUT",
) -> None:
    """Imputation pipeline, fully unfolded. One model load, then one
    sampling sweep per missing ratio."""

    # ============================================================
    # STEP 1: args namespace expected by Logger / dataloader builder.
    # ============================================================
    config_file = f"Config/{cfg}.yaml"
    save_dir = os.path.join(output, cfg)
    args = SimpleNamespace(
        name=cfg,
        config_file=config_file,
        output=output,
        save_dir=save_dir,
        tensorboard=False,
        seed=seed,
        gpu=gpu,
        mode='infill',
        missing_ratio=0.0,   # overwritten per-ratio below
        pred_len=0,
    )

    # ============================================================
    # STEP 2: determinism + GPU pin (main.py:57-61).
    # ============================================================
    seed_everything(seed)
    torch.cuda.set_device(gpu)

    # ============================================================
    # STEP 3: load YAML config (main.py:63).
    # ============================================================
    config = load_yaml_config(config_file)

    # ============================================================
    # STEP 4: Logger (writes save_dir/configs/*, save_dir/logs/*).
    # ============================================================
    logger = Logger(args)
    logger.save_config(config)

    # ============================================================
    # STEP 5: build the diffusion model (only opaque step).
    # ============================================================
    model = instantiate_from_config(config['model']).cuda()
    device = model.betas.device

    # ============================================================
    # STEP 6 (TODO): build the EMA wrapper.
    # We're not training, but Trainer.restore samples from `ema.ema_model`
    # (solver.py:176, 179), so the EMA wrapper has to exist to receive
    # the EMA weights from the checkpoint.
    #
    # Hint:
    #   ema_decay = config['solver']['ema']['decay']
    #   ema_update_every = config['solver']['ema']['update_interval']
    #   ema = EMA(model, beta=ema_decay, update_every=ema_update_every).to(device)
    # ============================================================
    raise NotImplementedError("STEP 6: build EMA wrapper around model")

    # ============================================================
    # STEP 7 (TODO): load the checkpoint.
    # See solver.py:77-86 for what Trainer.load does. Note the path
    # convention: results_folder gets `_{seq_length}` appended.
    #
    # Hint:
    #   ckpt_dir = Path(config['solver']['results_folder'] + f"_{model.seq_length}")
    #   ckpt_path = ckpt_dir / f"checkpoint-{milestone}.pt"
    #   if not ckpt_path.exists():
    #       raise FileNotFoundError(f"no checkpoint at {ckpt_path} — train first")
    #   data = torch.load(str(ckpt_path), map_location=device)
    #   model.load_state_dict(data['model'])
    #   ema.load_state_dict(data['ema'])
    # ============================================================
    raise NotImplementedError("STEP 7: torch.load checkpoint, restore model + ema state_dicts")

    # ============================================================
    # STEP 8: read the three test-time sampling params from config
    # (main.py:80-82). These are paper params from Yuan & Qiao 2024
    # §B.3 + Table 6.
    # ============================================================
    coef = config['dataloader']['test_dataset']['coefficient']
    stepsize = config['dataloader']['test_dataset']['step_size']
    sampling_steps = config['dataloader']['test_dataset']['sampling_steps']
    model_kwargs = {'coef': coef, 'learning_rate': stepsize}

    # ============================================================
    # STEP 9: per-ratio sweep.
    # For each ratio: rebuild the cond dataloader (the mask is fixed
    # at dataset construction time), run the sampling loop, save .npy.
    # ============================================================
    for r in ratios:
        args.missing_ratio = r
        dataloader, dataset = build_cond_dataloader(config, args)
        window, var_num = dataset.window, dataset.var_num

        # --------------------------------------------------------
        # ====== UNFOLDED Trainer.restore (solver.py:162-188) =====
        # --------------------------------------------------------

        # STEP 10 (TODO): the per-batch sampling loop.
        # The dataset yields (x, t_m) where t_m is the mask
        # (1=observed, 0=missing). We feed `target=x*t_m` and
        # `partial_mask=t_m` to the sampler.
        #
        # Hint:
        #   samples = np.empty([0, window, var_num])
        #   for x, t_m in dataloader:
        #       x, t_m = x.to(device), t_m.to(device)
        #       if sampling_steps == model.num_timesteps:
        #           sample = ema.ema_model.sample_infill(
        #               shape=x.shape, target=x * t_m,
        #               partial_mask=t_m, model_kwargs=model_kwargs,
        #           )
        #       else:
        #           sample = ema.ema_model.fast_sample_infill(
        #               shape=x.shape, target=x * t_m,
        #               partial_mask=t_m, model_kwargs=model_kwargs,
        #               sampling_timesteps=sampling_steps,
        #           )
        #       samples = np.row_stack([samples, sample.detach().cpu().numpy()])
        # --------------------------------------------------------
        raise NotImplementedError("STEP 10: per-batch sampling loop over the cond dataloader")

        # --------------------------------------------------------
        # STEP 11: rescale [-1,1] → [0,1] when the dataset normalised
        # to neg_one_to_one (main.py:84-86).
        # --------------------------------------------------------
        if dataset.auto_norm:
            samples = unnormalize_to_zero_to_one(samples)

        # --------------------------------------------------------
        # STEP 12: save with a per-ratio tag so different ratios
        # don't clobber each other.
        # --------------------------------------------------------
        tag = f"m{int(round(r * 100))}"
        out_path = os.path.join(save_dir, f"ddpm_infill_{cfg}_{tag}.npy")
        np.save(out_path, samples)
        logger.log_info(f"[{cfg}] ratio={r} -> {out_path} (shape={samples.shape})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--only", metavar="DATASET", required=True,
        help=f"dataset to impute; one of: {', '.join(GPU_OF)}",
    )
    ap.add_argument(
        "--ratios", type=float, nargs="+", default=list(MISSING_RATIOS_DEFAULT),
        metavar="R", help="missing ratios to sweep (default: 0.7 0.8 0.9)",
    )
    ap.add_argument(
        "--milestone", type=int, default=MILESTONE_DEFAULT,
        help="checkpoint milestone to load (default: 10 = final)",
    )
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--gpu", type=int, default=None,
                    help="override GPU_OF[<dataset>]")
    args = ap.parse_args()

    if args.only not in GPU_OF:
        print(f"error: unknown dataset {args.only!r}; choices: {', '.join(GPU_OF)}",
              file=sys.stderr)
        return 2

    gpu = args.gpu if args.gpu is not None else GPU_OF[args.only]
    impute_one(args.only, gpu, args.seed, args.milestone, args.ratios)
    return 0


if __name__ == "__main__":
    sys.exit(main())
