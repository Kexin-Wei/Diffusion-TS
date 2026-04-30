#!/usr/bin/env python3
"""Run Diffusion-TS forecasting on one benchmark dataset, in-process.

Dismantled version of `python main.py --sample 1 --mode predict …` —
inlines main.py's setup AND unfolds Trainer.__init__ + Trainer.load +
Trainer.restore so the full conditional-sampling pipeline is visible
top-to-bottom. The diffusion model itself stays opaque
(`instantiate_from_config(config['model'])`).

Run from the Diffusion-TS directory:
  ./Diffusion-TS/.venv/bin/python scripts/forecasting_one.py   # uses __main__ defaults

This module exposes `forecast_one(...)` plus the small helpers it needs. It does
NOT orchestrate multi-GPU runs — that lives in `forecasting_reproduce.py`.

Outputs (per pred_len L), all under OUTPUT/<cfg>/:
  ddpm_predict_<cfg>_h{L}.npy   — samples, original units
  reals_<cfg>_h{L}.npy          — ground truth, original units
  masks_<cfg>_h{L}.npy          — bool mask (True=observed, False=future)
  forecast_<cfg>_h{L}.png       — Tutorial_2-style history/GT/prediction plot
  metrics_<cfg>.json            — {'h{L}': {pred_len, mse, mae, n_target_points}}

READING (the source this script unfolds):
  main.py:54-95              — top-level dispatch
  main.py:77-87              — sample==1 conditional branch (load → restore → save)
  Data/build_dataloader.py:26-48 — build_dataloader_cond:
                                injects predict_length for mode='predict'
                                (the dataset's mask zeroes out the last `pred_len`
                                 timesteps, i.e. the future)
  engine/solver.py:25-55     — Trainer.__init__: builds EMA wrapper
  engine/solver.py:77-86     — Trainer.load: torch.load + state_dict restore
  engine/solver.py:162-188   — Trainer.restore: per-batch sample_infill loop
                                (note: predict reuses sample_infill — only the
                                 mask shape changes vs imputation)
  Models/interpretable_diffusion/* — sample_infill / fast_sample_infill (opaque)
  Config/etth.yaml:dataloader.test_dataset.{coefficient,step_size,sampling_steps}
                                 — Yuan & Qiao 2024 §B.3 + Table 6
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from ema_pytorch import EMA

from engine.logger import Logger
from Utils.io_utils import (
    load_yaml_config,
    seed_everything,
    instantiate_from_config,
)

from scripts.utils import get_results_folder

PRED_LENS_DEFAULT: tuple[int, ...] = (24,)
MILESTONE_DEFAULT = 10
SEED_DEFAULT = 12345
SEQ_LENGTH_DEFAULT = 24


def build_cond_dataloader(config: dict, args: SimpleNamespace):
    """Unfolded Data.build_dataloader.build_dataloader_cond — only the
    mode='predict' branch (build_dataloader.py:26-48). Returns
    (dataloader, dataset)."""
    # STEP A: pick batch size + inject save_dir into the dataset params.
    batch_size = config["dataloader"]["sample_size"]
    config["dataloader"]["test_dataset"]["params"]["output_dir"] = args.save_dir

    # STEP B: dispatch on mode — for predict we set predict_length; the
    # dataset zeroes the last `pred_len` timesteps in the mask
    # (build_dataloader.py:31-32).
    config["dataloader"]["test_dataset"]["params"]["predict_length"] = args.pred_len

    # STEP C: instantiate dataset, wrap in DataLoader (no shuffle so
    # outputs line up with reals; drop_last=False so we don't lose tail samples).
    dataset = instantiate_from_config(config["dataloader"]["test_dataset"])
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


def forecast_one(
    cfg: str,
    gpu: int,
    seed: int,
    milestone: int,
    pred_lenth: list[int],
    seq_length: int,
    output: str = "OUTPUT",
    n_plot_samples: int = 5,
    n_plot_features: int = 4,
) -> None:
    """Forecasting pipeline, fully unfolded. One model load, then one
    sampling sweep per prediction horizon."""

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
        mode="predict",
        missing_ratio=0.0,
        pred_len=0,  # overwritten per-horizon in the sweep below
    )

    # ============================================================
    # STEP 2: determinism + GPU pin (main.py:57-61).
    # ============================================================
    seed_everything(seed)
    torch.cuda.set_device(gpu)

    # ============================================================
    # STEP 3: load YAML config (main.py:63), then propagate seq_length
    # like train_one.py:99-107 — model arch + dataloader windows must
    # match the checkpoint trained at this seq_length, otherwise the
    # state_dict load below silently mismatches conv kernels.
    # ============================================================
    config = load_yaml_config(config_file)
    config["model"]["params"]["seq_length"] = seq_length
    config["dataloader"]["test_dataset"]["params"]["window"] = seq_length
    if "train_dataset" in config["dataloader"]:
        config["dataloader"]["train_dataset"]["params"]["window"] = seq_length
    config["model"]["params"].pop("kernel_size", None)
    config["model"]["params"].pop("padding_size", None)

    results_folder = get_results_folder(config, cfg, seq_length)
    config["solver"]["results_folder"] = results_folder

    # ============================================================
    # STEP 4: Logger.
    # ============================================================
    logger = Logger(args)
    logger.save_config(config)

    # ============================================================
    # STEP 5: build the diffusion model (only opaque step).
    # ============================================================
    model = instantiate_from_config(config["model"]).cuda()
    device = model.betas.device

    # ============================================================
    # STEP 6 (TODO): build the EMA wrapper.
    # Same as imputation.py — Trainer.restore samples from `ema.ema_model`,
    # so EMA must exist to receive the EMA weights from the checkpoint.
    #
    # Hint:
    #   ema_decay = config['solver']['ema']['decay']
    #   ema_update_every = config['solver']['ema']['update_interval']
    #   ema = EMA(model, beta=ema_decay, update_every=ema_update_every).to(device)
    # ============================================================
    ema_decay = config["solver"]["ema"]["decay"]
    ema_update_every = config["solver"]["ema"]["update_interval"]
    ema = EMA(model, beta=ema_decay, update_every=ema_update_every).to(device)

    # ============================================================
    # STEP 7 (TODO): load the checkpoint.
    # See solver.py:77-86. Same path convention as imputation.py.
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
    ckpt_dir = results_folder
    ckpt_path = ckpt_dir / f"checkpoint-{milestone}.pt"
    data = torch.load(
        str(ckpt_path),
        map_location=device,
    )
    model.load_state_dict(data["model"])
    ema.load_state_dict(data["ema"])

    # ============================================================
    # STEP 8: read the three test-time sampling params (main.py:80-82).
    # These don't change per-horizon, so read once.
    # ============================================================
    coef = config["dataloader"]["test_dataset"]["coefficient"]
    stepsize = config["dataloader"]["test_dataset"]["step_size"]
    sampling_steps = config["dataloader"]["test_dataset"]["sampling_steps"]
    model_kwargs = {"coef": coef, "learning_rate": stepsize}

    # ============================================================
    # STEP 9: per-horizon sweep.
    # For each pred_len: rebuild the cond dataloader (mask is set at
    # dataset construction time), run the sampling loop, save .npy +
    # metrics + plot.
    # ============================================================
    metrics_all: dict[str, dict[str, float]] = {}
    for L in pred_lenth:
        args.pred_len = L
        dataloader, dataset = build_cond_dataloader(config, args)
        window, var_num = dataset.window, dataset.var_num

        # --------------------------------------------------------
        # ====== UNFOLDED Trainer.restore (solver.py:162-188) =====
        # --------------------------------------------------------
        logger.log_info(f"[{cfg}] pred_len={L}: begin sampling...")
        samples_norm = np.empty([0, window, var_num])
        reals_norm = np.empty([0, window, var_num])
        masks_bool = np.empty([0, window, var_num], dtype=bool)

        for x, t_m in dataloader:
            x, t_m = x.to(device), t_m.to(device)
            if sampling_steps == model.num_timesteps:
                sample = ema.ema_model.sample_infill(
                    shape=x.shape,
                    target=x * t_m,
                    partial_mask=t_m,
                    model_kwargs=model_kwargs,
                )
            else:
                sample = ema.ema_model.fast_sample_infill(
                    shape=x.shape,
                    target=x * t_m,
                    partial_mask=t_m,
                    model_kwargs=model_kwargs,
                    sampling_timesteps=sampling_steps,
                )
            samples_norm = np.row_stack([samples_norm, sample.detach().cpu().numpy()])
            reals_norm = np.row_stack([reals_norm, x.detach().cpu().numpy()])
            masks_bool = np.row_stack([masks_bool, t_m.detach().cpu().numpy().astype(bool)])

        # --------------------------------------------------------
        # STEP 11: invert the dataset's normalisation back to original
        # units (matches Tutorial_2.ipynb cell 8: scaler.inverse(unnorm(...))).
        # `dataset.unnormalize` does both [-1,1]→[0,1] and scaler.inverse.
        # --------------------------------------------------------
        samples_orig = dataset.unnormalize(samples_norm)
        reals_orig = dataset.unnormalize(reals_norm)

        # --------------------------------------------------------
        # STEP 12: save samples / reals / masks with per-horizon tag.
        # --------------------------------------------------------
        tag = f"h{L}"
        out_path = os.path.join(save_dir, f"ddpm_predict_{cfg}_{tag}.npy")
        reals_path = os.path.join(save_dir, f"reals_{cfg}_{tag}.npy")
        masks_path = os.path.join(save_dir, f"masks_{cfg}_{tag}.npy")
        np.save(out_path, samples_orig)
        np.save(reals_path, reals_orig)
        np.save(masks_path, masks_bool)
        logger.log_info(
            f"[{cfg}] pred_len={L} -> {out_path} (shape={samples_orig.shape})"
        )

        # --------------------------------------------------------
        # STEP 13: metrics on the future tail (mask==0).
        # Mirrors Tutorial_2.ipynb: mse = mean_squared_error(sample[~mask], real[~mask]).
        # MSE is what the paper reports (Table 5 + §C.3); MAE added for
        # convenience as it's the other standard forecasting metric.
        # --------------------------------------------------------
        target = ~masks_bool
        diff = samples_orig[target] - reals_orig[target]
        mse = float((diff ** 2).mean())
        mae = float(np.abs(diff).mean())
        metrics_all[tag] = {"pred_len": L, "mse": mse, "mae": mae,
                            "n_target_points": int(target.sum())}
        logger.log_info(f"[{cfg}] pred_len={L}: MSE={mse:.6f}  MAE={mae:.6f}")

        # --------------------------------------------------------
        # STEP 14: Tutorial_2-style plot — history (cyan) + ground-truth
        # tail (green) + prediction tail (red), first N samples × first
        # M features.
        # --------------------------------------------------------
        plot_path = os.path.join(save_dir, f"forecast_{cfg}_{tag}.png")
        _plot_forecast(
            samples_orig, reals_orig, pred_len=L,
            save_path=plot_path,
            title=f"{cfg}  pred_len={L}",
            n_samples=n_plot_samples,
            n_features=n_plot_features,
        )
        logger.log_info(f"[{cfg}] pred_len={L} -> {plot_path}")

    # ============================================================
    # STEP 15: dump the full metric table once.
    # ============================================================
    metrics_path = os.path.join(save_dir, f"metrics_{cfg}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_all, f, indent=2)
    logger.log_info(f"[{cfg}] metrics -> {metrics_path}")


def _plot_forecast(samples, reals, *, pred_len, save_path, title,
                    n_samples=5, n_features=4):
    """Tutorial_2.ipynb cell 9 style: per panel, plot history (cyan) up
    to t = window-pred_len, then ground-truth tail (green) and prediction
    tail (red) overlapping by one point so the lines visually connect.

    Plots a (n_samples × n_features) grid from the first samples/features
    of `samples` and `reals` (already in original units)."""
    seq_length = reals.shape[1]
    history_end = seq_length - pred_len  # first index of the future tail

    n_samples = min(n_samples, reals.shape[0])
    n_features = min(n_features, reals.shape[2])

    fig, axes = plt.subplots(
        n_samples, n_features,
        figsize=(4 * n_features, 2.0 * n_samples),
        squeeze=False,
    )
    t = np.arange(seq_length)
    hist_t = t[:history_end]
    fut_t = t[history_end - 1:]  # overlap by 1 so the line connects

    for s in range(n_samples):
        for f in range(n_features):
            ax = axes[s, f]
            ax.plot(hist_t, reals[s, :history_end, f], color="c", label="History")
            ax.plot(fut_t, reals[s, history_end - 1:, f], color="g", label="Ground Truth")
            ax.plot(fut_t, samples[s, history_end - 1:, f], color="r", label="Prediction")
            ax.axvline(history_end - 1, color="grey", linewidth=0.5, linestyle="--")
            if s == 0 and f == 0:
                ax.legend(loc="best", fontsize=7)
            if f == 0:
                ax.set_ylabel(f"sample {s}")
            if s == 0:
                ax.set_title(f"feat {f}")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    forecast_one(
        cfg="etth",
        gpu=0,
        seed=SEED_DEFAULT,
        milestone=MILESTONE_DEFAULT,
        pred_lenth=list(PRED_LENS_DEFAULT),
        seq_length=SEQ_LENGTH_DEFAULT,
    )
