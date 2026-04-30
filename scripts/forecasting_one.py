#!/usr/bin/env python3
"""Pure single-dataset Diffusion-TS forecaster — importable library.

Dismantled version of `python main.py --sample 1 --mode predict …` —
inlines main.py's setup AND unfolds Trainer.__init__ + Trainer.load +
Trainer.restore so the full conditional-sampling pipeline is visible
top-to-bottom in one file. Only `instantiate_from_config(config['model'])`
stays opaque (the diffusion model itself).

This module exposes `forecast_one(...)` plus the small helpers it needs. It
does NOT orchestrate multi-GPU runs — that lives in `forecasting_reproduce.py`.

Outputs (per pred_len L), under OUTPUT/<cfg>/:
  ddpm_predict_<cfg>_h{L}.npy   — samples, original units
  reals_<cfg>_h{L}.npy          — ground truth, original units
  masks_<cfg>_h{L}.npy          — bool mask (True=observed, False=future)
  forecast_<cfg>_h{L}.png       — Tutorial_2-style history/GT/prediction plot
  metrics_<cfg>.json            — per-horizon MSE/MAE on the future tail

READING (source this script unfolds):
  main.py:54-95                  — top-level dispatch
  main.py:77-87                  — sample==1 branch (load → restore → save)
  Data/build_dataloader.py:26-48 — build_dataloader_cond (predict_length branch)
  engine/solver.py:25-55, 77-86  — Trainer.__init__ + Trainer.load
  engine/solver.py:162-188       — Trainer.restore (per-batch sample_infill loop)
  Models/interpretable_diffusion/* — sample_infill / fast_sample_infill (opaque)
  Config/<name>.yaml             — solver + dataloader blocks
  Tutorial_2.ipynb               — forecasting metric + plot reference
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
SEQ_LENGTH_DEFAULT = 64


def build_cond_dataloader(config: dict, args: SimpleNamespace):
    """Unfolded Data.build_dataloader.build_dataloader_cond — predict-mode
    branch only. Sets predict_length on the test dataset (which zeroes the
    last `pred_len` mask positions, i.e. the future), then wraps in a
    deterministic, no-shuffle DataLoader so output rows align with reals."""
    batch_size = config["dataloader"]["sample_size"]
    config["dataloader"]["test_dataset"]["params"]["output_dir"] = args.save_dir
    config["dataloader"]["test_dataset"]["params"]["predict_length"] = args.pred_len

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
    """Forecast one dataset end-to-end — inlines main.py setup + unfolds Trainer."""

    config_file = f"Config/{cfg}.yaml"
    run_name = f"{cfg}_seq_{seq_length}"
    save_dir = os.path.join(output, run_name)
    args = SimpleNamespace(
        name=run_name,
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

    seed_everything(seed)
    torch.cuda.set_device(gpu)

    config = load_yaml_config(config_file)

    # Propagate seq_length into model + dataloader (mirrors train_one.py:99-107).
    # Required so the model arch + dataset window match the trained checkpoint;
    # otherwise the state_dict load below silently mismatches conv kernels.
    config["model"]["params"]["seq_length"] = seq_length
    config["dataloader"]["test_dataset"]["params"]["window"] = seq_length
    if "train_dataset" in config["dataloader"]:
        config["dataloader"]["train_dataset"]["params"]["window"] = seq_length

    # Drop YAML kernel/padding so the model auto-picks based on (n_feat, seq_length).
    # Required when seq_length is overridden — a YAML pair tuned for one length
    # silently violates p=(k-1)//2 (or just diverges from the heuristic) at another.
    config["model"]["params"].pop("kernel_size", None)
    config["model"]["params"].pop("padding_size", None)

    results_folder = get_results_folder(config, cfg, seq_length)
    config["solver"]["results_folder"] = results_folder

    logger = Logger(args)
    logger.save_config(config)

    model = instantiate_from_config(config["model"]).cuda()
    device = model.betas.device

    # ============ UNFOLDED Trainer.__init__ + Trainer.load ============

    # Not training, but Trainer.restore samples from `ema.ema_model`
    # (solver.py:176, 179) — the EMA wrapper has to exist to receive the
    # EMA weights from the checkpoint.
    ema_decay = config["solver"]["ema"]["decay"]
    ema_update_every = config["solver"]["ema"]["update_interval"]
    ema = EMA(model, beta=ema_decay, update_every=ema_update_every).to(device)

    ckpt_path = results_folder / f"checkpoint-{milestone}.pt"
    data = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(data["model"])
    ema.load_state_dict(data["ema"])

    # Three test-time sampling params from the YAML (Yuan & Qiao 2024 §B.3 + Table 6).
    # Constant across horizons, so read once outside the loop.
    coef = config["dataloader"]["test_dataset"]["coefficient"]
    stepsize = config["dataloader"]["test_dataset"]["step_size"]
    sampling_steps = config["dataloader"]["test_dataset"]["sampling_steps"]
    model_kwargs = {"coef": coef, "learning_rate": stepsize}

    # ============ UNFOLDED Trainer.restore (per horizon) ============

    metrics_all: dict[str, dict[str, float]] = {}
    for L in pred_lenth:
        # Mask is fixed at dataset construction time, so a fresh dataset+loader
        # is needed per horizon.
        args.pred_len = L
        dataloader, dataset = build_cond_dataloader(config, args)
        window, var_num = dataset.window, dataset.var_num

        logger.log_info(f"[{run_name}] pred_len={L}: begin sampling...")
        samples_norm = np.empty([0, window, var_num])
        reals_norm = np.empty([0, window, var_num])
        masks_bool = np.empty([0, window, var_num], dtype=bool)

        for x, t_m in dataloader:
            x, t_m = x.to(device), t_m.to(device)
            # sample_infill is the full DDPM loop; fast_sample_infill is DDIM —
            # picked by whether sampling_steps matches the trained num_timesteps.
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

        # `dataset.unnormalize` does both [-1,1]→[0,1] and scaler.inverse,
        # matching Tutorial_2.ipynb cell 8 — i.e. metrics + plots are in the
        # dataset's natural units, not the diffusion-model latent scale.
        samples_orig = dataset.unnormalize(samples_norm)
        reals_orig = dataset.unnormalize(reals_norm)

        tag = f"h{L}"
        out_path = os.path.join(save_dir, f"ddpm_predict_{run_name}_{tag}.npy")
        reals_path = os.path.join(save_dir, f"reals_{run_name}_{tag}.npy")
        masks_path = os.path.join(save_dir, f"masks_{run_name}_{tag}.npy")
        np.save(out_path, samples_orig)
        np.save(reals_path, reals_orig)
        np.save(masks_path, masks_bool)
        logger.log_info(
            f"[{run_name}] pred_len={L} -> {out_path} (shape={samples_orig.shape})"
        )

        # Future-tail (mask==0) MSE/MAE — Tutorial_2.ipynb form. MSE is the
        # paper's reported metric (Table 5 + §C.3); MAE added as the standard
        # forecasting companion.
        target = ~masks_bool
        diff = samples_orig[target] - reals_orig[target]
        mse = float((diff ** 2).mean())
        mae = float(np.abs(diff).mean())
        metrics_all[tag] = {
            "pred_len": L,
            "mse": mse,
            "mae": mae,
            "n_target_points": int(target.sum()),
        }
        logger.log_info(f"[{run_name}] pred_len={L}: MSE={mse:.6f}  MAE={mae:.6f}")

        plot_path = os.path.join(save_dir, f"forecast_{run_name}_{tag}.png")
        _plot_forecast(
            samples_orig, reals_orig,
            pred_len=L,
            save_path=plot_path,
            title=f"{run_name}  pred_len={L}",
            n_samples=n_plot_samples,
            n_features=n_plot_features,
        )
        logger.log_info(f"[{run_name}] pred_len={L} -> {plot_path}")

    metrics_path = os.path.join(save_dir, f"metrics_{run_name}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_all, f, indent=2)
    logger.log_info(f"[{run_name}] metrics -> {metrics_path}")


def _plot_forecast(samples, reals, *, pred_len, save_path, title,
                   n_samples=5, n_features=4):
    """Tutorial_2.ipynb cell 9 style — per panel: history (cyan) up to
    t = window-pred_len, then ground-truth tail (green) and prediction tail
    (red). Tails overlap by one point so the lines visually connect at the
    history/future boundary."""
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
