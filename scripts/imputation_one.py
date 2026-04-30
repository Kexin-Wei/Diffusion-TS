#!/usr/bin/env python3
"""Pure single-dataset Diffusion-TS imputer — importable library.

Dismantled version of `python main.py --sample 1 --mode infill …` —
inlines main.py's setup AND unfolds Trainer.__init__ + Trainer.load +
Trainer.restore so the full conditional-sampling pipeline is visible
top-to-bottom in one file. Only `instantiate_from_config(config['model'])`
stays opaque (the diffusion model itself).

This module exposes `impute_one(...)` plus the small helpers it needs. It
does NOT orchestrate multi-GPU runs — that lives in `imputation_reproduce.py`.

Outputs (per missing ratio r), under OUTPUT/<cfg>_seq_<N>/:
  ddpm_infill_<run>_m{int(r*100)}.npy   — samples, original units
  reals_<run>_m{int(r*100)}.npy         — ground truth, original units
  masks_<run>_m{int(r*100)}.npy         — bool mask (True=observed, False=target)
  impute_<run>_m{int(r*100)}.png        — Tutorial_1-style observed/target/imputation plot
  metrics_<run>.json                    — per-ratio MSE/MAE on masked positions

READING (source this script unfolds):
  main.py:54-95                  — top-level dispatch
  main.py:77-87                  — sample==1 branch (load → restore → save)
  Data/build_dataloader.py:26-48 — build_dataloader_cond (missing_ratio branch)
  engine/solver.py:25-55, 77-86  — Trainer.__init__ + Trainer.load
  engine/solver.py:162-188       — Trainer.restore (per-batch sample_infill loop)
  Models/interpretable_diffusion/* — sample_infill / fast_sample_infill (opaque)
  Config/<name>.yaml             — solver + dataloader blocks
  Tutorial_1.ipynb               — imputation marker style reference
"""
from __future__ import annotations

import json
import os
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


MISSING_RATIOS_DEFAULT: tuple[float, ...] = (0.7, 0.8, 0.9)
MILESTONE_DEFAULT = 10
SEED_DEFAULT = 12345
SEQ_LENGTH_DEFAULT = 64


def build_cond_dataloader(config: dict, args: SimpleNamespace):
    """Unfolded Data.build_dataloader.build_dataloader_cond — infill-mode
    branch only. Sets missing_ratio on the test dataset (which builds a
    random mask), then wraps in a deterministic, no-shuffle DataLoader so
    output rows align with reals."""
    batch_size = config["dataloader"]["sample_size"]
    config["dataloader"]["test_dataset"]["params"]["output_dir"] = args.save_dir
    config["dataloader"]["test_dataset"]["params"]["missing_ratio"] = args.missing_ratio

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


def impute_one(
    cfg: str,
    gpu: int,
    seed: int,
    milestone: int,
    ratios: list[float],
    seq_length: int,
    output: str = "OUTPUT",
    n_plot_samples: int = 5,
    n_plot_features: int = 4,
) -> None:
    """Impute one dataset end-to-end — inlines main.py setup + unfolds Trainer."""

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
        mode="infill",
        missing_ratio=0.0,  # overwritten per-ratio in the sweep below
        pred_len=0,
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
    # Constant across ratios, so read once outside the loop.
    coef = config["dataloader"]["test_dataset"]["coefficient"]
    stepsize = config["dataloader"]["test_dataset"]["step_size"]
    sampling_steps = config["dataloader"]["test_dataset"]["sampling_steps"]
    model_kwargs = {"coef": coef, "learning_rate": stepsize}

    # ============ UNFOLDED Trainer.restore (per ratio) ============

    metrics_all: dict[str, dict[str, float]] = {}
    for r in ratios:
        # Mask is fixed at dataset construction time, so a fresh dataset+loader
        # is needed per ratio.
        args.missing_ratio = r
        dataloader, dataset = build_cond_dataloader(config, args)
        window, var_num = dataset.window, dataset.var_num

        logger.log_info(f"[{run_name}] ratio={r}: begin sampling...")
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

        # `dataset.unnormalize` does both [-1,1]→[0,1] and scaler.inverse — same
        # convention used by forecasting_one. Metrics + plots live in the
        # dataset's natural units, not the diffusion-model latent scale.
        samples_orig = dataset.unnormalize(samples_norm)
        reals_orig = dataset.unnormalize(reals_norm)

        tag = f"m{int(round(r * 100))}"
        out_path = os.path.join(save_dir, f"ddpm_infill_{run_name}_{tag}.npy")
        reals_path = os.path.join(save_dir, f"reals_{run_name}_{tag}.npy")
        masks_path = os.path.join(save_dir, f"masks_{run_name}_{tag}.npy")
        np.save(out_path, samples_orig)
        np.save(reals_path, reals_orig)
        np.save(masks_path, masks_bool)
        logger.log_info(
            f"[{run_name}] ratio={r} -> {out_path} (shape={samples_orig.shape})"
        )

        # Masked-position (mask==0) MSE/MAE — the standard imputation form.
        # MSE matches Yuan & Qiao 2024 Table 4; MAE added as the standard companion.
        target = ~masks_bool
        diff = samples_orig[target] - reals_orig[target]
        mse = float((diff ** 2).mean())
        mae = float(np.abs(diff).mean())
        metrics_all[tag] = {
            "missing_ratio": r,
            "mse": mse,
            "mae": mae,
            "n_target_points": int(target.sum()),
        }
        logger.log_info(f"[{run_name}] ratio={r}: MSE={mse:.6f}  MAE={mae:.6f}")

        plot_path = os.path.join(save_dir, f"impute_{run_name}_{tag}.png")
        _plot_imputation(
            samples_orig, reals_orig, masks_bool,
            save_path=plot_path,
            title=f"{run_name}  missing={int(round(r * 100))}%",
            n_samples=n_plot_samples,
            n_features=n_plot_features,
        )
        logger.log_info(f"[{run_name}] ratio={r} -> {plot_path}")

    metrics_path = os.path.join(save_dir, f"metrics_{run_name}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_all, f, indent=2)
    logger.log_info(f"[{run_name}] metrics -> {metrics_path}")


def _plot_imputation(samples, reals, masks, *, save_path, title,
                     n_samples=5, n_features=4):
    """Tutorial_1.ipynb / plot.py style — per panel: red x at observed
    positions (mask=1, truth value), blue o at target positions (mask=0,
    truth value), green line for the full Diffusion-TS imputation."""
    seq_length = reals.shape[1]
    n_samples = min(n_samples, reals.shape[0])
    n_features = min(n_features, reals.shape[2])

    fig, axes = plt.subplots(
        n_samples, n_features,
        figsize=(4 * n_features, 2.0 * n_samples),
        squeeze=False,
    )
    t = np.arange(seq_length)

    for s in range(n_samples):
        for f in range(n_features):
            ax = axes[s, f]
            obs_t = t[masks[s, :, f]]
            tgt_t = t[~masks[s, :, f]]
            ax.plot(obs_t, reals[s, obs_t, f], "rx", label="Observed")
            ax.plot(tgt_t, reals[s, tgt_t, f], "bo", mfc="none", label="Target")
            ax.plot(t, samples[s, :, f], "g-", label="Imputation")
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
    impute_one(
        cfg="etth",
        gpu=0,
        seed=SEED_DEFAULT,
        milestone=MILESTONE_DEFAULT,
        ratios=list(MISSING_RATIOS_DEFAULT),
        seq_length=SEQ_LENGTH_DEFAULT,
    )
