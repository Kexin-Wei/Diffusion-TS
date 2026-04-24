#!/usr/bin/env python3
"""Plot Diffusion-TS imputation samples vs. ground truth.

Reproduces the visualization style of Diffusion-TS/Tutorial_1.ipynb cell 7:
  red x  -> observed points (mask=1)
  blue o -> ground-truth at missing positions (mask=0)
  green  -> Diffusion-TS imputation (full sequence)

Important: real_datasets.py / mujoco_dataset.py / sine_dataset.py write a
single mask file per dataset (no missing-ratio in the filename), so a fresh
imputation run overwrites it. To plot per-ratio correctly this script must
be invoked between sweep iterations — that's what test_diffusion_ts.py does.

Run from the Diffusion-TS directory:
  python scripts/plot_diffusion_ts.py --name etth --ratio 0.7
  python scripts/plot_diffusion_ts.py --name etth --ratio 0.7 --sample-idx 3
  python scripts/plot_diffusion_ts.py --name etth --ratio 0.7 --n-features 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DTS = Path(__file__).resolve().parent.parent
OUTPUT_DIR = DTS / "OUTPUT"


def find_truth(samples_dir: Path) -> Path:
    for pattern in ("*_norm_truth_*_test.npy", "*_ground_truth_*_test.npy"):
        matches = sorted(samples_dir.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"no truth file in {samples_dir}")


def find_mask(samples_dir: Path) -> Path:
    matches = sorted(samples_dir.glob("*_masking_*.npy"))
    if not matches:
        raise FileNotFoundError(f"no mask file in {samples_dir}")
    return matches[0]


def tag(ratio: float) -> str:
    return f"m{int(round(ratio * 100))}"


def plot_imputation(name: str, ratio: float, sample_idx: int, n_features: int) -> Path:
    pct = int(round(ratio * 100))
    ds_dir = OUTPUT_DIR / name
    samples_dir = ds_dir / "samples"

    samples_path = ds_dir / f"ddpm_infill_{name}_{tag(ratio)}.npy"
    truth_path = find_truth(samples_dir)
    mask_path = find_mask(samples_dir)

    samples = np.load(samples_path)
    truth = np.load(truth_path)
    mask = np.load(mask_path).astype(bool)

    if not (samples.shape == truth.shape == mask.shape):
        raise ValueError(
            f"shape mismatch: samples={samples.shape} truth={truth.shape} mask={mask.shape}"
        )

    n_samples, seq_len, feat_dim = truth.shape
    if not 0 <= sample_idx < n_samples:
        raise IndexError(f"sample_idx={sample_idx} out of range [0,{n_samples})")
    feats = min(feat_dim, max(1, n_features))

    fig, axes = plt.subplots(nrows=feats, ncols=1, figsize=(12, 3 * feats), squeeze=False)
    t = np.arange(seq_len)
    for i in range(feats):
        ax = axes[i, 0]
        observed_t = t[mask[sample_idx, :, i]]
        target_t = t[~mask[sample_idx, :, i]]
        ax.plot(observed_t, truth[sample_idx, observed_t, i], "rx", label="observed")
        ax.plot(target_t, truth[sample_idx, target_t, i], "bo", label="target", mfc="none")
        ax.plot(t, samples[sample_idx, :, i], "g-", label="Diffusion-TS")
        ax.set_ylabel(f"feat {i}")
        if i == 0:
            ax.legend(loc="best", fontsize=9)
    axes[-1, 0].set_xlabel("time")
    fig.suptitle(f"{name}  missing={pct}%  sample={sample_idx}  ({feats}/{feat_dim} feats)")
    fig.tight_layout()

    plots_dir = ds_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    out = plots_dir / f"{name}_{tag(ratio)}_s{sample_idx}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True, help="dataset name (e.g. etth)")
    ap.add_argument("--ratio", type=float, required=True, help="missing ratio used for the run")
    ap.add_argument("--sample-idx", type=int, default=0, help="which test sample to plot (default 0)")
    ap.add_argument("--n-features", type=int, default=5, help="features to stack (default 5)")
    args = ap.parse_args()

    try:
        out = plot_imputation(args.name, args.ratio, args.sample_idx, args.n_features)
    except (FileNotFoundError, ValueError, IndexError) as e:
        print(f"plot failed for {args.name} m={args.ratio}: {e}", file=sys.stderr)
        return 1
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
