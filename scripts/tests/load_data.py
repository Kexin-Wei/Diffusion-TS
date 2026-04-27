"""Smoke test: build Simglucose{1,3}MDataset for train + test and check shapes/dtypes/range.

Run from the Diffusion-TS root:
  ./.venv/bin/python scripts/tests/load_data.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from Utils.Data_utils.real_datasets import Simglucose1MDataset, Simglucose3MDataset

DATA_ROOT = Path(__file__).resolve().parents[2] / "Data" / "datasets" / "simglucose"
WINDOW = 24
EXPECTED_FEATURES = 22
EXPECTED_RAW_TIMESTEPS = 30 * 1450  # patients × per-patient timesteps after flatten

CASES = [
    (Simglucose1MDataset, "simglucose1m"),
    (Simglucose3MDataset, "simglucose3m"),
]


def _check(cls, name: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        # ---- train: full data, no holdout (mirrors yaml proportion=1.0) ----
        train = cls(
            name=name,
            data_root=str(DATA_ROOT),
            window=WINDOW,
            proportion=1.0,
            save2npy=False,
            neg_one_to_one=True,
            seed=123,
            period="train",
            output_dir=tmp,
        )
        assert train.rawdata.shape == (
            EXPECTED_RAW_TIMESTEPS,
            EXPECTED_FEATURES,
        ), f"{name}: unexpected rawdata shape: {train.rawdata.shape}"
        x = train[0]
        assert isinstance(x, torch.Tensor) and x.dtype == torch.float32
        assert x.shape == (WINDOW, EXPECTED_FEATURES)
        # neg_one_to_one=True ⇒ MinMax scaled then mapped to [-1, 1]
        assert (
            -1.0 <= x.min().item() and x.max().item() <= 1.0
        ), f"{name}: range out of [-1, 1]: [{x.min().item()}, {x.max().item()}]"
        print(f"{name} train: {len(train):>5d} samples, sample {tuple(x.shape)} {x.dtype}")

        # ---- test: 10% holdout, imputation masking ----
        test = cls(
            name=name,
            data_root=str(DATA_ROOT),
            window=WINDOW,
            proportion=0.9,
            save2npy=False,
            neg_one_to_one=True,
            seed=123,
            period="test",
            missing_ratio=0.3,
            output_dir=tmp,
        )
        x_test, m_test = test[0]
        assert x_test.shape == (WINDOW, EXPECTED_FEATURES)
        assert m_test.shape == (WINDOW, EXPECTED_FEATURES)
        assert m_test.dtype == torch.bool
        # ~30% of mask should be False (masked-out positions)
        masked_frac = 1.0 - m_test.float().mean().item()
        print(
            f"{name} test:  {len(test):>5d} samples, sample {tuple(x_test.shape)} {x_test.dtype}, "
            f"masked≈{masked_frac:.0%}"
        )


def test_simglucose() -> None:
    for cls, name in CASES:
        _check(cls, name)


if __name__ == "__main__":
    test_simglucose()
