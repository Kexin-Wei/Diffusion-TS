#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from Utils.Data_utils.physi_datasets import (  # noqa: E402
    PhysiCGMDataset,
    PhysiECGDataset,
    PhysiEEGDataset,
    PhysiEMGDataset,
)


DATASETS = {
    "cgm": (PhysiCGMDataset, 6),
    "ecg": (PhysiECGDataset, 12),
    "eeg": (PhysiEEGDataset, 128),
    "emg": (PhysiEMGDataset, 32),
}


def check_modality(name, cls, feature_size, args):
    train = cls(
        name=f"physi_{name}",
        data_root=args.data_root,
        window=args.window,
        stride=args.stride,
        period="train",
        proportion=1.0,
        save2npy=False,
        max_files=args.max_files,
    )
    x = train[0]
    assert isinstance(x, torch.Tensor)
    assert tuple(x.shape) == (args.window, feature_size)
    assert train.var_num == feature_size
    assert train.auto_norm is False

    batch = next(iter(DataLoader(train, batch_size=2, shuffle=False)))
    assert tuple(batch.shape[1:]) == (args.window, feature_size)

    test = cls(
        name=f"physi_{name}",
        data_root=args.data_root,
        window=args.window,
        stride=args.stride,
        period="test",
        proportion=0.5,
        missing_ratio=args.missing_ratio,
        save2npy=False,
        max_files=max(args.max_files, 4),
    )
    x_test, mask = test[0]
    assert tuple(x_test.shape) == (args.window, feature_size)
    assert tuple(mask.shape) == (args.window, feature_size)
    assert mask.dtype == torch.bool

    print(
        f"{name}: train_windows={len(train)} test_windows={len(test)} "
        f"shape={tuple(x.shape)} observed={float(mask.float().mean()):.3f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/mnt/nvme2/kexin/Physi_post_processed")
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--stride", type=int, default=4096)
    parser.add_argument("--max_files", type=int, default=4)
    parser.add_argument("--missing_ratio", type=float, default=0.2)
    parser.add_argument("--modalities", nargs="+", default=list(DATASETS))
    args = parser.parse_args()

    for modality in args.modalities:
        cls, feature_size = DATASETS[modality]
        check_modality(modality, cls, feature_size, args)


if __name__ == "__main__":
    main()
