import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from Utils.masking_utils import noise_mask


MODALITY_DIRS = {
    "cgm": "CGM_metabonet_pp",
    "ecg": "ECG_physionet_pp",
    "eeg": "EEG_Zuco_pp",
    "emg": "EMG_emg2qwerty_pp",
}

FEATURE_SIZES = {"cgm": 6, "ecg": 12, "eeg": 128, "emg": 32}


class PhysiBaseDataset(Dataset):
    """Physi loader matching Diffusion-TS/DiMTS/CGM-GEN dataset contract.

    The post-processed files are already z-scored per recording. This loader
    leaves values in that scale and exposes identity normalize/unnormalize
    helpers so existing trainer scripts keep working.
    """

    modality = None

    def __init__(
        self,
        name,
        data_root="/mnt/nvme2/kexin/Physi_post_processed",
        window=64,
        stride=None,
        proportion=0.8,
        save2npy=False,
        neg_one_to_one=False,
        seed=123,
        period="train",
        output_dir="./OUTPUT",
        predict_length=None,
        missing_ratio=None,
        style="separate",
        distribution="geometric",
        mean_mask_length=3,
        max_files=None,
    ):
        super().__init__()
        assert period in ["train", "test"], "period must be train or test."
        if period == "train":
            assert not (predict_length is not None or missing_ratio is not None)
        if self.modality not in MODALITY_DIRS:
            raise ValueError(f"Unknown Physi modality: {self.modality}")

        self.name = name
        self.pred_len = predict_length
        self.missing_ratio = missing_ratio
        self.style = style
        self.distribution = distribution
        self.mean_mask_length = mean_mask_length
        self.window = int(window)
        self.stride = int(stride or window)
        self.period = period
        self.save2npy = save2npy
        self.auto_norm = False
        self.var_num = FEATURE_SIZES[self.modality]
        self.dir = os.path.join(output_dir, "samples")
        os.makedirs(self.dir, exist_ok=True)

        files = self._discover_files(data_root, max_files=max_files)
        self.files = self._select_split(files, proportion, seed)
        self.index = self._build_index(self.files)
        if not self.index:
            raise ValueError(
                f"No Physi windows found for {self.modality}, period={period}, "
                f"window={self.window}, stride={self.stride}"
            )
        self.sample_num = len(self.index)
        self.sample_num_total = len(self.index)
        self.len = sum(item["length"] for item in self.files)

        if period == "test":
            if missing_ratio is not None:
                self.masking = self.mask_data(seed)
            elif predict_length is not None:
                self.masking = None
            else:
                raise NotImplementedError()

        if self.save2npy:
            self._save_truth_arrays()

    def _discover_files(self, data_root, max_files=None):
        directory = Path(data_root) / MODALITY_DIRS[self.modality]
        if not directory.exists():
            raise FileNotFoundError(f"Missing Physi directory: {directory}")
        meta = np.load(directory / "_meta.npy", allow_pickle=True).item()
        if meta.get("n_features", self.var_num) != self.var_num:
            raise ValueError(
                f"Expected {self.var_num} features for {self.modality}, "
                f"but _meta.npy reports {meta.get('n_features')}"
            )
        paths = sorted(directory / fn for fn in meta["filenames"])
        if max_files is not None:
            paths = paths[:max_files]
        lengths = meta["lengths"]
        return [{"path": p, "length": int(lengths[i])} for i, p in enumerate(paths)]

    def _select_split(self, files, proportion, seed):
        if proportion >= 1.0:
            return files if self.period == "train" else []
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(files))
        train_n = int(np.ceil(len(files) * proportion))
        selected = order[:train_n] if self.period == "train" else order[train_n:]
        return [files[i] for i in sorted(selected)]

    def _build_index(self, files):
        index = []
        for item in files:
            for start in range(0, max(item["length"] - self.window + 1, 0), self.stride):
                index.append({"path": item["path"], "start": start, "length": item["length"]})
        return index

    @staticmethod
    def _load_record(path):
        record = np.load(path, allow_pickle=True).item()
        if "observed_mask" not in record:
            record["observed_mask"] = np.isfinite(record["data"])
        return record

    def _window(self, ind):
        item = self.index[ind]
        record = self._load_record(item["path"])
        start, end = item["start"], item["start"] + self.window
        data = record["data"][start:end].astype(np.float32, copy=False)
        observed = record["observed_mask"][start:end].astype(bool, copy=False)
        data = np.nan_to_num(data) * observed.astype(np.float32)
        return data, observed

    def mask_data(self, seed=2023):
        st0 = np.random.get_state()
        np.random.seed(seed)
        masks = []
        for ind in range(len(self.index)):
            data, observed = self._window(ind)
            artificial = noise_mask(
                data,
                self.missing_ratio,
                self.mean_mask_length,
                self.style,
                self.distribution,
            )
            masks.append(observed & artificial)
        masks = np.asarray(masks, dtype=bool)
        if self.save2npy:
            np.save(os.path.join(self.dir, f"{self.name}_masking_{self.window}.npy"), masks)
        np.random.set_state(st0)
        return masks

    def _save_truth_arrays(self):
        samples = np.asarray([self._window(i)[0] for i in range(len(self.index))], dtype=np.float32)
        suffix = "train" if self.period == "train" else "test"
        np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_{suffix}.npy"), samples)
        np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_{suffix}.npy"), samples)

    def normalize(self, sq):
        return sq

    def unnormalize(self, sq):
        return sq

    def __getitem__(self, ind):
        x, observed = self._window(ind)
        if self.period == "test":
            if self.missing_ratio is not None:
                mask = self.masking[ind]
            else:
                mask = observed.copy()
                mask[-self.pred_len :, :] = False
            return torch.from_numpy(x).float(), torch.from_numpy(mask)
        return torch.from_numpy(x).float()

    def __len__(self):
        return self.sample_num


class PhysiCGMDataset(PhysiBaseDataset):
    modality = "cgm"

    def __init__(self, *args, window=256, **kwargs):
        super().__init__(*args, window=window, **kwargs)


class PhysiECGDataset(PhysiBaseDataset):
    modality = "ecg"

    def __init__(self, *args, window=128, **kwargs):
        super().__init__(*args, window=window, **kwargs)


class PhysiEEGDataset(PhysiBaseDataset):
    modality = "eeg"

    def __init__(self, *args, window=256, **kwargs):
        super().__init__(*args, window=window, **kwargs)


class PhysiEMGDataset(PhysiBaseDataset):
    modality = "emg"

    def __init__(self, *args, window=256, **kwargs):
        super().__init__(*args, window=window, **kwargs)
