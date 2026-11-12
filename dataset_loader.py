"""Patient-level precomputed WSI/MRI features with shape and split checks."""
import os
import warnings
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


def load_split(split_dir, fold):
    path = os.path.join(split_dir, f"splits_{fold}_bool.csv")
    source = f"split file for fold {fold}"
    try:
        df = pd.read_csv(path, dtype={"case_id": str})
    except (OSError, ValueError, pd.errors.ParserError):
        raise ValueError(f"Unable to read {source}") from None
    required = {"case_id", "label", "train", "val", "test"}
    if not required.issubset(df.columns):
        raise ValueError(f"{source}: missing columns {required - set(df.columns)}")
    if df.case_id.isna().any() or df.case_id.duplicated().any():
        raise ValueError(f"{source}: missing or duplicate patient IDs")
    if not df.label.isin([0, 1]).all():
        raise ValueError(f"{source}: labels must be 0 or 1")
    for column in ("train", "val", "test"):
        values = df[column].astype(str).str.strip().str.lower()
        if not values.isin(["true", "false", "1", "0"]).all():
            raise ValueError(f"{source}: invalid {column} flags")
        df[column] = values.isin(["true", "1"])
    train_overlap = df["train"] & (df["val"] | df["test"])
    if train_overlap.any():
        raise ValueError(f"{source}: training patients overlap with validation/test")
    return df


def feature_matrix(array, name):
    tensor = torch.from_numpy(np.asarray(array)).float()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or tensor.size(0) == 0 or not torch.isfinite(tensor).all():
        raise ValueError(f"{name}: expected finite, nonempty [N,D] features")
    return tensor


class HyperSpaceUnifiedDataset(Dataset):
    def __init__(self, df, wsi_feat_dir, mri_feat_dir, task="hyper", allow_empty=False):
        self.task, self.wsi_dir, self.mri_dir = task, wsi_feat_dir, mri_feat_dir
        self.has_wsi = task != "mri"
        self.has_mri = task != "wsi"
        valid = []
        for _, row in df.iterrows():
            cid = str(row.case_id)
            ok = ((not self.has_wsi or os.path.isfile(os.path.join(wsi_feat_dir, f"{cid}_1.h5"))) and
                  (not self.has_mri or os.path.isfile(os.path.join(mri_feat_dir, f"{cid}_1.h5"))))
            valid.append(ok)
        self.df = df.loc[np.asarray(valid, dtype=bool)].reset_index(drop=True)
        if len(self.df) != len(df):
            warnings.warn(f"{task}: dropped {len(df)-len(self.df)} patients with missing features")
        if not allow_empty and self.df.empty:
            raise ValueError("Empty training or validation dataset")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cid = str(row.case_id)
        wsi = mri = None
        if self.has_wsi:
            path = os.path.join(self.wsi_dir, f"{cid}_1.h5")
            try:
                feature_file = h5py.File(path, "r")
            except OSError:
                raise OSError("Unable to read WSI feature file") from None
            with feature_file as f:
                if "patches" in f and "regions" in f:
                    patches = feature_matrix(f["patches"][:], "WSI feature file")
                    regions = feature_matrix(f["regions"][:], "WSI feature file")
                    slide = feature_matrix(f["slide"][:], "WSI feature file")
                    p2r = torch.from_numpy(f["patch_to_region_map"][:]).long().reshape(-1)
                else:
                    key = next((k for k in ("features", "feature") if k in f), None)
                    if key is None:
                        raise KeyError("WSI feature file is missing features")
                    patches = feature_matrix(f[key][:], "WSI feature file")
                    if "coords" in f:
                        coords = torch.from_numpy(f["coords"][:]).float()
                        if coords.shape != (len(patches), 2) or not torch.isfinite(coords).all():
                            raise ValueError("WSI coordinates must be finite [N,2]")
                        lo = coords.min(0).values
                        span = (coords.max(0).values - lo).clamp_min(1.0)
                        grid = ((coords - lo) / span * 4).floor().long().clamp(0, 3)
                        raw_index = grid[:, 1] * 4 + grid[:, 0]
                        # Up to 16 occupied cells; do not invent empty tissue.
                        _, p2r = torch.unique(raw_index, sorted=True, return_inverse=True)
                    else:
                        p2r = torch.zeros(len(patches), dtype=torch.long)
                    regions = torch.stack([patches[p2r == r].mean(0) for r in range(int(p2r.max()) + 1)])
                    slide = regions.mean(0, keepdim=True)
            if len(slide) != 1 or patches.size(1) != regions.size(1) or slide.size(1) != patches.size(1):
                raise ValueError("WSI feature file has inconsistent dimensions")
            if p2r.numel() != len(patches) or (p2r < 0).any() or (p2r >= len(regions)).any():
                raise ValueError("WSI feature file has an invalid patch-to-region mapping")
            wsi = dict(patches=patches, regions=regions, slide=slide, p2r=p2r)
        if self.has_mri:
            path = os.path.join(self.mri_dir, f"{cid}_1.h5")
            try:
                feature_file = h5py.File(path, "r")
            except OSError:
                raise OSError("Unable to read MRI feature file") from None
            with feature_file as f:
                if "roi_feat" not in f or "fs_feat" not in f:
                    raise KeyError("MRI feature file requires distinct roi_feat and fs_feat keys")
                roi = feature_matrix(f["roi_feat"][:], "MRI feature file")
                fs = feature_matrix(f["fs_feat"][:], "MRI feature file")
            if len(roi) != 1 or len(fs) != 1 or roi.shape != fs.shape:
                raise ValueError("MRI features must each be [1,D]")
            mri = dict(roi_feat=roi, fs_feat=fs)
        return wsi, mri, torch.tensor(int(row.label), dtype=torch.long)


def unified_collate(batch):
    if len(batch) != 1:
        raise ValueError("Variable-size patient bags require batch_size=1")
    wsi, mri, label = batch[0]
    wc = {k: v.unsqueeze(0) for k, v in wsi.items()} if wsi is not None else None
    mc = {k: v.unsqueeze(0) for k, v in mri.items()} if mri is not None else None
    return wc, mc, label.unsqueeze(0)


def get_dataloader(fold, split_dir, wsi_dir, mri_dir, num_workers=4, task="hyper"):
    df = load_split(split_dir, fold)
    loaders = []
    for split in ("train", "val", "test"):
        dataset = HyperSpaceUnifiedDataset(
            df[df[split]], wsi_dir, mri_dir, task, allow_empty=(split == "test"))
        loaders.append(DataLoader(dataset, batch_size=1, shuffle=(split == "train"),
                                  num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                                  collate_fn=unified_collate))
    return tuple(loaders)
