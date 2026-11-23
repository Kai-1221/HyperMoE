from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import Config

SplitName = Literal["train", "val", "test", "holdout"]
MAX_WARNINGS = 5


def _clean_id(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: empty CSV")
        rows = []
        for raw in reader:
            row = {k: v for k, v in raw.items() if k is not None}
            row["case_id"] = _clean_id(row["case_id"])
            row["label"] = int(float(row["label"]))
            row["wsi_id"] = _clean_id(row.get("wsi_id") or row["case_id"])
            row["mri_id"] = _clean_id(row.get("mri_id") or row["case_id"])
            rows.append(row)
    return rows


def load_split_rows(cfg: Config, fold: int, split: SplitName) -> list[dict]:
    if split == "holdout":
        path = cfg.split_dir / "holdout.csv"
        rows = read_rows(path)
    else:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"unknown split: {split}")
        path = cfg.split_dir / f"splits_{fold}_bool.csv"
        rows = [r for r in read_rows(path) if _as_bool(r.get(split, False))]
    if not rows:
        raise ValueError(f"{split} fold {fold}: no rows in {path}")
    return rows


def build_grid_regions(
    coords: np.ndarray,
    features: np.ndarray,
    grid: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if features.shape[0] == 0:
        return (
            np.zeros((0, features.shape[1]), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    coords = coords.astype(np.float64)
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    ij = np.floor((coords - lo) / span * grid).astype(np.int64)
    ij = np.clip(ij, 0, grid - 1)
    cells = ij[:, 0] * grid + ij[:, 1]

    uniq, inv = np.unique(cells, return_inverse=True)
    sums = np.zeros((uniq.size, features.shape[1]), dtype=np.float64)
    np.add.at(sums, inv, features.astype(np.float64))
    counts = np.bincount(inv, minlength=uniq.size).astype(np.float64)
    regions = (sums / counts[:, None]).astype(np.float32)

    centers = np.zeros((uniq.size, 2), dtype=np.float32)
    for k, cell in enumerate(uniq):
        r, c = divmod(int(cell), grid)
        centers[k] = ((r + 0.5) / grid, (c + 0.5) / grid)
    return regions, centers, inv.astype(np.int64)


def load_wsi(
    cfg: Config, wsi_id: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = cfg.wsi_dir / f"{wsi_id}.h5"
    if not path.is_file():
        raise FileNotFoundError(f"WSI not found: {path}")
    with h5py.File(path, "r") as f:
        coords = f["coords"][:].astype(np.float32)
        features = f["features"][:].astype(np.float32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"{path}: coords must be [N,2], got {coords.shape}")
    if features.ndim != 2 or features.shape[1] != cfg.feature_dim:
        raise ValueError(
            f"{path}: features must be [N,{cfg.feature_dim}], got {features.shape}"
        )
    if features.shape[0] != coords.shape[0]:
        raise ValueError(f"{path}: coords/features length mismatch")
    if not np.isfinite(features).all():
        raise ValueError(f"{path}: features contain NaN/Inf")

    regions, centers, p2r = build_grid_regions(coords, features, cfg.wsi_grid)
    slide = features.mean(axis=0).astype(np.float32)
    return features, regions, slide, p2r


def load_mri(cfg: Config, mri_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = cfg.mri_dir / f"{mri_id}.h5"
    if not path.is_file():
        raise FileNotFoundError(f"MRI tokens not found: {path}")
    with h5py.File(path, "r") as f:
        fs = f["fs_feat"][:].astype(np.float32)
        roi = f["roi_feat"][:].astype(np.float32)
        state = f["roi_state"][:].astype(np.int64)
    if fs.shape != (4, cfg.feature_dim) or roi.shape != (4, cfg.feature_dim):
        raise ValueError(f"{path}: expected (4,{cfg.feature_dim}), got {fs.shape}/{roi.shape}")
    if state.shape != (4,):
        raise ValueError(f"{path}: roi_state must be (4,), got {state.shape}")
    if np.any(state == 2):
        raise ValueError(f"{path}: roi_state=2 is not allowed; use 0/1")
    if not np.isfinite(fs).all() or not np.isfinite(roi).all():
        raise ValueError(f"{path}: features contain NaN/Inf")
    return fs, roi, state


class PCRDataset(Dataset):

    def __init__(self, cfg: Config, fold: int, split: SplitName):
        self.cfg = cfg
        self.fold = fold
        self.split = split
        self.rows = load_split_rows(cfg, fold, split)
        self.warnings: list[str] = []

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        case_id = row["case_id"]
        try:
            patch, region, slide, p2r = load_wsi(self.cfg, row["wsi_id"])
            fs, roi, state = load_mri(self.cfg, row["mri_id"])
        except Exception as exc:
            raise RuntimeError(f"{self.split} fold {self.fold} case {case_id}: {exc}") from exc

        return {
            "case_id": case_id,
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "wsi_patch": torch.from_numpy(patch),
            "wsi_region": torch.from_numpy(region),
            "wsi_slide": torch.from_numpy(slide),
            "wsi_p2r": torch.from_numpy(p2r),
            "mri_fs": torch.from_numpy(fs),
            "mri_roi": torch.from_numpy(roi),
            "roi_state": torch.from_numpy(state),
        }


def collate_single(batch: list[dict]) -> dict:
    if len(batch) != 1:
        raise ValueError(f"batch_size must be 1, got {len(batch)}")
    return batch[0]


def make_loader(
    cfg: Config,
    fold: int,
    split: SplitName,
    shuffle: bool | None = None,
) -> DataLoader:
    if shuffle is None:
        shuffle = split == "train"
    ds = PCRDataset(cfg, fold, split)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        collate_fn=collate_single,
        pin_memory=torch.cuda.is_available(),
    )


def describe(cfg: Config, fold: int) -> None:
    print(f"fold {fold}")
    for split in ("train", "val", "test", "holdout"):
        try:
            rows = load_split_rows(cfg, fold, split)
        except Exception as exc:
            print(f"  {split:7s}: unavailable ({exc})")
            continue
        pos = sum(int(r["label"]) for r in rows)
        print(f"  {split:7s}: n={len(rows):4d} pos={pos:3d} ({pos / len(rows):.1%})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--check-one", action="store_true")
    args = ap.parse_args()

    cfg = Config()
    cfg.validate()
    describe(cfg, args.fold)
    if args.check_one:
        ds = PCRDataset(cfg, args.fold, "train")
        item = ds[0]
        for key, value in item.items():
            if torch.is_tensor(value):
                print(f"  {key}: {tuple(value.shape)} {value.dtype}")
            else:
                print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
