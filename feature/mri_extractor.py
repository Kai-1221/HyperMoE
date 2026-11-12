"""Extract global and ROI-pooled MRI features with a 3D ResNet."""
import os
import random

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import R3D_18_Weights, r3d_18
from tqdm import tqdm


DATA_ROOT = os.environ.get("HYPERMOE_DATA_ROOT", "./data")
WORK_DIR = os.environ.get("HYPERMOE_MRI_WORK_DIR", os.path.join(DATA_ROOT, "mri_work"))
SAVE_DIR = os.environ.get("HYPERMOE_MRI_DIR", os.path.join(DATA_ROOT, "MRI"))
MRI_SEED = int(os.environ.get("HYPERMOE_MRI_SEED", "42"))


def configure_determinism(seed=MRI_SEED):
    """Configure repeatable extraction for a fixed software and hardware environment."""
    if seed < 0:
        raise ValueError("MRI extraction seed must be nonnegative")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


class MRI_TSR_Extractor(nn.Module):
    """Encode paired timepoints as full-sequence and ROI features."""

    def __init__(self, feature_dim=512):
        super().__init__()
        base = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        # Initialize the two-channel stem from the pretrained RGB weights.
        new_weight = torch.mean(base.stem[0].weight.data, dim=1, keepdim=True).repeat(1, 2, 1, 1, 1)

        self.encoder = nn.Sequential(
            nn.Conv3d(2, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                      padding=(1, 3, 3), bias=False),
            *list(base.children())[1:-1]
        )
        self.encoder[0].weight.data = new_weight

        self.fs_fc = nn.Linear(512, feature_dim)
        self.roi_fc = nn.Linear(512, feature_dim)
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.diff_proj = nn.Linear(feature_dim * 2, feature_dim)

    def forward(self, b_t1, b_t2, b_r1, b_r2, a_t1, a_t2, a_r1, a_r2):
        """Return full-sequence and ROI features for the paired scans."""
        b_input = torch.cat([b_t1.unsqueeze(1), b_t2.unsqueeze(1)], dim=1)
        a_input = torch.cat([a_t1.unsqueeze(1), a_t2.unsqueeze(1)], dim=1)

        b_feat = self.encoder(b_input)
        a_feat = self.encoder(a_input)

        b_fs = self.fs_fc(F.adaptive_avg_pool3d(b_feat, 1).flatten(1))
        a_fs = self.fs_fc(F.adaptive_avg_pool3d(a_feat, 1).flatten(1))

        def masked_pool(feat, mask):
            mask = mask.unsqueeze(1).clamp(0, 1)
            mask = F.interpolate(mask, size=feat.shape[-3:], mode="trilinear", align_corners=False)
            mask = mask / mask.sum(dim=(-3, -2, -1), keepdim=True).clamp_min(1e-8)
            return (feat * mask).sum(dim=(-3, -2, -1))

        b_roi = self.roi_fc(masked_pool(b_feat, (b_r1 + b_r2).clamp(0, 1)))
        a_roi = self.roi_fc(masked_pool(a_feat, (a_r1 + a_r2).clamp(0, 1)))

        def fuse_timepoints(before, after):
            tokens = torch.stack((before, after), dim=1)
            query = self.query(after).unsqueeze(1)
            keys = self.key(tokens)
            values = self.value(tokens)
            attention = torch.softmax(
                query @ keys.transpose(-2, -1) * (self.fs_fc.out_features ** -0.5), dim=-1)
            temporal_context = (attention @ values).squeeze(1)
            return self.diff_proj(torch.cat((after - before, temporal_context), dim=1))

        fs_out = fuse_timepoints(b_fs, a_fs)
        roi_out = fuse_timepoints(b_roi, a_roi)
        return fs_out, roi_out


def normalize(arr):
    return (arr - np.mean(arr)) / (np.std(arr) + 1e-8)


def extract_and_save_all():
    """Extract all configured cases and write one HDF5 file per case."""
    configure_determinism()
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MRI_TSR_Extractor().to(device).eval()
    cases = sorted(d for d in os.listdir(WORK_DIR) if os.path.isdir(os.path.join(WORK_DIR, d)))
    success = 0

    for case in tqdm(cases, desc="MRI FS+ROI 提取"):
        try:
            path = os.path.join(WORK_DIR, case)

            def load(filename):
                array = np.load(os.path.join(path, filename))
                return torch.from_numpy(array).float().unsqueeze(0).to(device)

            b_t1 = load("before_T1_FS.npy")
            b_t2 = load("before_T2_FS.npy")
            b_r1 = load("before_T1_ROI.npy")
            b_r2 = load("before_T2_ROI.npy")
            a_t1 = load("after_T1_FS.npy")
            a_t2 = load("after_T2_FS.npy")
            a_r1 = load("after_T1_ROI.npy")
            a_r2 = load("after_T2_ROI.npy")

            with torch.no_grad():
                fs_feat, roi_feat = model(b_t1, b_t2, b_r1, b_r2, a_t1, a_t2, a_r1, a_r2)

            save_path = os.path.join(SAVE_DIR, f"{case}_1.h5")
            with h5py.File(save_path, "w") as feature_file:
                feature_file.create_dataset("fs_feat", data=fs_feat.cpu().numpy())
                feature_file.create_dataset("roi_feat", data=roi_feat.cpu().numpy())
                feature_file.attrs["random_seed"] = MRI_SEED
            success += 1
            print("MRI 特征提取成功")
        except Exception as exc:
            print(f"MRI 特征提取失败 ({type(exc).__name__})")

    print(f"MRI 特征提取完成，成功 {success}/{len(cases)} 例")


if __name__ == "__main__":
    extract_and_save_all()
