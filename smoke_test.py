"""Build the model, run one forward pass and report the parameter count."""
import torch

from config import Config
from model.hyperbolic import build_model

cfg = Config()
cfg.temporal = "t2"
cfg.temporal_dim = 128
cfg.temporal_delta = True
cfg.level_radii = [0.75, 0.60, 0.45, 0.30, 0.15]
cfg.level_adaptor = True
cfg.level_radius_band = 0.05
cfg.cone_level_edges = True
cfg.cone_normalize = True
cfg.cone_tie_class = True
cfg.cone_aperture_deg = 60
cfg.lambda_cone = 0.2
cfg.expert_split = "modality"
cfg.fusion_weight_mode = "fixed"
cfg.fusion_w_init = 0.1
cfg.validate()

model = build_model(cfg)
total = sum(p.numel() for p in model.parameters())
print("parameters", total)

n_patch, n_region = 300, 16
batch = dict(
    wsi_patch=torch.randn(n_patch, 512),
    wsi_region=torch.randn(n_region, 512),
    wsi_slide=torch.randn(1, 512),
    wsi_p2r=torch.randint(0, n_region, (n_patch,)),
    mri_fs=torch.randn(4, 512),
    mri_roi=torch.randn(4, 512),
    roi_state=torch.ones(4, dtype=torch.long),
)

out = model(
    batch["wsi_patch"],
    batch["wsi_region"],
    batch["wsi_slide"],
    batch["wsi_p2r"],
    batch["mri_fs"],
    batch["mri_roi"],
    batch["roi_state"],
)
logits = out[0] if isinstance(out, tuple) else out
print("logits", tuple(logits.shape), float(logits.sum()))
