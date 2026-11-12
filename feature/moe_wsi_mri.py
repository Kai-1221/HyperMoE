"""Shared node encoder, multiscale pooling, and dense Lorentz experts.

One pair of class prototypes is used after pooling, not per-level forests.
The three hyper* task names are aliases for the same multimodal architecture.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from ..utils.lorentz_utils import (
    angular_separation, exp_map, log_map, lorentz_distance, lorentz_mean,
    minkowski_dot, origin_like, project_to_lorentz,
)
from .unified_topology import build_unified_topology

NUM_LEVELS = 5
MODEL_VERSION = "hypermoe_1110_v2"


class HyperbolicExpert(nn.Module):
    def __init__(self, dim, hidden=None):
        super().__init__()
        hidden = int(hidden) if hidden else dim
        self.linear1 = nn.Linear(dim, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.proj = nn.Linear(hidden, dim) if hidden != dim else nn.Identity()
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, x_h):
        o = origin_like(x_h)
        x_t = self.proj(F.relu(self.norm(self.linear1(log_map(o, x_h)))))
        # exp_map projects to T_o H: its time component is exactly zero.
        x_h = exp_map(o, x_t)
        return exp_map(o, self.linear2(log_map(o, x_h)))


class UnifiedHierarchyMoE(nn.Module):
    def __init__(self, dim=512, num_experts=2, task="hyper", expert_hidden=None):
        super().__init__()
        if dim < 3 or num_experts < 1 or (expert_hidden is not None and expert_hidden < 1):
            raise ValueError("Invalid model dimensions")
        if task not in ("hyper", "hyper_new", "hyper_final", "wsi", "mri"):
            raise ValueError(f"Unknown task: {task}")
        self.dim, self.task = dim, task
        self.num_experts = num_experts
        # Both prediction and geometry consume this encoder. dim-1 spatial
        # outputs leave no unused learned time-coordinate output.
        self.node_proj = nn.Linear(dim, dim - 1)
        # A region containing one patch may have the same raw vector as its
        # child. Scale identity makes a strict parent/child radius learnable.
        self.level_embeddings = nn.Parameter(torch.randn(NUM_LEVELS, dim - 1) * 0.02)
        self.wsi_scale = nn.Parameter(torch.tensor(0.0))
        self.mri_scale = nn.Parameter(torch.tensor(0.0))
        self.prototypes = nn.Parameter(torch.randn(num_experts, dim - 1) * 0.1)
        self.experts = nn.ModuleList(HyperbolicExpert(dim, expert_hidden) for _ in range(num_experts))
        self.class_prototypes = nn.Parameter(torch.randn(2, dim - 1) * 0.1)
        self.cone_prototypes = nn.Parameter(torch.randn(2, dim - 1) * 0.1)
        self.cone_targets = nn.Parameter(torch.randn(2, dim - 1) * 0.1)
        self.cone_apertures = nn.Parameter(torch.ones(2) * 0.8)
        self.temperature = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))

    def _encode(self, x, scale, level):
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2 or x.size(-1) != self.dim or not len(x):
            raise ValueError(f"Expected nonempty [N,{self.dim}] features, got {tuple(x.shape)}")
        # Normalize inputs, not the output radius: fixed output norms would
        # make parent/child radial ordering impossible to learn.
        spatial = (torch.tanh(self.node_proj(F.layer_norm(x, (self.dim,))) + self.level_embeddings[level]) *
                   (2.0 * F.softplus(scale) / math.sqrt(self.dim - 1)))
        return torch.cat((torch.zeros_like(spatial[:, :1]), spatial), dim=-1)

    @staticmethod
    def _lift(spatial):
        return project_to_lorentz(torch.cat((torch.zeros_like(spatial[..., :1]), spatial), dim=-1))

    def forward(self, wsi_patches=None, wsi_regions=None, wsi_slide=None,
                mri_roi=None, mri_fs=None, p2r=None):
        has_wsi = wsi_patches is not None
        has_mri = mri_roi is not None
        if has_wsi != (wsi_regions is not None) or has_wsi != (wsi_slide is not None):
            raise ValueError("WSI requires patches, regions and slide together")
        if has_mri != (mri_fs is not None):
            raise ValueError("MRI requires ROI and full-sequence together")
        if self.task == "wsi" and (not has_wsi or has_mri):
            raise ValueError("task=wsi requires WSI only")
        if self.task == "mri" and (has_wsi or not has_mri):
            raise ValueError("task=mri requires MRI only")
        if self.task.startswith("hyper") and not (has_wsi and has_mri):
            raise ValueError("Multimodal tasks require both WSI and MRI")
        if has_wsi:
            wsi_patches = self._encode(wsi_patches, self.wsi_scale, 0)
            wsi_regions = self._encode(wsi_regions, self.wsi_scale, 1)
            wsi_slide = self._encode(wsi_slide, self.wsi_scale, 2)
            if len(wsi_slide) != 1:
                raise ValueError("Each patient must have one slide-level vector")
            n, m = len(wsi_patches), len(wsi_regions)
            if p2r is None or p2r.numel() != n or (p2r < 0).any() or (p2r >= m).any():
                raise ValueError("p2r must map every patch to a valid region")
            p2r = p2r.reshape(-1).long()
        else:
            n = m = 0
        if has_mri:
            mri_roi = self._encode(mri_roi, self.mri_scale, 3)
            mri_fs = self._encode(mri_fs, self.mri_scale, 4)
            if len(mri_roi) != 1 or len(mri_fs) != 1:
                raise ValueError("Each patient must have one ROI and one full-sequence vector")
        topo, _, raw = build_unified_topology(
            wsi_patches, wsi_regions, wsi_slide, mri_roi, mri_fs, p2r)
        self.latest_raw_h = project_to_lorentz(raw)
        levels = []
        edges = []
        if has_wsi:
            levels.extend((topo[:n].mean(0, keepdim=True), topo[n:n+m].mean(0, keepdim=True),
                           topo[n+m:n+m+1]))
            patch_index = torch.arange(n, device=raw.device)
            region_index = torch.arange(n, n + m, device=raw.device)
            edges.extend(((n + p2r, patch_index),
                          (torch.full((m,), n + m, device=raw.device, dtype=torch.long), region_index)))
        if has_mri:
            levels.extend((topo[-2:-1], topo[-1:]))
            edges.append((torch.tensor([len(raw)-1], device=raw.device),
                          torch.tensor([len(raw)-2], device=raw.device)))
        self.hierarchy_edges = edges
        level_samples = []
        if has_wsi:
            level_samples.extend((lorentz_mean(self.latest_raw_h[:n].unsqueeze(0)),
                                  lorentz_mean(self.latest_raw_h[n:n + m].unsqueeze(0)),
                                  self.latest_raw_h[n + m:n + m + 1]))
        if has_mri:
            level_samples.extend((self.latest_raw_h[-2:-1], self.latest_raw_h[-1:]))
        self.latest_level_h = torch.cat(level_samples, dim=0)
        combined_h = project_to_lorentz(torch.cat(levels).mean(0, keepdim=True))
        self.latest_combined_h = combined_h
        router_h = self._lift(self.prototypes)
        weights = torch.softmax(
            -lorentz_distance(combined_h.unsqueeze(1), router_h.unsqueeze(0)).squeeze(-1), -1)
        outputs = torch.stack([expert(combined_h) for expert in self.experts], dim=1)
        tangent_outputs = log_map(combined_h.unsqueeze(1), outputs)
        fused_tangent = (weights.unsqueeze(-1) * tangent_outputs).sum(dim=1)
        fh = exp_map(combined_h, fused_tangent)
        self.latest_expert_h = outputs
        self.latest_fh = fh
        self.latest_h = fh.detach()
        self.latest_class_prototypes = self._lift(self.class_prototypes)
        self.latest_cone_apex = self._lift(self.cone_prototypes)
        self.latest_cone_axis = self._lift(self.cone_targets)
        return self.forward_head(fh), weights

    def forward_head(self, fh):
        cph = self._lift(self.class_prototypes)
        temp = F.softplus(self.temperature) + 1e-4
        return minkowski_dot(fh.unsqueeze(1), cph.unsqueeze(0)).squeeze(-1) / temp
