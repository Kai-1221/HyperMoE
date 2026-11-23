from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from config import Config

from .geometry import HyperGeo, build_geo
from .temporal_t2 import DeltaFusion, DualTemporalMean, DualTemporalT2
from .topology import build_unified_topology

NUM_LEVELS = 5
MODEL_VERSION = "1.0"


class HyperbolicExpert(nn.Module):

    def __init__(
        self,
        dim: int,
        hidden: int | None = None,
        dropout: float = 0.0,
        geo=None,
    ):
        super().__init__()
        hidden = int(hidden) if hidden else dim
        self.geo = geo if geo is not None else HyperGeo()
        self.linear1 = nn.Linear(dim, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.proj = nn.Linear(hidden, dim) if hidden != dim else nn.Identity()
        self.linear2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x_h: torch.Tensor) -> torch.Tensor:
        o = self.geo.origin(x_h)
        x_t = self.proj(F.relu(self.norm(self.linear1(self.geo.log(o, x_h)))))
        x_t = self.dropout(x_t)
        x_h = self.geo.exp(o, x_t)
        return self.geo.exp(o, self.linear2(self.geo.log(o, x_h)))


class UnifiedHyperbolicMoE(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()
        if cfg.geometry not in ("hyper", "euclidean"):
            raise NotImplementedError(f"geometry={cfg.geometry!r} is not supported")
        if cfg.temporal not in ("t2", "t0"):
            raise NotImplementedError("temporal='t1' is not supported")
        dim = cfg.feature_dim
        if dim < 3:
            raise ValueError("feature_dim must be >= 3")
        self.cfg = cfg
        self.dim = dim
        self.geo = build_geo(cfg)
        self.num_experts = cfg.num_experts

        if cfg.temporal == "t2":
            self.temporal = DualTemporalT2(
                dim=cfg.temporal_dim,
                heads=cfg.temporal_heads,
                dropout=cfg.temporal_dropout,
                share=cfg.temporal_share,
                in_dim=cfg.feature_dim,
            )
        elif cfg.temporal == "t0":
            self.temporal = DualTemporalMean(
                share=cfg.temporal_share,
                in_dim=cfg.feature_dim,
            )
        else:
            raise NotImplementedError(
                f"temporal={cfg.temporal!r} is not supported; use 't2' or 't0'"
            )

        self.delta = (
            DeltaFusion(dim, getattr(cfg, "moe_dropout", 0.0))
            if getattr(cfg, "temporal_delta", False)
            else None
        )

        self.node_proj = nn.Linear(dim, dim - 1)
        self.level_embeddings = nn.Parameter(torch.randn(NUM_LEVELS, dim - 1) * 0.02)
        self.wsi_scale = nn.Parameter(torch.tensor(0.0))
        self.mri_scale = nn.Parameter(torch.tensor(0.0))

        self.level_adaptor = bool(getattr(cfg, "level_adaptor", False))
        self.level_adaptors: nn.ModuleList | None = None
        if self.level_adaptor:
            hidden = int(getattr(cfg, "level_adaptor_hidden", 64))
            adaptors = nn.ModuleList()
            for _ in range(NUM_LEVELS):
                adaptor = nn.Sequential(
                    nn.Linear(dim - 1, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, dim - 1),
                )
                nn.init.zeros_(adaptor[-1].weight)
                nn.init.zeros_(adaptor[-1].bias)
                adaptors.append(adaptor)
            self.level_adaptors = adaptors

        self.level_radii = [float(r) for r in (getattr(cfg, "level_radii", None) or [])]
        self.level_radius_band = float(getattr(cfg, "level_radius_band", 0.05))
        self.modality_mode = str(getattr(cfg, "modality_mode", "both"))
        if self.level_radii:
            self.level_delta = nn.Parameter(torch.zeros(len(self.level_radii)))
        else:
            self.register_parameter("level_delta", None)
        self.level_edges: list[tuple[int, int]] = []
        self.head_radius = float(getattr(cfg, "head_radius", 0.0) or 0.0)

        self.prototypes = nn.Parameter(torch.randn(cfg.num_experts, dim - 1) * 0.1)
        self.experts = nn.ModuleList(
            HyperbolicExpert(
                dim,
                cfg.moe_hidden,
                getattr(cfg, "moe_dropout", 0.0),
                geo=self.geo,
            )
            for _ in range(cfg.num_experts)
        )
        self.router_mode = str(getattr(cfg, "router_mode", "proto"))
        if self.router_mode not in ("proto", "tangent"):
            raise ValueError(f"router_mode={self.router_mode!r} is not supported")
        if self.router_mode == "tangent":
            self.router_gate = nn.Linear(dim - 1, cfg.num_experts)
            tau0 = float(getattr(cfg, "router_tau_init", 1.0))
            self.router_log_tau = nn.Parameter(
                torch.tensor(math.log(math.expm1(max(tau0, 1e-3))))
            )
        if getattr(cfg, "expert_indep_init", False):
            self._independent_expert_init(int(cfg.seed))
        self.hierarchy_mode = str(getattr(cfg, "hierarchy_mode", "full"))
        if self.hierarchy_mode not in ("full", "no_topo", "no_levels"):
            raise ValueError(f"hierarchy_mode={self.hierarchy_mode!r} is not supported")
        self.expert_split = str(getattr(cfg, "expert_split", "mixed"))
        self.cone_tie_class = bool(getattr(cfg, "cone_tie_class", False))
        if self.expert_split not in ("mixed", "modality"):
            raise ValueError(f"expert_split={self.expert_split!r} is not supported")
        if self.expert_split == "modality":
            if cfg.num_experts != 2:
                raise ValueError("expert_split='modality' expects exactly two experts")
            self.fusion_weight_mode = str(getattr(cfg, "fusion_weight_mode", "gate"))
            if self.fusion_weight_mode not in ("gate", "scalar", "fixed"):
                raise ValueError(
                    f"fusion_weight_mode={self.fusion_weight_mode!r} is not supported"
                )
            if self.fusion_weight_mode in ("scalar", "fixed"):
                w0 = float(getattr(cfg, "fusion_w_init", 0.5))
                w0 = min(max(w0, 1e-3), 1.0 - 1e-3)
                if self.fusion_weight_mode == "fixed":
                    self.register_buffer("fusion_w_const", torch.tensor(w0))
                else:
                    self.fusion_logit = nn.Parameter(
                        torch.tensor(math.log(w0 / (1.0 - w0)))
                    )
            else:
                self.modality_gate = nn.Linear(2 * (dim - 1), cfg.num_experts)
                if not hasattr(self, "router_log_tau"):
                    tau0 = float(getattr(cfg, "router_tau_init", 1.0))
                    self.router_log_tau = nn.Parameter(
                        torch.tensor(math.log(math.expm1(max(tau0, 1e-3))))
                    )
        self.class_prototypes = nn.Parameter(torch.randn(2, dim - 1) * 0.1)
        self.cone_prototypes = nn.Parameter(torch.randn(2, dim - 1) * 0.1)
        self.cone_targets = nn.Parameter(torch.randn(2, dim - 1) * 0.1)
        self.cone_apertures = nn.Parameter(torch.ones(2) * 0.8)
        self.temperature = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))

        self.patch_aux = float(getattr(cfg, "patch_aux", 0.0) or 0.0)
        self.patch_topk_frac = float(getattr(cfg, "patch_topk_frac", 0.05) or 0.05)
        if self.patch_aux > 0.0:
            self.patch_head = nn.Linear(dim, 1)
            self.slide_aux = nn.Linear(3, dim)
            with torch.no_grad():
                self.slide_aux.weight.zero_()
                self.slide_aux.bias.zero_()

    def _independent_expert_init(self, seed: int) -> None:
        for index, expert in enumerate(self.experts):
            generator = torch.Generator().manual_seed(seed * 131 + index + 1)
            for module in expert.modules():
                if isinstance(module, nn.Linear):
                    scale = 1.0 / math.sqrt(module.in_features)
                    module.weight.data = (
                        torch.randn(module.weight.shape, generator=generator) * scale
                    )
                    if module.bias is not None:
                        module.bias.data.zero_()

    def _encode(self, x: torch.Tensor, scale: torch.Tensor, level: int) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2 or x.size(-1) != self.dim or not len(x):
            raise ValueError(f"Expected nonempty [N,{self.dim}] features, got {tuple(x.shape)}")
        spatial = (
            torch.tanh(
                self.node_proj(F.layer_norm(x, (self.dim,)))
                + self.level_embeddings[level]
            )
            * (2.0 * F.softplus(scale) / math.sqrt(self.dim - 1))
        )
        if self.level_adaptors is not None:
            adaptor = self.level_adaptors[min(level, len(self.level_adaptors) - 1)]
            spatial = spatial + adaptor(spatial)
        return torch.cat((torch.zeros_like(spatial[:, :1]), spatial), dim=-1)

    def _radial_level(self, vec: torch.Tensor, index: int) -> torch.Tensor:
        # keep only the direction, then rebuild the point at the prescribed radius
        slot = min(index, len(self.level_radii) - 1)
        target = self.level_radii[slot]
        radius = target + self.level_radius_band * torch.tanh(self.level_delta[slot])
        spatial = vec[..., 1:]
        unit = spatial / spatial.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        scaled = unit * torch.sinh(radius)
        return torch.cat((torch.zeros_like(scaled[..., :1]), scaled), dim=-1)

    def _lift(self, spatial: torch.Tensor) -> torch.Tensor:
        return self.geo.lift(spatial)

    def _place_at_radius(self, point: torch.Tensor, radius: float) -> torch.Tensor:
        tangent = self.geo.log(self.geo.origin(point), point)
        spatial = tangent[..., 1:]
        unit = spatial / spatial.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        scaled = unit * math.sinh(radius)
        return self.geo.reproject(torch.cat((torch.zeros_like(scaled[..., :1]), scaled), dim=-1))

    def forward(
        self,
        wsi_patch: torch.Tensor | None,
        wsi_region: torch.Tensor | None,
        wsi_slide: torch.Tensor | None,
        wsi_p2r: torch.Tensor | None,
        mri_fs: torch.Tensor | None,
        mri_roi: torch.Tensor | None,
        roi_state: torch.Tensor | None,
    ):
        has_wsi = wsi_patch is not None
        has_mri = mri_fs is not None
        if self.modality_mode == "mri":
            has_wsi = False
        elif self.modality_mode == "wsi":
            has_mri = False
        if has_wsi and (wsi_region is None or wsi_slide is None or wsi_p2r is None):
            raise ValueError("WSI requires patches, regions, slide and p2r")
        if has_mri and (mri_roi is None or roi_state is None):
            raise ValueError("MRI requires fs/roi tokens and roi_state")
        if not has_wsi and not has_mri:
            raise ValueError("At least one modality is required")

        fs_node = roi_node = None
        if has_mri:
            if mri_fs.ndim != 2 or mri_fs.shape != (4, self.dim):
                raise ValueError(f"mri_fs must be (4,{self.dim}), got {tuple(mri_fs.shape)}")
            if mri_roi.shape != mri_fs.shape:
                raise ValueError("mri_fs and mri_roi must have the same shape")
            device = mri_fs.device
            time_ids = torch.arange(4, device=device, dtype=torch.long).unsqueeze(0)
            fs_node, roi_node = self.temporal(
                mri_fs.unsqueeze(0),
                mri_roi.unsqueeze(0),
                time_ids,
                roi_state.unsqueeze(0).long(),
            )
            fs_node = fs_node.squeeze(0)
            roi_node = roi_node.squeeze(0)
            if self.delta is not None:
                d_fs, d_roi = self.delta(
                    mri_fs.unsqueeze(0), mri_roi.unsqueeze(0), roi_state.unsqueeze(0)
                )
                fs_node = fs_node + d_fs.squeeze(0)
                roi_node = roi_node + d_roi.squeeze(0)

        if has_wsi:
            if self.patch_aux > 0.0:
                patch_logits = self.patch_head(wsi_patch).squeeze(-1)
                with torch.no_grad():
                    probs = torch.sigmoid(patch_logits)
                    k = min(probs.numel(), max(1, int(round(self.patch_topk_frac * probs.numel()))))
                    stats = torch.stack(
                        [probs.mean(), probs.max(), torch.topk(probs, k).values.mean()]
                    ).reshape(1, -1)
                wsi_slide = wsi_slide + self.slide_aux(stats.reshape(-1).to(wsi_slide.dtype))
                self.latest_patch_logits = patch_logits
            patch_h = self._encode(wsi_patch, self.wsi_scale, 0)
            region_h = self._encode(wsi_region, self.wsi_scale, 1)
            slide_h = self._encode(wsi_slide, self.wsi_scale, 2)
            if len(slide_h) != 1:
                raise ValueError("Each patient must have one slide vector")
            n, m = len(patch_h), len(region_h)
            p2r = wsi_p2r.reshape(-1).long()
            if p2r.numel() != n or (p2r < 0).any() or (p2r >= m).any():
                raise ValueError("p2r must map every patch to a valid region")
        else:
            n = m = 0
            p2r = None

        if has_mri:
            roi_h = self._encode(roi_node, self.mri_scale, 3)
            fs_h = self._encode(fs_node, self.mri_scale, 4)

        topo, _, raw = build_unified_topology(
            patch_h if has_wsi else None,
            region_h if has_wsi else None,
            slide_h if has_wsi else None,
            roi_h if has_mri else None,
            fs_h if has_mri else None,
            p2r,
            mix=self.hierarchy_mode != "no_topo",
        )
        self.latest_raw_h = self.geo.reproject(raw)

        levels: list[torch.Tensor] = []
        edges: list[tuple[torch.Tensor, torch.Tensor]] = []
        if has_wsi:
            levels.extend(
                (
                    topo[:n].mean(0, keepdim=True),
                    topo[n:n + m].mean(0, keepdim=True),
                    topo[n + m:n + m + 1],
                )
            )
            patch_index = torch.arange(n, device=raw.device)
            region_index = torch.arange(n, n + m, device=raw.device)
            edges.extend(
                (
                    (n + p2r, patch_index),
                    (
                        torch.full((m,), n + m, device=raw.device, dtype=torch.long),
                        region_index,
                    ),
                )
            )
        if has_mri:
            levels.extend((topo[-2:-1], topo[-1:]))
            edges.append(
                (
                    torch.tensor([len(raw) - 1], device=raw.device),
                    torch.tensor([len(raw) - 2], device=raw.device),
                )
            )
        if self.level_radii:
            levels = [self._radial_level(vec, i) for i, vec in enumerate(levels)]
        self.hierarchy_edges = edges

        level_samples: list[torch.Tensor] = []
        if has_wsi:
            level_samples.extend(
                (
                    self.geo.mean(self.latest_raw_h[:n].unsqueeze(0)),
                    self.geo.mean(self.latest_raw_h[n:n + m].unsqueeze(0)),
                    self.latest_raw_h[n + m:n + m + 1],
                )
            )
        if has_mri:
            level_samples.extend(
                (self.latest_raw_h[-2:-1], self.latest_raw_h[-1:])
            )
        self.latest_level_h = torch.cat(level_samples, dim=0)
        if self.level_radii:
            self.latest_level_h = self.geo.reproject(torch.cat(levels, dim=0))
            self.level_edges = []
            if has_wsi:
                self.level_edges.append((2, 1))
            if has_mri:
                last = len(levels) - 1
                self.level_edges.append((last, last - 1))

        if self.expert_split == "modality":
            return self._forward_modality_split(has_wsi, has_mri, levels)

        combined_h = self.geo.reproject(torch.cat(levels).mean(0, keepdim=True))
        self.latest_combined_h = combined_h

        if self.router_mode == "tangent":
            origin = self.geo.origin(combined_h)
            tangent = self.geo.log(origin, combined_h)[..., 1:]
            tau = F.softplus(self.router_log_tau) + 1e-3
            weights = torch.softmax(self.router_gate(tangent) / tau, -1)
        else:
            router_h = self._lift(self.prototypes)
            weights = torch.softmax(
                -self.geo.dist(combined_h.unsqueeze(1), router_h.unsqueeze(0)).squeeze(-1),
                -1,
            )
        outputs = torch.stack([expert(combined_h) for expert in self.experts], dim=1)
        tangent_outputs = self.geo.log(combined_h.unsqueeze(1), outputs)
        fused_tangent = (weights.unsqueeze(-1) * tangent_outputs).sum(dim=1)
        fh = self.geo.exp(combined_h, fused_tangent)

        self.latest_expert_h = outputs
        self.latest_fh = fh
        self.latest_h = fh.detach()
        self.latest_class_prototypes = self._lift(self.class_prototypes)
        self.latest_cone_apex = self._lift(self._cone_proto())
        self.latest_cone_axis = self._lift(self.cone_targets)
        return self.forward_head(fh), weights

    def _cone_proto(self) -> torch.Tensor:
        if getattr(self, "cone_tie_class", False):
            return self.class_prototypes
        return self.cone_prototypes

    def forward_head(self, fh: torch.Tensor) -> torch.Tensor:
        cph = self._lift(self.class_prototypes)
        temp = F.softplus(self.temperature) + 1e-4
        # score = minus the geodesic distance to each class prototype, over a
        # learned temperature
        return self.geo.score(fh, cph).squeeze(-1) / temp

    def _forward_modality_split(
        self, has_wsi: bool, has_mri: bool, levels: list[torch.Tensor]
    ):
        geo = self.geo
        n_wsi_levels = 3 if has_wsi else 0
        wsi_levels = levels[:n_wsi_levels] if has_wsi else []
        mri_levels = levels[n_wsi_levels:] if has_mri else []

        if self.hierarchy_mode == "no_levels":
            wsi_vec = wsi_levels[2] if has_wsi else None
            mri_vec = mri_levels[0] if has_mri else None
        else:
            wsi_vec = torch.cat(wsi_levels).mean(0, keepdim=True) if has_wsi else None
            mri_vec = torch.cat(mri_levels).mean(0, keepdim=True) if has_mri else None
        wsi_h = geo.reproject(wsi_vec) if wsi_vec is not None else None
        mri_h = geo.reproject(mri_vec) if mri_vec is not None else None
        anchor = wsi_h if has_wsi else mri_h
        origin = geo.origin(anchor)

        slots = [wsi_h if index == 0 else mri_h for index in range(self.num_experts)]
        spatial = anchor.new_zeros(self.num_experts, self.dim - 1)
        outs: list[torch.Tensor | None] = []
        for index, point in enumerate(slots):
            if point is None:
                outs.append(None)
                continue
            out = self.experts[index](point)
            outs.append(out)
            spatial[index] = geo.log(origin, out)[0, 1:]

        available = torch.tensor(
            [point is not None for point in slots], device=anchor.device, dtype=torch.bool
        )
        mode = getattr(self, "fusion_weight_mode", "gate")
        if mode in ("scalar", "fixed"):
            if mode == "fixed":
                w = self.fusion_w_const.reshape(1).to(anchor.dtype)
            else:
                w = torch.sigmoid(self.fusion_logit).reshape(1)
            weights = torch.cat([w, 1.0 - w]).reshape(1, -1)
            weights = weights * available.reshape(1, -1).to(weights.dtype)
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
        else:
            tau = F.softplus(self.router_log_tau) + 1e-3
            gate_logits = self.modality_gate(spatial.reshape(1, -1)) / tau
            gate_logits = gate_logits.masked_fill(
                ~available.unsqueeze(0), float("-inf")
            )
            weights = torch.softmax(gate_logits, -1)
        self.latest_fusion_w = weights[0, 0].detach()

        fused = anchor.new_zeros(1, self.dim)
        for index, out in enumerate(outs):
            if out is None:
                continue
            fused = fused + weights[0, index] * geo.log(origin, out)
        fh = geo.exp(origin, fused)
        if self.head_radius > 0.0:
            fh = self._place_at_radius(fh, self.head_radius)

        self.latest_expert_h = torch.stack(
            [out if out is not None else anchor for out in outs], dim=1
        )
        num_classes = self.class_prototypes.shape[0]
        self.latest_expert_logits = torch.stack(
            [
                self.forward_head(out)
                if out is not None
                else anchor.new_zeros(1, num_classes)
                for out in outs
            ],
            dim=0,
        )
        self.latest_expert_valid = available
        self.latest_combined_h = fh
        self.latest_fh = fh
        self.latest_h = fh.detach()
        self.latest_class_prototypes = self._lift(self.class_prototypes)
        self.latest_cone_apex = self._lift(self._cone_proto())
        self.latest_cone_axis = self._lift(self.cone_targets)
        return self.forward_head(fh), weights


def build_model(cfg: Config) -> UnifiedHyperbolicMoE:
    cfg.validate()
    return UnifiedHyperbolicMoE(cfg)
