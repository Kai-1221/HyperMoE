from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from config import Config
from model.lorentz import minkowski_dot


def classification_loss(
    logits: torch.Tensor,
    label: torch.Tensor,
    class_weights: torch.Tensor | None,
) -> torch.Tensor:
    return nn.CrossEntropyLoss(weight=class_weights, reduction="mean")(logits, label)


def edge_cone_penalty(geo, parent: torch.Tensor, child: torch.Tensor, cfg: Config) -> torch.Tensor:
    eps = 1e-8
    origin = geo.origin(parent)
    child_v = geo.log(parent, child)
    outward = -geo.log(parent, origin)
    denom = (
        minkowski_dot(child_v, child_v).clamp_min(eps)
        * minkowski_dot(outward, outward).clamp_min(eps)
    ).sqrt()
    cosine = minkowski_dot(child_v, outward) / denom
    angle = torch.acos(cosine.clamp(-1.0 + eps, 1.0 - eps))

    if cfg.cone_aperture_deg > 0:
        aperture = angle.new_full((), math.radians(cfg.cone_aperture_deg))
    else:
        aperture = geo.aperture(geo.radius(parent), cfg.class_cone_k)

    hinge = (angle - aperture).relu()
    if cfg.cone_normalize:
        hinge = hinge / aperture.clamp_min(1e-3)
    return hinge.mean()


def geometry_losses(model, label: torch.Tensor, cfg: Config) -> dict[str, torch.Tensor]:
    geo = model.geo
    fh = model.latest_fh
    cph = geo.lift(model.class_prototypes)
    target = label.reshape(-1)

    pos = geo.dist(fh, cph[target]).mean()
    neg = geo.dist(fh, cph[1 - target]).mean()
    hyp = (pos + F.relu(pos - neg + cfg.hyp_margin)).clamp(max=cfg.hyp_cap)

    raw = model.latest_raw_h
    if getattr(cfg, "cone_level_edges", False) and getattr(model, "level_edges", None):
        points = model.latest_level_h
        edge_index = model.level_edges
    else:
        points = raw
        edge_index = model.hierarchy_edges
    ent = raw.new_zeros(())
    cone = raw.new_zeros(())
    for parent_index, child_index in edge_index:
        parent, child = points[parent_index], points[child_index]
        parent_r = geo.radius(parent)
        child_r = geo.radius(child)
        ent = ent + F.relu(parent_r - child_r + cfg.ent_margin).mean()
        cone = cone + edge_cone_penalty(geo, parent, child, cfg)

    families = max(1, len(edge_index))
    ent = ent / families
    cone = cone / families

    class_index = int(target[0].item())
    apex = model.latest_cone_apex[class_index:class_index + 1]
    axis = model.latest_cone_axis[class_index:class_index + 1]
    samples = model.latest_level_h
    angle = geo.angle(apex.expand_as(samples), samples, axis.expand_as(samples))
    if cfg.cone_aperture_deg > 0:
        aperture = angle.new_full((), math.radians(cfg.cone_aperture_deg))
    else:
        aperture = torch.sigmoid(model.cone_apertures[class_index]) * (math.pi / 2.0)
    class_hinge = (angle - aperture).relu()
    if cfg.cone_normalize:
        class_hinge = class_hinge / aperture.clamp_min(1e-3)
    cone = cone + class_hinge.mean()

    class_distance = geo.dist(cph[0:1], cph[1:2])
    proto_sep = F.relu(cfg.proto_sep_margin - class_distance).mean()
    if not cfg.cone_tie_class:
        cone_distance = geo.dist(
            model.latest_cone_apex[0:1], model.latest_cone_apex[1:2]
        )
        proto_sep = proto_sep + F.relu(cfg.proto_sep_margin - cone_distance).mean()
    return {
        "hyp": hyp,
        "ent": ent,
        "cone": cone,
        "proto_sep": proto_sep,
        "pos": pos,
        "neg": neg,
    }


def router_balance(model, detached_features: torch.Tensor) -> torch.Tensor:
    geo = model.geo
    if str(getattr(model, "expert_split", "mixed")) == "modality":
        return detached_features.new_zeros(())
    if str(getattr(model, "router_mode", "proto")) == "tangent":
        origin = geo.origin(detached_features)
        tangent = geo.log(origin, detached_features)[..., 1:]
        tau = torch.nn.functional.softplus(model.router_log_tau) + 1e-3
        usage = torch.softmax(model.router_gate(tangent) / tau, dim=-1).mean(0)
    else:
        ph = geo.lift(model.prototypes)
        distances = geo.dist(
            detached_features.unsqueeze(1), ph.unsqueeze(0)
        ).squeeze(-1)
        usage = torch.softmax(-distances, dim=-1).mean(0)
    return (usage - 1.0 / model.num_experts).square().sum()


def compute_losses(
    model,
    logits: torch.Tensor,
    label: torch.Tensor,
    cfg: Config,
    class_weights: torch.Tensor | None = None,
    lambda_scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    cls = classification_loss(logits, label, class_weights)
    geo = geometry_losses(model, label, cfg)
    total = cls + lambda_scale * (
        cfg.lambda_hyp * geo["hyp"]
        + cfg.lambda_ent * geo["ent"]
        + cfg.lambda_cone * geo["cone"]
        + cfg.lambda_proto_sep * geo["proto_sep"]
    )

    aux = None
    aux_weight = float(getattr(cfg, "expert_aux", 0.0) or 0.0)
    expert_logits = getattr(model, "latest_expert_logits", None)
    if aux_weight > 0.0 and expert_logits is not None:
        valid = getattr(model, "latest_expert_valid", None)
        per_expert = []
        for index in range(expert_logits.shape[0]):
            if valid is not None and not bool(valid[index].item()):
                continue
            per_expert.append(
                classification_loss(expert_logits[index], label, class_weights)
            )
        if per_expert:
            aux = torch.stack(per_expert).mean()
            total = total + aux_weight * aux

    patch_weight = float(getattr(cfg, "patch_aux", 0.0) or 0.0)
    patch_logits = getattr(model, "latest_patch_logits", None)
    patch = None
    if patch_weight > 0.0 and patch_logits is not None and patch_logits.numel() > 0:
        target = label.view(1).to(patch_logits.dtype).expand_as(patch_logits)
        patch = F.binary_cross_entropy_with_logits(patch_logits, target)
        total = total + patch_weight * patch

    return {
        "total": total,
        "cls": cls,
        "hyp": geo["hyp"],
        "ent": geo["ent"],
        "cone": geo["cone"],
        "proto_sep": geo["proto_sep"],
        "pos": geo["pos"],
        "neg": geo["neg"],
        "aux": aux if aux is not None else cls.new_zeros(()),
        "patch": patch if patch is not None else cls.new_zeros(()),
    }
