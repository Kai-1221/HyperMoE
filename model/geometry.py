from __future__ import annotations

import math

import torch

from .lorentz import (
    angular_separation,
    cone_aperture,
    entail_cone_loss,
    exp_map,
    log_map,
    lorentz_distance,
    lorentz_mean,
    minkowski_dot,
    origin_like,
    project_to_lorentz,
)

EPS = 1e-6


class HyperGeo:

    name = "hyper"
    spatial_scale = 1.0

    def origin(self, x: torch.Tensor) -> torch.Tensor:
        return origin_like(x)

    def lift(self, spatial: torch.Tensor) -> torch.Tensor:
        return project_to_lorentz(
            torch.cat((torch.zeros_like(spatial[..., :1]), spatial), dim=-1)
        )

    def reproject(self, ambient: torch.Tensor) -> torch.Tensor:
        return project_to_lorentz(ambient)

    def exp(self, base: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
        return exp_map(base, tangent)

    def log(self, base: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
        return log_map(base, point)

    def dist(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return lorentz_distance(u, v)

    def mean(self, x: torch.Tensor) -> torch.Tensor:
        return lorentz_mean(x)

    def radius(self, x: torch.Tensor) -> torch.Tensor:
        return lorentz_distance(x, origin_like(x))

    def score(self, features: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        return minkowski_dot(features.unsqueeze(1), prototypes.unsqueeze(0))

    def cone_loss(self, parent: torch.Tensor, child: torch.Tensor, k: float) -> torch.Tensor:
        return entail_cone_loss(parent, child, k)

    def angle(
        self, apex: torch.Tensor, samples: torch.Tensor, axis: torch.Tensor
    ) -> torch.Tensor:
        return angular_separation(apex, samples, axis)

    def aperture(self, radius: torch.Tensor, k: float) -> torch.Tensor:
        return cone_aperture(radius, k)


class EuclidGeo(HyperGeo):

    name = "euclidean"

    def __init__(self, dim: int):
        self.dim = dim
        self.scale = math.sqrt(max(1, dim - 1))
        self.spatial_scale = self.scale

    def origin(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    def lift(self, spatial: torch.Tensor) -> torch.Tensor:
        return torch.cat((torch.zeros_like(spatial[..., :1]), spatial), dim=-1)

    def reproject(self, ambient: torch.Tensor) -> torch.Tensor:
        return ambient

    def exp(self, base: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
        return base + tangent

    def log(self, base: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
        return point - base

    def _spatial(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., 1:]

    def dist(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(
            self._spatial(u - v), dim=-1, keepdim=True
        ) / self.scale

    def mean(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=-2)

    def radius(self, x: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(self._spatial(x), dim=-1, keepdim=True) / self.scale

    def score(self, features: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        return -self.dist(features.unsqueeze(1), prototypes.unsqueeze(0))

    def _cosine_angle(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        v1 = self._spatial(v1)
        v2 = self._spatial(v2)
        denom = (
            v1.square().sum(-1, keepdim=True).clamp_min(EPS)
            * v2.square().sum(-1, keepdim=True).clamp_min(EPS)
        ).sqrt()
        cosine = (v1 * v2).sum(-1, keepdim=True) / denom
        return torch.acos(cosine.clamp(-1.0 + EPS, 1.0 - EPS))

    def cone_loss(self, parent: torch.Tensor, child: torch.Tensor, k: float) -> torch.Tensor:
        child_v = self._spatial(child - parent)
        outward = self._spatial(self.origin(parent) - parent)
        angle = self._cosine_angle(child_v, outward)
        return (angle - self.aperture(self.radius(parent), k)).relu().mean()

    def angle(
        self, apex: torch.Tensor, samples: torch.Tensor, axis: torch.Tensor
    ) -> torch.Tensor:
        return self._cosine_angle(samples - apex, axis - apex)

    def aperture(self, radius: torch.Tensor, k: float) -> torch.Tensor:
        return torch.asin((k / radius.clamp_min(EPS)).clamp(0.0, 1.0 - EPS))


def build_geo(cfg) -> HyperGeo:
    if getattr(cfg, "geometry", "hyper") == "euclidean":
        return EuclidGeo(cfg.feature_dim)
    return HyperGeo()
