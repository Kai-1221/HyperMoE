"""Unit-curvature Lorentz geometry; coordinate 0 is the time coordinate."""
import torch

EPS = 1e-6


def minkowski_dot(u, v):
    return -(u[..., :1] * v[..., :1]) + (u[..., 1:] * v[..., 1:]).sum(-1, keepdim=True)


def project_to_lorentz(x, eps=EPS):
    """Lift spatial coordinates to the upper sheet, without hiding NaN/Inf."""
    spatial = x[..., 1:]
    time = torch.sqrt(1.0 + spatial.square().sum(-1, keepdim=True))
    return torch.cat((time, spatial), dim=-1)


def origin_like(x):
    o = torch.zeros_like(x)
    o[..., 0] = 1.0
    return o


def project_tangent(x, v):
    """Project v onto T_x H: <x, v>_L = 0 for <x, x>_L = -1."""
    return v + minkowski_dot(x, v) * x


def exp_map(x, v, max_norm=3.0):
    """Exponential map with tangent projection and bounded step length.

    The bound is a numerical trust region, not an unconstrained map for
    arbitrarily long tangent vectors.
    """
    v = project_tangent(x, v)
    norm = minkowski_dot(v, v).clamp_min(EPS ** 2).sqrt()
    if max_norm is not None:
        v = v * (max_norm / norm).clamp(max=1.0)
        norm = minkowski_dot(v, v).clamp_min(EPS ** 2).sqrt()
    result = torch.cosh(norm) * x + torch.sinh(norm) / norm * v
    return project_to_lorentz(result)


def log_map(x, y):
    """Logarithm with a finite coincident-point limit."""
    alpha = (-minkowski_dot(x, y)).clamp_min(1.0)
    safe_alpha = alpha.clamp_min(1.0 + EPS)
    ratio = torch.acosh(safe_alpha) / (safe_alpha.square() - 1.0).sqrt()
    ratio = torch.where(alpha > 1.0 + EPS, ratio, torch.ones_like(ratio))
    return project_tangent(x, ratio * (y - alpha * x))


def log_map0(y):
    return log_map(origin_like(y), y)[..., 1:]


def lorentz_distance(u, v):
    # On H, d(u,v) = 2 asinh(sqrt(<u-v,u-v>)/2).
    # A finite zero-distance branch avoids the sqrt/acosh gradient singularity.
    delta = u - v
    square = 0.25 * minkowski_dot(delta, delta).clamp_min(0.0)
    distance = 2.0 * torch.asinh(square.clamp_min(EPS ** 2).sqrt())
    return torch.where(square > 0, distance, torch.zeros_like(distance))


def cone_aperture(radius, K=0.5):
    return torch.asin((K / torch.sinh(radius).clamp_min(EPS)).clamp(0.0, 1.0 - EPS))


def angular_separation(parent, child, target, eps=EPS):
    child_v = log_map(parent, child)
    target_v = log_map(parent, target)
    denom = (minkowski_dot(child_v, child_v).clamp_min(eps) *
             minkowski_dot(target_v, target_v).clamp_min(eps)).sqrt()
    cosine = minkowski_dot(child_v, target_v) / denom
    return torch.acos(cosine.clamp(-1.0 + eps, 1.0 - eps))


def entail_cone_loss(parent, child, K=0.5):
    """Constrain the child to the parent's outward-facing entailment cone."""
    o = origin_like(parent)
    child_v = log_map(parent, child)
    outward = -log_map(parent, o)
    denom = (minkowski_dot(child_v, child_v).clamp_min(EPS) *
             minkowski_dot(outward, outward).clamp_min(EPS)).sqrt()
    cosine = minkowski_dot(child_v, outward) / denom
    angle = torch.acos(cosine.clamp(-1.0 + EPS, 1.0 - EPS))
    return (angle - cone_aperture(lorentz_distance(parent, o), K)).relu().mean()


def lorentz_mean(x, weights=None):
    """Normalized extrinsic barycenter, not an exact Frechet mean.

    Input shape is (..., number_of_points, dim); weights are nonnegative.
    """
    if weights is None:
        weights = torch.ones_like(x[..., 0])
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(EPS)
    average = (weights.unsqueeze(-1) * x).sum(-2)
    norm = (-minkowski_dot(average, average)).clamp_min(EPS).sqrt()
    return project_to_lorentz(average / norm)
