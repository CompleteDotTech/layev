"""Projected Gaussian Proper-Score Policy Gradient (PGPS-v1), not proprietary RLCD.

Adapted conceptually from Laya's proper_reward and noisy-logit training notebook
at 4066d5d5fbf08b66c6757ddeedbd797bd7655bc0 (Apache-2.0).
Differences: no clipped log score; leave-one-out baseline; no advantage rescaling;
no auxiliary action head. See docs/LEARNING.md for the complete estimator.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import Tensor


@dataclass(frozen=True)
class ObjectiveConfig:
    ce: float = 1.0
    reinforce: float = 0.0
    ordinal: float = 0.0
    spherical_reward: float = 0.75
    ordinal_reward: float = 1.0
    sigma: float = 0.35
    samples: int = 4
    def __post_init__(self):
        values = (self.ce, self.reinforce, self.ordinal, self.spherical_reward, self.ordinal_reward, self.sigma)
        if not all(math.isfinite(v) and v >= 0 for v in values) or self.sigma == 0 or self.samples < 2:
            raise ValueError("invalid objective coefficients, noise, or sample count")
        if self.ce + self.reinforce + self.ordinal == 0:
            raise ValueError("objective must have an active term")


def padded(logits: list[Tensor], targets: list[list[float]]) -> tuple[Tensor, Tensor, Tensor]:
    if not logits or len(logits) != len(targets):
        raise ValueError("logits and targets must have the same nonzero length")
    size = max(z.numel() for z in logits)
    zs, ys, masks = [], [], []
    for z, target in zip(logits, targets, strict=True):
        if z.ndim != 1 or len(target) != z.numel() or not torch.isfinite(z).all():
            raise ValueError("invalid logit or target dimensions")
        t = torch.tensor(target, dtype=torch.float32, device=z.device)
        if not torch.isfinite(t).all() or (t < 0).any() or not torch.isclose(t.sum(), t.new_tensor(1.), atol=1e-6):
            raise ValueError("target must be a finite probability distribution")
        zs.append(torch.nn.functional.pad(z.float(), (0, size - len(z))))
        ys.append(torch.nn.functional.pad(t, (0, size - len(t))))
        masks.append(torch.arange(size, device=z.device) < len(z))
    return torch.stack(zs), torch.stack(ys), torch.stack(masks)


def proper_reward(logq: Tensor, target: Tensor, mask: Tensor, ordinal: Tensor,
                  spherical: float, rps_weight: float) -> Tensor:
    q = logq.exp() * mask
    log_score = (target * logq.masked_fill(~mask, 0)).sum(-1)
    sphere = (q * target).sum(-1) / q.norm(dim=-1).clamp_min(1e-12)
    cdf_error = (q.cumsum(-1) - target.cumsum(-1)).square() * mask
    rps = cdf_error.sum(-1) / (mask.sum(-1) - 1).clamp_min(1)
    return log_score + spherical * sphere - rps_weight * rps * ordinal


def objective(logits: list[Tensor], targets: list[list[float]], types: list[str],
              cfg: ObjectiveConfig) -> tuple[Tensor, dict[str, float]]:
    z, target, mask = padded(logits, targets)
    is_score = torch.tensor([t == "score" for t in types], device=z.device, dtype=torch.float32)
    logp = z.masked_fill(~mask, -1e9).log_softmax(-1)
    ce = -(target * logp).sum(-1).mean()
    p = logp.exp() * mask
    rps = (((p.cumsum(-1) - target.cumsum(-1)).square() * mask).sum(-1)
           / (mask.sum(-1) - 1).clamp_min(1) * is_score).mean()
    rl = z.sum() * 0
    reward = proper_reward(logp, target, mask, is_score, cfg.spherical_reward, cfg.ordinal_reward).mean()
    if cfg.reinforce:
        k = mask.sum(-1, keepdim=True)
        mu = (z - (z * mask).sum(-1, keepdim=True) / k) * mask
        noise = torch.randn((cfg.samples,) + tuple(z.shape), device=z.device) * cfg.sigma * mask
        noise = (noise - noise.sum(-1, keepdim=True) / k) * mask
        actions = mu.detach()[None] + noise  # detached sample; do NOT use rsample gradients
        with torch.no_grad():
            logq = actions.masked_fill(~mask, -1e9).log_softmax(-1)
            rewards = proper_reward(logq, target, mask, is_score, cfg.spherical_reward, cfg.ordinal_reward)
            baseline = (rewards.sum(0, keepdim=True) - rewards) / (cfg.samples - 1)
            advantage = rewards - baseline
        # Gaussian log density on the (K-1)-dimensional zero-sum subspace; constants omit no gradients.
        log_density = -((actions - mu[None]).square() * mask).sum(-1) / (2 * cfg.sigma ** 2)
        rl = -(advantage * log_density).mean()
        reward = rewards.mean()
    loss = cfg.ce * ce + cfg.reinforce * rl + cfg.ordinal * rps
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite objective")
    return loss, {"loss/total": float(loss.detach()), "loss/ce": float(ce.detach()),
                  "loss/policy_gradient": float(rl.detach()), "loss/ordinal": float(rps.detach()),
                  "reward/proper": float(reward.detach())}
