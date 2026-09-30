"""Explicit experimental CUDA AdamW with persistent CPU moments.

Requires torch 2.10.0+cu128. Primitive qualification and full-weight production
qualification are separate; this backend never activates as an automatic fallback.
"""
from __future__ import annotations

import math
import copy
import torch
from torch.optim import Optimizer
from torch.optim.adamw import adamw

VERSION = "cpu-state-streaming-adamw-v1-torch2.10.0+cu128"
TORCH_VERSION = "2.10.0+cu128"


class CPUStateStreamingAdamW(Optimizer):
    def __init__(self, named_parameters, *, lr: float, weight_decay: float,
                 max_state_bytes: int, betas=(0.9, 0.999), eps=1e-8):
        if torch.__version__ != TORCH_VERSION:
            raise RuntimeError(f"{VERSION} requires torch {TORCH_VERSION}")
        pairs = [(name, p) for name, p in named_parameters if p.requires_grad]
        if (not pairs or len({id(p) for _, p in pairs}) != len(pairs)
                or len({name for name, _ in pairs}) != len(pairs)):
            raise ValueError("empty or duplicate trainable parameter list")
        if len({p.untyped_storage().data_ptr() for _, p in pairs}) != len(pairs):
            raise ValueError("shared parameter storage is unsupported")
        for name, p in pairs:
            if not isinstance(name, str) or not name or p.device.type != "cuda" or p.dtype != torch.float32:
                raise ValueError("CPU-state backend requires named CUDA FP32 parameters")
        if type(max_state_bytes) is not int or max_state_bytes < 1:
            raise ValueError("explicit positive max_state_bytes required")
        required = sum(2 * p.numel() * p.element_size() + 4 for _, p in pairs)
        if required > max_state_bytes:
            raise MemoryError(f"CPU AdamW state requires {required} bytes, budget {max_state_bytes}")
        if not (isinstance(lr, (int, float)) and math.isfinite(lr) and lr >= 0
                and isinstance(weight_decay, (int, float)) and math.isfinite(weight_decay) and weight_decay >= 0
                and isinstance(eps, (int, float)) and math.isfinite(eps) and eps > 0
                and len(betas) == 2 and all(isinstance(b, (int, float)) and 0 <= b < 1 for b in betas)):
            raise ValueError("unsupported AdamW options")
        specs = [(name, tuple(p.shape), str(p.dtype)) for name, p in pairs]
        defaults = dict(lr=float(lr), betas=tuple(float(b) for b in betas), eps=float(eps),
                        weight_decay=float(weight_decay), amsgrad=False, foreach=False,
                        fused=False, capturable=False, differentiable=False, maximize=False,
                        algorithm_version=VERSION, max_state_bytes=max_state_bytes,
                        param_specs=specs)
        super().__init__([p for _, p in pairs], defaults)
        self._fixed = copy.deepcopy({key: value for key, value in defaults.items()
                                     if key != "lr"})
        # Process-local identity. Never put addresses or tensor versions into
        # config/checkpoints: fresh-process resume allocates different objects.
        self._private_parameter_identity = tuple(
            self._live_signature(p) for p in self.param_groups[0]["params"])

    @staticmethod
    def _live_signature(p):
        return (id(p), tuple(p.shape), p.dtype, p.device,
                p.untyped_storage().data_ptr(), p.storage_offset(),
                tuple(p.stride()), p.requires_grad)

    def _check_group(self):
        if len(self.param_groups) != 1:
            raise ValueError("CPU AdamW requires one parameter group")
        group = self.param_groups[0]
        if tuple(self._live_signature(p) for p in group["params"]) != self._private_parameter_identity:
            raise ValueError("live CPU AdamW parameter identity/order/layout changed")
        if (set(group) - set(self._fixed) - {"params", "lr", "initial_lr"}
                or any(group.get(key) != value for key, value in self._fixed.items())
                or not isinstance(group.get("lr"), (float, int))
                or not math.isfinite(group["lr"]) or group["lr"] < 0):
            raise ValueError("CPU AdamW group options changed")
        if "initial_lr" in group and (not isinstance(group["initial_lr"], (float, int))
                                       or not math.isfinite(group["initial_lr"])
                                       or group["initial_lr"] < 0):
            raise ValueError("invalid initial learning rate")
        return group

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("closure is unsupported by CPU-state backend")
        group = self._check_group()
        for p in group["params"]:
            grad = p.grad
            if grad is None:
                continue  # stock AdamW semantics
            if grad.is_sparse or grad.device != p.device or grad.dtype != p.dtype:
                raise ValueError("unsupported optimizer gradient")
            state = self.state[p]
            if not state:
                state["step"] = torch.tensor(0.0, dtype=torch.float32, device="cpu")
                state["exp_avg"] = torch.zeros_like(p, device="cpu", memory_format=torch.preserve_format)
                state["exp_avg_sq"] = torch.zeros_like(p, device="cpu", memory_format=torch.preserve_format)
            self._check_state(state, p)
            exp_avg = state["exp_avg"].to(p.device)
            exp_avg_sq = state["exp_avg_sq"].to(p.device)
            try:
                adamw([p], [grad], [exp_avg], [exp_avg_sq], [], [state["step"]],
                      foreach=False, capturable=False, differentiable=False, fused=False,
                      grad_scale=None, found_inf=None, has_complex=False,
                      amsgrad=False, beta1=group["betas"][0], beta2=group["betas"][1],
                      lr=group["lr"], weight_decay=group["weight_decay"],
                      eps=group["eps"], maximize=False)
                state["exp_avg"].copy_(exp_avg, non_blocking=False)
                state["exp_avg_sq"].copy_(exp_avg_sq, non_blocking=False)
            finally:
                del exp_avg, exp_avg_sq
        return None

    @staticmethod
    def _check_state(state, p, *, deep=False):
        if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("invalid CPU AdamW state keys")
        step = state["step"]
        if (not isinstance(step, torch.Tensor) or step.device.type != "cpu"
                or step.dtype != torch.float32 or step.shape != ()
                or step.requires_grad or not math.isfinite(float(step))
                or float(step) < 0 or not float(step).is_integer()):
            raise ValueError("invalid CPU AdamW step")
        for key in ("exp_avg", "exp_avg_sq"):
            item = state[key]
            if (not isinstance(item, torch.Tensor) or item.device.type != "cpu"
                    or item.dtype != p.dtype or item.shape != p.shape or item.requires_grad):
                raise ValueError("invalid CPU AdamW moment")
            if deep and (not bool(torch.isfinite(item).all())
                         or (key == "exp_avg_sq" and bool((item < 0).any()))):
                raise ValueError("nonfinite or negative CPU AdamW moment")

    def load_state_dict(self, state_dict):
        # Optimizer.load_state_dict would silently move all moments to CUDA.
        if not isinstance(state_dict, dict) or set(state_dict) != {"state", "param_groups"}:
            raise ValueError("invalid CPU AdamW optimizer payload")
        self._check_group()
        groups = state_dict["param_groups"]
        if not isinstance(groups, list) or len(groups) != 1:
            raise ValueError("CPU AdamW requires one parameter group")
        saved, current = groups[0], self.param_groups[0]
        ids = saved.get("params")
        if (not isinstance(ids, list) or any(type(key) is not int for key in ids)
                or ids != list(range(len(current["params"])))):
            raise ValueError("CPU AdamW parameter count/order changed")
        immutable = ("betas", "eps", "weight_decay", "amsgrad", "foreach", "fused",
                     "capturable", "differentiable", "maximize", "algorithm_version",
                     "max_state_bytes", "param_specs")
        if any(saved.get(key) != self._fixed[key] for key in immutable):
            raise ValueError("CPU AdamW parameter identity/options changed")
        if (set(saved) - set(self._fixed) - {"lr", "params", "initial_lr"}
                or not isinstance(saved.get("lr"), (float, int))):
            raise ValueError("unknown CPU AdamW group option")
        if (not math.isfinite(saved["lr"]) or saved["lr"] < 0
                or ("initial_lr" in saved and
                    (not isinstance(saved["initial_lr"], (float, int))
                     or not math.isfinite(saved["initial_lr"]) or saved["initial_lr"] < 0))):
            raise ValueError("invalid CPU AdamW learning rate")
        raw = state_dict["state"]
        if (not isinstance(raw, dict) or any(type(key) is not int for key in raw)
                or set(raw) - set(ids)):
            raise ValueError("unknown CPU AdamW state parameter")
        mapped = {}
        state_storages = set()
        for key, p in zip(ids, current["params"], strict=True):
            item = raw.get(key, {})
            if not isinstance(item, dict):
                raise ValueError("invalid CPU AdamW per-parameter state")
            if item:
                self._check_state(item, p, deep=True)
                for field in ("step", "exp_avg", "exp_avg_sq"):
                    ptr = item[field].untyped_storage().data_ptr()
                    if ptr in state_storages:
                        raise ValueError("aliased CPU AdamW state storage")
                    state_storages.add(ptr)
            mapped[p] = item
        # Commit only after full validation; reuse deserialized CPU tensors so
        # resume does not duplicate the entire host optimizer state.
        self.state.clear()
        self.state.update(mapped)
        for key in ("lr", "initial_lr"):
            if key in saved:
                current[key] = saved[key]

    def state_dict(self):
        self._check_group()
        return super().state_dict()
