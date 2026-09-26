"""Qwen2-compatible causal attention with functional shared KV branches.

The learned pointer formulation is adapted from Jared Palmer's Kev PointerHead
at 3d9973b80b34d187f9fc8ce81de940b5767eb624 (Apache-2.0).
The backbone implementation is independent; native HF parity is an explicit gate.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from contextlib import nullcontext
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.attention.bias import causal_lower_right
from .encoding import Encoding
from .execution import BatchPolicy, plan_batches, EXECUTION_VERSION


@dataclass(frozen=True)
class BackboneConfig:
    vocab_size: int = 261
    hidden_size: int = 64
    intermediate_size: int = 128
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    max_position_embeddings: int = 512
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-6
    pointer_dim: int = 32
    lora_rank: int = 0
    lora_alpha: float = 16.0
    activation_checkpointing: bool = False
    source: str = "tiny-qwen2-fixture"
    revision: str = "local-fixture-v1"

    def __post_init__(self):
        if min(self.vocab_size, self.hidden_size, self.num_hidden_layers,
               self.num_attention_heads, self.num_key_value_heads, self.intermediate_size) < 1:
            raise ValueError("invalid architecture dimensions")
        if self.hidden_size % self.num_attention_heads or self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("invalid GQA head dimensions")
        if (self.hidden_size // self.num_attention_heads) % 2:
            raise ValueError("RoPE requires even head dimension")
        if self.lora_rank < 0 or self.rope_theta <= 0:
            raise ValueError("invalid architecture configuration")

    @classmethod
    def from_qwen(cls, config: dict, **overrides) -> BackboneConfig:
        if config.get("model_type") != "qwen2" or config.get("use_sliding_window", False):
            raise ValueError("only full-attention qwen2 is supported; hybrid/sliding models rejected")
        if config.get("rope_scaling") or config.get("use_mrope", False) or config.get("hidden_act") != "silu":
            raise ValueError("unsupported positional encoding or activation")
        keys = ("vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                "num_key_value_heads", "max_position_embeddings", "rope_theta", "rms_norm_eps")
        values = {key: config[key] for key in keys}
        return cls(**(values | overrides))


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps
    def forward(self, x: Tensor) -> Tensor:
        y = x.float()
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps)
        return self.weight * y.to(x.dtype)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.empty(rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = alpha / rank
    def forward(self, x: Tensor) -> Tensor:
        return self.base(x) + F.linear(F.linear(x, self.a), self.b) * self.scale


KV = tuple[Tensor, Tensor]
Prefix = tuple[KV, ...]


class Attention(nn.Module):
    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        d, self.h, self.kh = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads
        self.hd = d // self.h
        self.theta = cfg.rope_theta
        self.q_proj = nn.Linear(d, self.h * self.hd, bias=True)
        self.k_proj = nn.Linear(d, self.kh * self.hd, bias=True)
        self.v_proj = nn.Linear(d, self.kh * self.hd, bias=True)
        self.o_proj = nn.Linear(self.h * self.hd, d, bias=False)

    def rotate(self, x: Tensor, positions: Tensor) -> Tensor:
        inv = 1.0 / (self.theta ** (torch.arange(0, self.hd, 2, device=x.device, dtype=torch.float32) / self.hd))
        angles = positions.float()[:, None] * inv[None, :]
        angles = torch.cat((angles, angles), -1)[None, None]
        half = self.hd // 2
        rotated = torch.cat((-x[..., half:], x[..., :half]), -1)
        return x * angles.cos().to(x.dtype) + rotated * angles.sin().to(x.dtype)

    def forward(self, x: Tensor, offset: int, past: KV | None = None) -> tuple[Tensor, KV]:
        b, length, _ = x.shape
        q = self.q_proj(x).view(b, length, self.h, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(b, length, self.kh, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(b, length, self.kh, self.hd).transpose(1, 2)
        positions = torch.arange(offset, offset + length, device=x.device)
        q, k = self.rotate(q, positions), self.rotate(k, positions)
        if past is not None:
            # Functional concatenation: never mutate a parent prefix; autograd remains connected.
            # expand is a read-only view, not detach/copy; backward sums branch gradients
            # into the one shared prefix. cat creates request-local child storage.
            k = torch.cat((past[0].expand(b, -1, -1, -1), k), -2)
            v = torch.cat((past[1].expand(b, -1, -1, -1), v), -2)
        keys = k.repeat_interleave(self.h // self.kh, dim=1)
        values = v.repeat_interleave(self.h // self.kh, dim=1)
        # A long CUDA request must not silently fall back to a dense quadratic math kernel.
        # Unsupported fused-kernel/hardware combinations fail explicitly and remain a deployment gate.
        from torch.nn.attention import SDPBackend, sdpa_kernel
        guard = sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]) if q.is_cuda and k.shape[-2] >= 8192 else nullcontext()
        with guard:
            if past is None:
                h = F.scaled_dot_product_attention(q, keys, values, is_causal=True, dropout_p=0.0)
            else:
                # Suffix queries are aligned at the RIGHT, not SDPA's default upper-left triangle.
                h = F.scaled_dot_product_attention(q, keys, values,
                        attn_mask=causal_lower_right(length, k.shape[-2]), dropout_p=0.0)
        return self.o_proj(h.transpose(1, 2).reshape(b, length, -1)), (k, v)


class MLP(nn.Module):
    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        d, width = cfg.hidden_size, cfg.intermediate_size
        self.gate_proj = nn.Linear(d, width, bias=False)
        self.up_proj = nn.Linear(d, width, bias=False)
        self.down_proj = nn.Linear(width, d, bias=False)
    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)
    def forward(self, x: Tensor, offset: int, past: KV | None = None) -> tuple[Tensor, Tensor, Tensor]:
        h, (k, v) = self.self_attn(self.input_layernorm(x), offset, past)
        x = x + h
        return x + self.mlp(self.post_attention_layernorm(x)), k, v


class Backbone(nn.Module):
    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(Layer(cfg) for _ in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, ids: tuple[int, ...] | Tensor, past: Prefix | None = None,
                *, return_cache: bool = True) -> tuple[Tensor, Prefix]:
        """Tuples retain the v1 [L,D] interface; tensor rows return [B,L,D].

        Rows MUST be right-padded. Real queries cannot attend to future padding
        under causal attention. Padding queries have valid keys (including self),
        so no all-masked rows arise, and their outputs are never read by the head.
        Every row has the same complete prefix and RoPE origin; no left padding.
        """
        batched = isinstance(ids, Tensor)
        device = self.embed_tokens.weight.device
        if batched:
            if ids.ndim != 2 or ids.dtype != torch.long or ids.device != device:
                raise ValueError("backbone expects device-local int64 [batch, length] tokens")
            tokens = ids
        else:
            tokens = torch.tensor(ids, device=device, dtype=torch.long)[None]
        if tokens.ndim != 2 or min(tokens.shape) < 1:
            raise ValueError("nonempty backbone batch required")
        if past is not None:
            if len(past) != len(self.layers):
                raise ValueError("prefix layer count mismatch")
            expected = (1, self.cfg.num_key_value_heads, past[0][0].shape[-2],
                        self.cfg.hidden_size // self.cfg.num_attention_heads)
            for kv in past:
                if len(kv) != 2 or any(tuple(t.shape) != expected or t.device != device for t in kv):
                    raise ValueError("prefix must be one consistent request-local KV row per layer")
        offset = 0 if past is None else past[0][0].shape[-2]
        if offset + tokens.shape[1] > self.cfg.max_position_embeddings:
            raise ValueError("backbone sequence limit exceeded")
        x = self.embed_tokens(tokens)
        cache = []
        for i, layer in enumerate(self.layers):
            parent = None if past is None else past[i]
            if self.cfg.activation_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint
                x, k, v = checkpoint(layer, x, offset, parent, use_reentrant=False)
            else:
                x, k, v = layer(x, offset, parent)
            if return_cache:
                cache.append((k, v))
        hidden = self.norm(x)
        return (hidden if batched else hidden[0]), tuple(cache)


class PointerHead(nn.Module):
    """Kev-style learned dot-product option readout; calibration is outside training."""
    def __init__(self, hidden: int, pointer: int):
        super().__init__()
        self.q = nn.Linear(hidden, pointer)
        self.k = nn.Linear(hidden, pointer)
        self.scale = 1.0 / math.sqrt(pointer)
    def forward(self, decide: Tensor, options: Tensor) -> Tensor:
        return (self.k(options) @ self.q(decide)).float() * self.scale


class DecisionEngine(nn.Module):
    def __init__(self, cfg: BackboneConfig, batch_policy: BatchPolicy | None = None):
        super().__init__()
        self.cfg = cfg
        self.batch_policy = batch_policy or BatchPolicy()
        self.backbone = Backbone(cfg)
        self.head = PointerHead(cfg.hidden_size, cfg.pointer_dim)
        self.temperatures = {"choice": 1.0, "score": 1.0, "noul": 1.0}
        self.calibration_provenance: dict = {"status": "unfitted"}
        self.native_weights_loaded = False
        self.training_steps = 0
        if cfg.lora_rank:
            self.enable_lora()

    def enable_lora(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        for layer in self.backbone.layers:
            for name in ("q_proj", "v_proj"):
                base = getattr(layer.self_attn, name)
                setattr(layer.self_attn, name, LoRALinear(base, self.cfg.lora_rank, self.cfg.lora_alpha))

    def load_qwen_weights(self, path: str | Path):
        from safetensors.torch import load_file
        tensors = load_file(str(path), device="cpu")
        tensors = {k.removeprefix("model."): v for k, v in tensors.items() if k.startswith("model.")}
        if not tensors:
            raise ValueError("checkpoint contains no model backbone weights")
        if self.cfg.lora_rank:
            rewritten = {}
            for k, v in tensors.items():
                if ".self_attn.q_proj." in k or ".self_attn.v_proj." in k:
                    prefix, leaf = k.rsplit(".", 1)
                    k = prefix + ".base." + leaf
                rewritten[k] = v
            expected = {k for k in self.backbone.state_dict() if not k.endswith((".a", ".b"))}
            if set(rewritten) != expected:
                raise ValueError("native weight keys do not match architecture")
            self.backbone.load_state_dict(rewritten, strict=False)
        else:
            self.backbone.load_state_dict(tensors, strict=True)
        self.native_weights_loaded = True
        from .io import sha256_file
        self.loaded_backbone_sha256 = sha256_file(Path(path))

    def forward(self, encoding: Encoding, *, reference: bool = False,
                serial_reference: bool = False, policy: BatchPolicy | None = None) -> tuple[list[Tensor], dict]:
        """Parallel cached branches by default, in serving AND training.

        reference=True recomputes state+branch per row (independent correctness
        oracle). serial_reference=True retains the old serial cached path for
        benchmarking. Neither is an implicit fallback. Diagnostics are returned,
        never stored on the module, so concurrent requests share no mutable cache.
        """
        if reference and serial_reference:
            raise ValueError("choose only one reference execution path")
        selected = policy or self.batch_policy
        element_size = max(p.element_size() for p in self.parameters())
        hd = self.cfg.hidden_size // self.cfg.num_attention_heads
        kv_rate = 2 * self.cfg.num_hidden_layers * self.cfg.num_key_value_heads * hd * element_size
        gqa_rate = 2 * self.cfg.num_hidden_layers * self.cfg.hidden_size * element_size
        plan = plan_batches(encoding, selected, kv_bytes_per_token=kv_rate, gqa_bytes_per_token=gqa_rate)
        state_length = len(encoding.state)
        if any(state_length + len(b.ids) > self.cfg.max_position_embeddings for b in encoding.branches):
            raise ValueError("backbone sequence limit exceeded")
        count = len(encoding.branches)
        result: list[Tensor | None] = [None] * count
        cache_bytes = 0
        prefix = None
        if not reference:
            _, prefix = self.backbone(encoding.state)
            cache_bytes = sum(t.numel() * t.element_size() for kv in prefix for t in kv)
        sizes, lengths, indices = [], [], []
        if reference or serial_reference:
            for i, branch in enumerate(encoding.branches):
                row = encoding.state + branch.ids if reference else branch.ids
                hidden, _ = self.backbone(row, None if reference else prefix, return_cache=False)
                if reference:
                    hidden = hidden[state_length:]
                result[i] = self.head(hidden[len(branch.ids) - 1], hidden[list(branch.option_ends)])
                sizes.append(1)
                lengths.append(len(row))
                indices.append([i])
        else:
            device = self.backbone.embed_tokens.weight.device
            for batch in plan:
                # Padding token 0 is an ordinary valid vocabulary index. Its value
                # cannot reach real readouts: it occurs strictly in their future.
                rows = [list(encoding.branches[i].ids) + [0] * (batch.padded_length - length)
                        for i, length in zip(batch.indices, batch.lengths, strict=True)]
                tokens = torch.tensor(rows, dtype=torch.long, device=device)
                hidden, _ = self.backbone(tokens, prefix, return_cache=False)
                # One vectorized head readout across ragged option sets. Only the
                # final output split/order restoration is Python-side.
                owners, ends = [], []
                for row, i in enumerate(batch.indices):
                    option_ends = encoding.branches[i].option_ends
                    owners.extend([row] * len(option_ends))
                    ends.extend(option_ends)
                row_index = torch.arange(len(batch.indices), device=device)
                decide_index = torch.tensor(batch.lengths, device=device) - 1
                owner = torch.tensor(owners, device=device)
                end = torch.tensor(ends, device=device)
                queries = self.head.q(hidden[row_index, decide_index])
                keys = self.head.k(hidden[owner, end])
                scores = (keys * queries[owner]).sum(-1).float() * self.head.scale
                pieces = scores.split([len(encoding.branches[i].option_ends) for i in batch.indices])
                for i, score in zip(batch.indices, pieces, strict=True):
                    result[i] = score
                sizes.append(len(batch.indices))
                lengths.append(batch.padded_length)
                indices.append(list(batch.indices))
        useful = sum(len(b.ids) for b in encoding.branches) + state_length * (count if reference else 1)
        compute = sum(b * length for b, length in zip(sizes, lengths, strict=True)) + (0 if reference else state_length)
        diagnostics = {
            "execution_version": EXECUTION_VERSION,
            "execution_mode": "full_row_reference" if reference else "serial_cached_reference" if serial_reference else "batched",
            "batch_policy": selected.to_dict(),
            "logical_input_tokens": encoding.logical_tokens,
            # v1 forward_tokens remains unpadded model tokens; new counters name
            # the padding overhead separately instead of changing telemetry units.
            "forward_tokens": useful, "compute_tokens": compute, "padding_tokens": compute - useful,
            "prefix_cache_bytes": cache_bytes,
            "prefix_reuses": max(0, count - 1) if not reference else 0,
            "prefix_passes": 0 if reference else 1, "branch_passes": len(sizes),
            "effective_batch_sizes": sizes, "padded_branch_lengths": lengths,
            "branch_indices": indices,
            "max_padded_tokens_with_prefix": max(b * (length + (0 if reference else state_length))
                                                  for b, length in zip(sizes, lengths, strict=True)),
            "estimated_peak_cache_bytes": max(b.estimated_cache_bytes for b in plan) if not (reference or serial_reference) else
                (0 if reference else state_length * kv_rate) + max(state_length + len(b.ids) for b in encoding.branches) * (kv_rate + gqa_rate),
        }
        assert all(z is not None for z in result)
        return result, diagnostics

    def config_dict(self) -> dict:
        return asdict(self.cfg)
