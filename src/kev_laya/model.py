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
from .native_gradients import CanonicalLoRAProjection, CanonicalLongSDPA


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


def backbone_project(module: nn.Linear | LoRALinear, x: Tensor) -> Tensor:
    """Keep native CUDA projection GEMMs independent of request row count.

    CUDA may select a different GEMM algorithm when the same token is
    projected as one cached branch, several branches, or part of a full row.
    BF16 uses 1024-row tiles; FP32 uses 64-row tiles to preserve the short
    pinned Transformers oracle. The last tile is padded before multiplication
    and trimmed afterwards. Apply the rule to LoRA base and low-rank terms.
    CPU and other precision paths retain the pinned reference arithmetic.
    """
    bf16 = (torch.is_autocast_enabled("cuda")
            and torch.get_autocast_dtype("cuda") == torch.bfloat16)
    fp32 = x.dtype == torch.float32 and not torch.is_autocast_enabled("cuda")
    if not (x.is_cuda and (bf16 or fp32)):
        return module(x)
    if bf16 and isinstance(module, LoRALinear) and torch.is_grad_enabled():
        return CanonicalLoRAProjection.apply(
            x, module.base.weight, module.base.bias, module.a, module.b,
            module.scale)
    tile_rows = 1024 if bf16 else 64
    rows = x.reshape(-1, x.shape[-1])
    parts = []
    for start in range(0, rows.shape[0], tile_rows):
        tile = rows[start:start + tile_rows]
        count = tile.shape[0]
        if count < tile_rows:
            tile = F.pad(tile, (0, 0, 0, tile_rows - count))
        if isinstance(module, LoRALinear):
            projected = module.base(tile) + F.linear(F.linear(tile, module.a), module.b) * module.scale
        else:
            projected = F.linear(tile, module.weight, module.bias)
        parts.append(projected[:count])
    return torch.cat(parts, dim=0).reshape(*x.shape[:-1], parts[0].shape[-1])


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
        q = backbone_project(self.q_proj, x).view(b, length, self.h, self.hd).transpose(1, 2)
        k = backbone_project(self.k_proj, x).view(b, length, self.kh, self.hd).transpose(1, 2)
        v = backbone_project(self.v_proj, x).view(b, length, self.kh, self.hd).transpose(1, 2)
        positions = torch.arange(offset, offset + length, device=x.device)
        q, k = self.rotate(q, positions), self.rotate(k, positions)
        if (self.training and torch.is_grad_enabled() and q.is_cuda
                and offset + length >= 8192 and q.dtype == torch.bfloat16):
            k, v = k.double(), v.double()
        if past is not None:
            # Functional concatenation: never mutate a parent prefix; autograd remains connected.
            # expand is a read-only view, not detach/copy; backward sums branch gradients
            # into the one shared prefix. cat creates request-local child storage.
            k = torch.cat((past[0].to(k.dtype).expand(b, -1, -1, -1), k), -2)
            v = torch.cat((past[1].to(v.dtype).expand(b, -1, -1, -1), v), -2)
        # During long CUDA training, keep grouped KV heads as views. Materializing
        # every repeated head retains large tensors through backward. Separate
        # fused calls per KV group still sum gradients into the shared parent.
        if self.training and torch.is_grad_enabled() and q.is_cuda and k.shape[-2] >= 8192:
            from torch.nn.attention import SDPBackend, sdpa_kernel
            canonical_bf16 = q.dtype == torch.bfloat16
            # The native forward still receives BF16. Double cache storage lets
            # separate branch VJPs sum into one parent before parameter rounding.
            if canonical_bf16:
                k, v = k.double(), v.double()
            ratio = self.h // self.kh
            parts = []
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                for head in range(self.kh):
                    query = q[:, head * ratio:(head + 1) * ratio]
                    key = k[:, head:head + 1].expand(-1, ratio, -1, -1)
                    value = v[:, head:head + 1].expand(-1, ratio, -1, -1)
                    mask = None if past is None else causal_lower_right(length, k.shape[-2])
                    if canonical_bf16:
                        part = CanonicalLongSDPA.apply(
                            query, key, value, mask, past is None, offset)
                    else:
                        part = F.scaled_dot_product_attention(
                            query, key, value, attn_mask=mask,
                            is_causal=past is None, dropout_p=0.0)
                    parts.append(part)
            h = torch.cat(parts, dim=1)
            return backbone_project(self.o_proj, h.transpose(1, 2).reshape(b, length, -1)), (k, v)
        keys = k.repeat_interleave(self.h // self.kh, dim=1)
        values = v.repeat_interleave(self.h // self.kh, dim=1)
        # Short CUDA sequences must retain the pinned eager-oracle arithmetic.
        # SDPA's fused reductions differ enough near zero to fail the unchanged
        # 1e-4 hidden-state parity gate on actual pretrained Qwen weights.
        if q.is_cuda and k.shape[-2] <= 256:
            # BF16 CUDA matmul can change its reduction order when one suffix is
            # computed alone or beside another. Use the same small exact product
            # in both shapes, then retain the original BF16 rounding boundary.
            bf16 = q.dtype == keys.dtype == values.dtype == torch.bfloat16
            scores = (torch.matmul(q.double(), keys.transpose(-2, -1).double()).to(q.dtype)
                      if bf16 else torch.matmul(q, keys.transpose(-2, -1))) * (self.hd ** -0.5)
            allowed = torch.ones((length, k.shape[-2]), device=q.device,
                                 dtype=torch.bool).tril(diagonal=k.shape[-2] - length)
            scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
            probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            h = (torch.matmul(probabilities.double(), values.double()).to(q.dtype)
                 if bf16 else torch.matmul(probabilities, values))
            return backbone_project(self.o_proj, h.transpose(1, 2).reshape(b, length, -1)), (k, v)
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
        return backbone_project(self.o_proj, h.transpose(1, 2).reshape(b, length, -1)), (k, v)


class MLP(nn.Module):
    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        d, width = cfg.hidden_size, cfg.intermediate_size
        self.gate_proj = nn.Linear(d, width, bias=False)
        self.up_proj = nn.Linear(d, width, bias=False)
        self.down_proj = nn.Linear(width, d, bias=False)
    def forward(self, x: Tensor) -> Tensor:
        def project(rows: Tensor) -> Tensor:
            return backbone_project(self.down_proj,
                                    F.silu(backbone_project(self.gate_proj, rows))
                                    * backbone_project(self.up_proj, rows))

        if not (self.training and torch.is_grad_enabled() and x.is_cuda and x.shape[-2] > 1024):
            return project(x)
        # Each slice is recomputed in backward inside the existing layer
        # checkpoint. This bounds saved MLP activations at long lengths.
        from torch.utils.checkpoint import checkpoint
        return torch.cat([checkpoint(project, rows, use_reentrant=False)
                          for rows in x.split(1024, dim=-2)], dim=-2)


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

    def logits(self, decide: Tensor, options: Tensor, owner: Tensor | None = None) -> Tensor:
        device_type = decide.device.type
        bf16_autocast = (device_type in ("cpu", "cuda")
                         and torch.is_autocast_enabled(device_type)
                         and torch.get_autocast_dtype(device_type) == torch.bfloat16)
        if device_type in ("cpu", "cuda") and (decide.dtype == torch.bfloat16 or bf16_autocast):
            # BF16 GEMM changes rounding with batch shape even for identical
            # hidden rows. Keep the small pointer projection and reduction
            # stable across independent and parallel question execution.
            with torch.autocast(device_type, enabled=False):
                query = F.linear(decide.double(), self.q.weight.double(),
                                 self.q.bias.double() if self.q.bias is not None else None)
                keys = F.linear(options.double(), self.k.weight.double(),
                                self.k.bias.double() if self.k.bias is not None else None)
                return (keys * (query if owner is None else query[owner])).sum(-1).float() * self.scale
        query = self.q(decide)
        keys = self.k(options)
        return (keys * (query if owner is None else query[owner])).sum(-1).float() * self.scale

    def forward(self, decide: Tensor, options: Tensor) -> Tensor:
        return self.logits(decide, options)


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

    def load_qwen_weights(self, path: str | Path, *, expected_sha256: str | None = None):
        """Load one immutable source read, validating before parameter mutation.

        The optional expected digest is required by native-init's manifest route.
        A digest calculated from local bytes is integrity evidence, not proof of
        upstream origin. This byte-based path uses additional CPU memory rather
        than silently falling back to mutable memory-mapped source tensors.
        Call only on an exclusively owned model, not during concurrent serving.
        """
        import hashlib
        import re
        from safetensors.torch import load

        if expected_sha256 is not None and (
            not isinstance(expected_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
        ):
            raise ValueError("expected native weight checksum must be lowercase SHA-256")
        raw = Path(path).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError("native weight checksum mismatch")
        # Hash and deserialize the SAME immutable bytes. Hashing the pathname
        # after load_file could instead describe a replacement file.
        tensors = load(raw)
        del raw
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
            tensors = rewritten
        destinations = {k: v for k, v in self.backbone.state_dict().items()
                        if not k.endswith((".a", ".b"))}
        if set(tensors) != set(destinations):
            raise ValueError("native weight keys do not match architecture")
        # load_state_dict can copy earlier keys before a later shape error.
        # Validate every source tensor before it has permission to change any.
        for key, value in tensors.items():
            target = destinations[key]
            if value.shape != target.shape:
                raise ValueError("native weight shape does not match architecture: " + key)
            if not value.is_floating_point() or not target.is_floating_point():
                raise ValueError("native backbone weights must be floating point: " + key)
            if not bool(torch.isfinite(value).all()):
                raise ValueError("native backbone weights must be finite: " + key)
            # A finite wider-dtype input can become infinity during copy_.
            # Only wider-range source types require an explicit bound check.
            if torch.finfo(value.dtype).max > torch.finfo(target.dtype).max and value.numel() and (
                value.abs().max().item() > torch.finfo(target.dtype).max
            ):
                raise ValueError("native weight is outside destination dtype range: " + key)
        # Unexpected copy/device failures may still partially mutate parameters.
        # Never leave a prior native-success receipt attached to that state.
        self.native_weights_loaded = False
        self.loaded_backbone_sha256 = None
        self.backbone.load_state_dict(tensors, strict=not bool(self.cfg.lora_rank))
        self.loaded_backbone_sha256 = digest
        self.native_weights_loaded = True

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
        long_training = any(len(encoding.state) + len(branch.ids) >= 8192
                            for branch in encoding.branches)
        cache_element_size = (8 if self.training and torch.is_grad_enabled()
                              and long_training
                              and self.backbone.embed_tokens.weight.is_cuda
                              and torch.is_autocast_enabled("cuda")
                              and torch.get_autocast_dtype("cuda") == torch.bfloat16
                              else element_size)
        kv_rate = 2 * self.cfg.num_hidden_layers * self.cfg.num_key_value_heads * hd * cache_element_size
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
                scores = self.head.logits(hidden[row_index, decide_index], hidden[owner, end], owner)
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
