"""
Basic Transformers attention implementation using `torch.nn.attention.varlen.varlen_attn`.

- Expects CUDA and a PyTorch build with `torch.nn.attention.varlen`. 
- The `dropout` argument must be zero. 
- Requires packed inputs. So `batch_size == 1`. 
  Position IDs which restart at 0 per concatenated segment.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch.nn.attention.varlen import varlen_attn

from transformers.integrations.sdpa_attention import repeat_kv
from transformers.modeling_flash_attention_utils import (
    _prepare_from_posids,
    fa_peft_integration_check,
    flash_attn_supports_top_left_mask,
)
from transformers.pytorch_utils import Conv1D

def _get_target_dtype(query: torch.Tensor, module: torch.nn.Module) -> torch.dtype | None:
    """Like `get_target_dtype` from transformers.integrations.flash_attention, 
    but GPT-2 attention uses `Conv1D`, not `nn.Linear`."""
    if query.dtype != torch.float32:
        return None
    if torch.is_autocast_enabled("cuda"):
        return torch.get_autocast_dtype("cuda")
    if hasattr(module.config, "_is_quantized"):
        return module.config.dtype
    for layer in module.modules():
        if isinstance(layer, (torch.nn.Linear, Conv1D)):
            return layer.weight.dtype
    return None


def _apply_custom_scaling(query: torch.Tensor, scaling: float | None) -> torch.Tensor:
    if scaling is None:
        return query
    head_dim = query.shape[-1]
    default = head_dim**-0.5
    if math.isclose(scaling, default, rel_tol=1e-5, abs_tol=1e-8):
        return query
    return query * (scaling / default)


def _varlen_causal_for_flash(is_causal: bool, query_length: int) -> bool:
    return is_causal and not (flash_attn_supports_top_left_mask() and query_length == 1)


def torch_varlen_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    batch = query.size(0)
    kv_seq_len = key.size(1)
    position_ids = kwargs.get("position_ids")

    assert batch == 1, "batch size must be 1 (packed inputs)"
    assert kv_seq_len == query.size(1), "kv caching not supported yet"
    assert position_ids is not None, "position ids required"

    assert kwargs.get("dropout", 0.0) == 0.0, "attn dropout not supported"
    assert kwargs.get("attention_mask") is None, "varlen attn doesn't use attention mask"

    seq_len = query.shape[2]
    is_causal = (
        is_causal if is_causal is not None else getattr(module, "is_causal", True)
    )
    varlen_is_causal = _varlen_causal_for_flash(is_causal, seq_len)

    query = query.transpose(1, 2).contiguous()
    key = key.transpose(1, 2).contiguous()
    value = value.transpose(1, 2).contiguous()

    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

    target_dtype = _get_target_dtype(query, module)
    query, key, value = fa_peft_integration_check(query, key, value, target_dtype)

    q, k, v, (cu_q, cu_k), (max_q, max_k) = _prepare_from_posids(
        query, key, value, position_ids
    )
    q = _apply_custom_scaling(q, scaling)
    out = varlen_attn(q, k, v, cu_q, cu_k, max_q, max_k, is_causal=varlen_is_causal)
    out = out[0] if isinstance(out, tuple) else out
    out = out.view(batch, -1, out.size(-2), out.size(-1))

    return out.contiguous(), None


def register_torch_varlen_attention(name: str) -> None:
    from transformers.modeling_utils import AttentionInterface

    AttentionInterface.register(name, torch_varlen_attention_forward)


__all__ = [
    "pack_sequences_for_causal_lm",
    "register_torch_varlen_attention",
    "torch_varlen_attention_forward",
]
