# %%
from __future__ import annotations

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer

from batched import (
    pack_sequences_for_causal_lm,
    register_torch_varlen_attention,
)
from batched.gpt2 import (
    patch_gpt2_blocks_for_async,
    patch_gpt2_transformer_for_trimmed_sequences,
)

register_torch_varlen_attention("torch_varlen")

device = "cuda:0"
model_id = "openai-community/gpt2"

model = (
    AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation="torch_varlen",
    )
    .to(device)
    .to(t.bfloat16)
)


def passthrough_dispatch(
    _layer: int,
    hook_kind: str,
    module_tensor: t.Tensor,
    residual_tensor: t.Tensor | None,
) -> t.Tensor:
    if hook_kind == "resid":
        return module_tensor
    return module_tensor + residual_tensor


patch_gpt2_blocks_for_async(model.transformer, passthrough_dispatch)
patch_gpt2_transformer_for_trimmed_sequences(model.transformer)

tok = AutoTokenizer.from_pretrained(model_id)

# %%
prompts = [
    "Hello, how are you?",
    "Hello, how are you?",
]
inputs = pack_sequences_for_causal_lm(tok, prompts, device=device)
print(inputs["position_ids"].shape)

# %%
with t.inference_mode():
    outputs = model(**inputs, use_cache=False)
    print(outputs.logits.shape)

# %%
