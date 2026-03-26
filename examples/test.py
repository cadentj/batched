# %%

import types

from transformers import AutoModelForCausalLM, AutoTokenizer
from batched import (
    register_torch_varlen_attention,
    pack_sequences_for_causal_lm,
)
import torch as t
from batched.gpt2 import (
    create_gpt2_forward,
    patch_gpt2_transformer_for_trimmed_sequences,
    Batch,
    Job,
)

register_torch_varlen_attention("torch_varlen")

device = "cuda:0"
model_id = "openai-community/gpt2"

model = (
    AutoModelForCausalLM.from_pretrained(
        model_id, attn_implementation="torch_varlen"
    )
    .to(device)
    .to(t.bfloat16)
)
batch = Batch(jobs=[])

custom_forward = create_gpt2_forward(batch)
for block in model.transformer.h:
    block.forward = types.MethodType(custom_forward, block)
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

batch.jobs.append(Job(
    start_idx=0,
    seq_lens=[6],
    status="queued",
))
batch.jobs.append(Job(
    start_idx=6,
    seq_lens=[6],
    status="queued",
))


def post_attn_hook_one(_module, _inputs, output):
    batch.jobs[0].status = "paused"
    return output

def post_attn_hook_three(_module, _inputs, output):
    print(output[0].shape)
    return output

hook_ref_one = model.transformer.h[1].attn.register_forward_hook(
    post_attn_hook_one
)

hook_ref_three = model.transformer.h[3].attn.register_forward_hook(
    post_attn_hook_three
)

with t.inference_mode():
    outputs = model(**inputs)
    print(outputs.logits.shape)

hook_ref_one.remove()
hook_ref_three.remove()
# %%
