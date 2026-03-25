# %%

from transformers import AutoModelForCausalLM, AutoTokenizer
from batched import register_torch_varlen_attention, pack_sequences_for_causal_lm
import torch as t

register_torch_varlen_attention("torch_varlen")

device = "cuda:0"

model_id = "openai-community/gpt2"
model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="torch_varlen").to(device).to(t.bfloat16)
tok = AutoTokenizer.from_pretrained(model_id)
# %%

prompts = [
    "Hello, how are you?",
    "Hello, how are you?",
]

inputs = pack_sequences_for_causal_lm(tok, prompts, device=device)

print(inputs["position_ids"].shape)

# %%

def post_attn_hook_zero(_module, _inputs, output):

    inputs["position_ids"] = inputs["position_ids"][:, :6]

    is_tuple = isinstance(output, tuple)
    if is_tuple:
        result = (
            output[0][:, :6],
            output[1:]
        )
    else:
        result = output[:, :6]
    
    return result


def post_attn_hook_one(_module, _inputs, output):
    print(output[0].shape)
    return output

hook_ref_zero = model.transformer.h[0].attn.register_forward_hook(post_attn_hook_zero)

hook_ref_one = model.transformer.h[0].attn.register_forward_hook(post_attn_hook_one)

with t.inference_mode():
    outputs = model(**inputs)

hook_ref_zero.remove()
hook_ref_one.remove()
# %%
