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

# %%

print(inputs)

# %%

def post_attn_hook(_module, _inputs, output):
    print(output[0].shape)
    return output

hook_ref = model.transformer.h[0].attn.register_forward_hook(post_attn_hook)

with t.inference_mode():
    outputs = model(**inputs)

hook_ref.remove()