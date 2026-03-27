from __future__ import annotations

import torch as t
from batched import Hook, Request, BatchedModel, register_torch_varlen_attention
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    device = "cuda" if t.cuda.is_available() else "cpu"
    model_name = "gpt2"
    register_torch_varlen_attention("torch_varlen")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="torch_varlen",
    ).to(device).to(t.bfloat16)

    runtime = BatchedModel(
        model=model,
        tokenizer=tokenizer,
        n=1,
        batch_window_ms=0,
    )

    try:
        request = Request(
            prompt="Hello from batched",
            hooks={
                "transformer.h.0.mlp": Hook(
                    hookpoint="transformer.h.0.mlp",
                    op="read",
                    fn=(
                        "def hook(_module, _inputs, output):\n"
                        "    state['mean_activation'] = float(output.mean())\n"
                    ),
                )
            },
        )
        status = runtime.run_request(request)
        print(f"run_request status: {status}")
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
