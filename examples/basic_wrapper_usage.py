from __future__ import annotations

from batched import Hook, Request, WrapperModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    runtime = WrapperModel(
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
