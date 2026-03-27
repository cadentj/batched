"""Smoke test: three concurrent requests; one hook sleeps 10s at layer 0 attn.

Run on a CUDA machine with torch varlen attention support. No pytest::

    python examples/async_three_jobs.py
"""

from __future__ import annotations

import threading
import time

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer

from batched import (
    Hook,
    Request,
    BatchedModel,
    register_torch_varlen_attention,
)


def main() -> None:
    model_name = "gpt2"
    register_torch_varlen_attention("torch_varlen")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="torch_varlen",
    ).to("cuda:0").to(t.bfloat16)
    model.eval()

    slow_hook = (
        "def hook(_module, _inputs, output):\n"
        "    import time\n"
        "    time.sleep(3)\n"
        "    return output\n"
    )

    requests = [
        Request(
            prompt="slow job",
            hooks={
                "transformer.h.0.attn": Hook(
                    hookpoint="transformer.h.0.attn",
                    op="write",
                    fn=slow_hook,
                ),
            },
        ),
        Request(prompt="fast a", hooks={}),
        Request(prompt="fast b", hooks={}),
    ]

    runtime = BatchedModel(
        model=model,
        tokenizer=tokenizer,
        n=3,
        batch_window_ms=500,
        hook_wait_ms=100,
    )

    results: list[tuple[str, str, float]] = []
    lock = threading.Lock()

    def run_one(req: Request) -> None:
        t0 = time.perf_counter()
        status = runtime.run_request(req)
        dt = time.perf_counter() - t0
        with lock:
            results.append((req.prompt, status, dt))
            print(results, flush=True)

    threads = [
        threading.Thread(target=run_one, args=(req,), name=f"req-{i}")
        for i, req in enumerate(requests)
    ]
    print("starting threads")
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    runtime.close()

    for prompt, status, dt in sorted(results, key=lambda x: x[0]):
        print(f"{prompt!r}: {status} ({dt:.1f}s)")

    assert all(status == "finished" for _, status, _ in results), results


if __name__ == "__main__":
    main()
