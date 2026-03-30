from __future__ import annotations

from multiprocessing.connection import Connection
import os
from typing import Any

import torch as t
import torch.multiprocessing as mp

from .types import (
    CompiledHook,
    WorkerRequest,
    WorkerRequestRuntime,
    WorkerResponse,
)


class InterventionWorker:
    def __init__(
        self,
        worker_id: int,
        pipe: Connection,
        shutdown: mp.synchronize.Event,
    ):
        self.worker_id = worker_id
        self.pipe = pipe
        self.shutdown = shutdown
        self._current_request_id: str | None = None
        self._current_request_runtime: WorkerRequestRuntime | None = None

    def _compile_hook(
        self,
        source: str,
        request_id: str,
        hookpoint: str,
        scope: dict[str, Any],
    ):
        scope.pop("hook", None)
        filename = f"<hook:{request_id}:{hookpoint}>"
        code = compile(source, filename, "exec")
        exec(code, scope, scope)
        hook_fn = scope.get("hook")
        if not callable(hook_fn):
            raise ValueError("missing_hook")
        return hook_fn

    def _start_request(self, message: WorkerRequest) -> None:
        self._current_request_id = None
        self._current_request_runtime = None
        scope: dict[str, Any] = {"torch": t, "t": t, "state": {}}

        compiled_hooks: dict[str, CompiledHook] = {}
        for hookpoint, hook in message.hooks.items():
            hook_fn = self._compile_hook(
                source=hook.fn,
                request_id=message.request_id,
                hookpoint=hookpoint,
                scope=scope,
            )
            compiled_hooks[hookpoint] = CompiledHook(
                hookpoint=hookpoint,
                op=hook.op,
                fn=hook_fn,
            )

        self._current_request_id = message.request_id
        self._current_request_runtime = WorkerRequestRuntime(
            hooks=compiled_hooks,
            scope=scope,
        )
        self.pipe.send(
            WorkerResponse(
                type="request_started",
                request_id=message.request_id,
            )
        )

    def _apply_hooks(self, message: WorkerRequest) -> None:
        if self._current_request_runtime is None:
            raise RuntimeError(f"missing_runtime:{message.request_id}")
        if not isinstance(message.tensor, t.Tensor):
            raise TypeError("missing_tensor")

        hookpoint = message.hookpoint
        hook = self._current_request_runtime.hooks[hookpoint]
        if hook.op == "read":
            hook.fn(None, None, message.tensor)
            result = None
        else:
            result = hook.fn(None, None, message.tensor)
            if not isinstance(result, t.Tensor):
                raise TypeError("write_hook_must_return_tensor")
            # If a hook returns the original IPC CUDA tensor, we must allocate
            # fresh storage before sending it back across processes.
            result = result.clone()

        self.pipe.send(
            WorkerResponse(
                type="hooks_applied",
                request_id=message.request_id,
                tensor=result,
            )
        )

    def run(self) -> None:
        self.pipe.send({"type": "worker_ready", "worker_id": self.worker_id})

        while not self.shutdown.is_set():
            if not self.pipe.poll(0.05):
                continue

            message = WorkerRequest.model_validate(self.pipe.recv())

            if message.type == "start_request":
                self._start_request(message)
                continue
            if message.type == "apply_hooks":
                self._apply_hooks(message)
                continue
            if message.type == "shutdown":
                return


def _worker_main(
    worker_id: int,
    pipe: Connection,
    shutdown: mp.synchronize.Event,
    worker_memory_fraction: float | None,
    cuda_mps_pipe_directory: str | None,
    cuda_mps_active_thread_percentage: int | None,
) -> None:
    if cuda_mps_pipe_directory:
        os.environ["CUDA_MPS_PIPE_DIRECTORY"] = cuda_mps_pipe_directory
    if cuda_mps_active_thread_percentage is not None:
        os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(
            cuda_mps_active_thread_percentage
        )

    if worker_memory_fraction is not None and t.cuda.is_available():
        if hasattr(t.cuda, "memory") and hasattr(
            t.cuda.memory, "set_per_process_memory_fraction"
        ):
            t.cuda.memory.set_per_process_memory_fraction(worker_memory_fraction)
        else:
            t.cuda.set_per_process_memory_fraction(worker_memory_fraction)

    InterventionWorker(worker_id=worker_id, pipe=pipe, shutdown=shutdown).run()


__all__ = ["InterventionWorker", "_worker_main"]
