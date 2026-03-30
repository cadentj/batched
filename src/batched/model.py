from __future__ import annotations

from multiprocessing.connection import wait, Connection

import time
from typing import Any

import torch as t

from .gpt2 import (
    patch_gpt2_blocks_for_async,
    patch_gpt2_transformer_for_trimmed_sequences,
)
from .torch_varlen_attention import register_torch_varlen_attention
from .types import Request, WorkerResponse, WorkerSlot
from .const import DEVICE
from .engine import Engine, Job, Batch

from transformers import AutoModelForCausalLM

_TORCH_VARLEN_ATTN_IMPL = "torch_varlen"


class AllJobsPaused(RuntimeError):
    """Internal control-flow signal used when no job is runnable in this pass."""


def _compose_gpt2_residual(
    hook_kind: str,
    module_slice: t.Tensor,
    residual_slice: t.Tensor | None,
) -> t.Tensor:
    """Default block output before hook rewrite: packed slice is [1, seq, d_model]."""
    if hook_kind == "resid":
        return module_slice
    assert residual_slice is not None
    return module_slice + residual_slice


def _normalize_worker_hook_tensor(x: Any) -> Any:
    """Write hooks receive the same slice we send: [1, seq, d_model]; normalize to [seq, d_model]."""
    if x is None or not isinstance(x, t.Tensor):
        return x
    if x.ndim == 3:
        return x[0]
    return x


def create_packed_input(batch: Batch) -> dict[str, t.Tensor]:
    input_ids: list[int] = []
    position_ids: list[int] = []
    for job in batch.jobs:
        input_ids.extend(job.token_ids)
        position_ids.extend(range(job.seq_len))

    return {
        "input_ids": t.tensor([input_ids], dtype=t.long, device=DEVICE),
        "position_ids": t.tensor(
            [position_ids],
            dtype=t.long,
            device=DEVICE,
        ),
    }


class BatchedModel(Engine):
    def __init__(
        self,
        model: AutoModelForCausalLM,
        n: int = 1,
        batch_window_ms: int = 25,
        hook_wait_ms: int = 50,
        worker_memory_fraction: float | None = None,
        cuda_mps_pipe_directory: str | None = None,
        cuda_mps_active_thread_percentage: int | None = None,
    ):
        super().__init__(
            n=n,
            batch_window_ms=batch_window_ms,
            worker_memory_fraction=worker_memory_fraction,
            cuda_mps_pipe_directory=cuda_mps_pipe_directory,
            cuda_mps_active_thread_percentage=cuda_mps_active_thread_percentage,
        )
        assert hook_wait_ms >= 0, "hook_wait_ms must be non-negative"

        self.model = model
        self.model.eval()
        register_torch_varlen_attention(_TORCH_VARLEN_ATTN_IMPL)

        self.hook_wait_ms = hook_wait_ms
        self._hook_wait_s = hook_wait_ms / 1000.0

        self._patch_model_for_async()

    def _patch_model_for_async(self) -> None:
        patch_gpt2_blocks_for_async(
            self.model.transformer,
            self._dispatch_gpt2_hook,
        )
        patch_gpt2_transformer_for_trimmed_sequences(self.model.transformer)

    def _run_forward_cycle(self, batch: Batch) -> None:
        try:
            inputs = create_packed_input(batch)
            with t.inference_mode():
                self.model(**inputs, use_cache=False)
            completed_jobs = list(self._pass_jobs)
            self._finish_live_jobs(completed_jobs, status="finished")
        except AllJobsPaused:
            self._wait_for_any_pending_hook_job(self._pass_jobs)
        finally:
            self._pass_jobs = []

    def _dispatch_gpt2_hook(
        self,
        layer: int,
        hook_kind: str,
        module_tensor: t.Tensor,
        residual_tensor: t.Tensor | None,
    ) -> t.Tensor:
        hookpoint = f"transformer.h.{layer}.{hook_kind}"
        live_jobs = self._batch.live_jobs()

        # 1) Execute jobs at this hookpoint
        next_chunks: list[t.Tensor] = []
        sent_jobs: list[Job] = []
        for job in live_jobs:
            start = job.idx_in_batch
            end = start + job.seq_len
            module_slice = module_tensor[:, start:end, :]
            residual_slice = (
                None
                if residual_tensor is None
                else residual_tensor[:, start:end, :]
            )

            if hookpoint not in job.request.hooks:
                next_chunks.append(
                    _compose_gpt2_residual(
                        hook_kind, module_slice, residual_slice
                    )
                )

            if hookpoint in job.request.hooks:
                job.worker.pipe.send(
                    {
                        "type": "apply_hooks",
                        "request_id": job.request.id,
                        "hookpoint": hookpoint,
                        "tensor": module_slice,
                    }
                )
                sent_jobs.append(job)

        # 2) Wait for all jobs to finish
        self._wait_for_hook_jobs(sent_jobs)

        # 3) Recombine the results
        for job in sent_jobs:
            op = job.request.hooks[hookpoint].op

            # A) Set jobs that didn't finish to dead and cache their activations
            if job.pending_hookpoint is None and job.pending_response is None:
                job.is_alive = False
                job.pending_hookpoint = hookpoint
                job.pending_response = None
                job.cached_residual_tensor = (
                    None
                    if residual_slice is None
                    else residual_slice[0].clone()
                )
                continue

            start = job.idx_in_batch
            end = start + job.seq_len
            module_slice = module_tensor[:, start:end, :]
            residual_slice = (
                None
                if residual_tensor is None
                else residual_tensor[:, start:end, :]
            )

            assert isinstance(residual_slice, t.Tensor)

            # B) For read jobs, append
            if op == "read":
                next_chunks.append(
                    _compose_gpt2_residual(
                        hook_kind, module_slice, residual_slice
                    )
                )
                continue

            # Write jobs have a pending response
            assert job.pending_response is not None

            # C) For jobs w pending hook points, combine
            if job.pending_hookpoint is not None:
                if hook_kind == "resid":
                    combined = job.pending_response.tensor.unsqueeze(0)
                else:
                    combined = (
                        job.pending_response.tensor + job.cached_residual_tensor
                    ).unsqueeze(0)

            # D) For jobs without pending hooks, use current residual tensor
            if job.pending_hookpoint is None:
                if hook_kind == "resid":
                    combined = job.pending_response.tensor.unsqueeze(0)

                else:
                    combined = (
                        job.pending_response.tensor + residual_slice
                    ).unsqueeze(0)

            next_chunks.append(combined)

            job.pending_hookpoint = None
            job.pending_response = None
            job.cached_residual_tensor = None

        if not next_chunks:
            raise AllJobsPaused()

        # Create updated position ids
        # NOTE(cadentj): This is wrong.
        next_position_ids = self._batch.packed_position_ids()
        next_position_ids = t.tensor(
            [next_position_ids],
            dtype=t.long,
            device=module_tensor.device,
        )

        self.model.transformer._batched_position_ids = next_position_ids
        # self.model.transformer._batched_cache_position = next_position_ids[0]
        return t.cat(next_chunks, dim=1)

    def _wait_for_hook_jobs(self, jobs: list[Job]) -> None:
        if not jobs:
            return

        pending: dict[Connection, Job] = {
            job.worker.pipe: job for job in jobs
        }

        adjusted = False
        deadline = time.monotonic() + self._hook_wait_s

        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            ready = wait(pending.keys(), timeout=remaining)
            for pipe in ready:
                job = pending.pop(pipe)
                self._recv_hook_response(job)

            # If at least one completed, wait a bit longer
            if len(pending) < len(jobs) and not adjusted:
                deadline = time.monotonic() + self._hook_wait_s
                adjusted = True

    def _wait_for_any_pending_hook_job(self, jobs: list[Job]) -> None:
        pending: dict[Connection, Job] = {
            job.worker.pipe: job
            for job in jobs
            if job.pending_hookpoint is not None
            and job.pending_response is None
        }
        assert pending, "no_pending_hook_jobs"

        # blocks until at least one pipe is ready — no sleep loop
        ready: list[Connection] = wait(pending.keys())  # type: ignore
        for pipe in ready:
            self._recv_hook_response(pending[pipe])

    def _recv_hook_response(self, job: Job) -> None:
        message = WorkerResponse.model_validate(job.worker.pipe.recv())
        assert message.type == "hooks_applied"
        assert message.request_id == job.request.id
        message.tensor = _normalize_worker_hook_tensor(message.tensor)
        job.pending_response = message

    def _finish_live_jobs(self, jobs: list[Job], status: str) -> None:
        if not jobs:
            return

        for job in jobs:
            while True:
                message = self._recv_request_message(
                    job.worker,
                    timeout_s=10.0,
                )
                if message.type == "hooks_applied":
                    message.tensor = _normalize_worker_hook_tensor(
                        message.tensor
                    )
                    job.pending_response = message
                    continue
                if message.type != "request_finished":
                    raise RuntimeError(
                        f"unexpected_worker_message:{message.type}"
                    )
                break

            if job.queued_request.status == "queued":
                job.queued_request.status = status
            job.queued_request.done.set()

        finished_ids = {job.request.id for job in jobs}
        with self._condition:
            self._live_jobs = [
                job
                for job in self._live_jobs
                if job.request.id not in finished_ids
            ]
            for job in jobs:
                self._idle_workers.append(job.worker)
            self._condition.notify_all()

    def _recv_request_message(
        self,
        worker: WorkerSlot,
        timeout_s: float = 2.0,
    ) -> WorkerResponse:
        if not worker.pipe.poll(timeout_s):
            raise RuntimeError("worker_timeout")
        message = WorkerResponse.model_validate(worker.pipe.recv())
        return message

    def __call__(self, request: Request) -> str:
        with self._condition:
            if self._closed:
                return "wrapper_closed"
            self._pending_requests.append(request)
            self._condition.notify_all()

        request.done.wait()
        return request.status


__all__ = ["BatchedModel"]
