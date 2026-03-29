from __future__ import annotations

from multiprocessing.connection import Connection

import time
from typing import Any

import torch as t

from .gpt2 import (
    patch_gpt2_blocks_for_async,
    patch_gpt2_transformer_for_trimmed_sequences,
)
from .torch_varlen_attention import register_torch_varlen_attention
from .types import QueuedRequest, Request, WorkerResponse, WorkerSlot
from .const import DEVICE
from .engine import Engine, Job, Batch

from transformers import AutoModelForCausalLM, AutoTokenizer

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
        """Rebuild packed hidden state after optional worker rewrites.

        Shapes: ``module_tensor`` / ``residual_tensor`` are the full packed batch
        ``[1, total_seq, d_model]``. Per-job slices are ``[1, job_seq, d_model]``;
        we cache ``[job_seq, d_model]`` for apply_hooks / resume.

        If a job is paused at a different hookpoint than ``hookpoint``, it still
        advances with the default residual path; hooks registered on *earlier*
        layers are not re-run on a later forward pass until the job returns to
        its paused hookpoint (retry semantics).
        """

        hookpoint = f"transformer.h.{layer}.{hook_kind}"

        # 1) Execute hooks for all jobs at this hookpoint
        hook_jobs: list[Job] = []
        for job in self._batch.alive_jobs(hookpoint):
            start = job.idx_in_batch
            end = start + job.seq_len
            module_slice = module_tensor[:, start:end, :]

            if job.pending_hookpoint is None:
                job.worker.pipe.send(
                    {
                        "type": "apply_hooks",
                        "request_id": job.request.id,
                        "hookpoint": hookpoint,
                        "tensor": module_slice,
                    }
                )

            hook_jobs.append(job)

        # 2) Wait for all jobs to finish
        self._wait_for_hook_jobs(hook_jobs)

        # 3) Recombine the results
        next_chunks: list[t.Tensor] = []
        for job in self._batch.alive_jobs():
            # If the job isn't at this hookpoint, combine
            if hookpoint not in job.request.hooks:
                start = job.idx_in_batch
                end = start + job.seq_len
                module_slice = module_tensor[:, start:end, :]
                residual_slice = (
                    None
                    if residual_tensor is None
                    else residual_tensor[:, start:end, :]
                )
                next_chunks.append(
                    _compose_gpt2_residual(
                        hook_kind, module_slice, residual_slice
                    )
                )
                continue

            # Set jobs that didn't finish to dead and cache their activations
            if job.pending_response is None:
                job.is_alive = False
                job.pending_hookpoint = hookpoint
                job.pending_response = None
                # Cache 2d activations for recombine after worker write
                job.cached_module_tensor = (
                    None if hook_kind == "resid" else module_slice[0].clone()
                )
                job.cached_residual_tensor = (
                    None
                    if residual_slice is None
                    else residual_slice[0].clone()
                )
                continue

            # can't just get the module_slice,
            # cached tensor might be from a previous forward pass
            # so should just check whether
            hook_tensor = (
                job.cached_module_tensor
                if job.cached_module_tensor is not None
                else job.pending_response.tensor
            )

            residual_tensor = (
                job.pending_response.tensor
                if job.pending_response is not None
                else job.cached_residual_tensor
            )

            if hook_kind == "resid":
                # Back to [1, seq, d_model] for concat along sequence dim
                combined = residual_tensor.unsqueeze(0)
            else:
                combined = (hook_tensor + job.cached_residual_tensor).unsqueeze(
                    0
                )

            next_chunks.append(combined)

            job.pending_hookpoint = None
            job.pending_response = None
            job.cached_module_tensor = None
            job.cached_residual_tensor = None

        if not next_chunks:
            raise AllJobsPaused()

        # Create updated position ids
        next_position_ids = self._batch.packed_position_ids()
        next_position_ids = t.tensor(
            [next_position_ids],
            dtype=t.long,
            device=module_tensor.device,
        )

        self.model.transformer._batched_position_ids = next_position_ids
        self.model.transformer._batched_cache_position = next_position_ids[0]
        return t.cat(next_chunks, dim=1)

    def _wait_for_hook_jobs(self, jobs: list[Job]) -> None:
        if not jobs:
            return

        deadline = time.monotonic() + self._hook_wait_s
        while time.monotonic() < deadline:
            if self._poll_hook_jobs(jobs):
                break
            time.sleep(0.001)

        if not any(job.pending_response is not None for job in jobs):
            return

        grace_deadline = time.monotonic() + self._hook_wait_s
        while time.monotonic() < grace_deadline:
            self._poll_hook_jobs(jobs)
            if all(job.pending_response is not None for job in jobs):
                break
            time.sleep(0.001)

    def _poll_hook_jobs(self, jobs: list[Job]) -> bool:
        completed = False
        for job in jobs:
            if job.pending_response is not None:
                continue
            if not job.worker.pipe.poll(0):
                continue

            message = WorkerResponse.model_validate(job.worker.pipe.recv())
            assert message.type == "hooks_applied", (
                f"unexpected_worker_message:{message.type}"
            )
            assert message.request_id == job.request.id, (
                f"request_id_mismatch:{message.request_id}:{job.request.id}"
            )
            message.tensor = _normalize_worker_hook_tensor(message.tensor)
            job.pending_response = message
            completed = True

        return completed

    def _wait_for_any_pending_hook_job(self, jobs: list[Job]) -> None:
        pending_jobs = [
            job
            for job in jobs
            if job.pending_hookpoint is not None
            and job.pending_response is None
        ]
        assert pending_jobs, "no_pending_hook_jobs"
        while True:
            if self._poll_hook_jobs(pending_jobs):
                return
            time.sleep(0.001)

    def _finish_live_jobs(self, jobs: list[Job], status: str) -> None:
        if not jobs:
            return

        for job in jobs:
            job.worker.pipe.send(
                {
                    "type": "finish_request",
                    "request_id": job.request.id,
                }
            )

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
        queued_request = QueuedRequest(request=request)

        with self._condition:
            if self._closed:
                return "wrapper_closed"
            self._pending_requests.append(queued_request)
            self._condition.notify_all()

        queued_request.done.wait()
        return queued_request.status


__all__ = ["BatchedModel"]
