from __future__ import annotations

from multiprocessing.connection import Connection, wait
import time
from typing import Any

import torch as t
from transformers import AutoModelForCausalLM

from .const import DEVICE
from .engine import Batch, Engine, Job
from .gpt2 import (
    patch_gpt2_blocks_for_async,
    patch_gpt2_transformer_for_trimmed_sequences,
)
from .torch_varlen_attention import register_torch_varlen_attention
from .types import Request, WorkerRequest, WorkerResponse, WorkerSlot

_TORCH_VARLEN_ATTN_IMPL = "torch_varlen"


class AllJobsPaused(RuntimeError):
    """Internal control-flow signal used when no job is runnable in this pass."""


def _compose_gpt2_residual(
    hook_kind: str,
    module_slice: t.Tensor,
    residual_slice: t.Tensor | None,
) -> t.Tensor:
    if hook_kind == "resid":
        return module_slice
    assert residual_slice is not None
    return module_slice + residual_slice


def _normalize_worker_hook_tensor(x: Any) -> Any:
    if x is None or not isinstance(x, t.Tensor):
        return x
    if x.ndim == 3:
        return x[0]
    return x


def create_packed_input(jobs: list[Job]) -> dict[str, t.Tensor]:
    input_ids: list[int] = []
    position_ids: list[int] = []
    for job in jobs:
        input_ids.extend(job.input_ids)
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

        self._hook_wait_s = hook_wait_ms / 1000.0
        self._pass_jobs: list[Job] = []

        self._patch_model_for_async()

    def _patch_model_for_async(self) -> None:
        patch_gpt2_blocks_for_async(
            self.model.transformer,
            self._dispatch_gpt2_hook,
        )
        patch_gpt2_transformer_for_trimmed_sequences(self.model.transformer)

    def _run_forward_cycle(self, batch: Batch) -> None:
        self._pass_jobs = batch.live_jobs()
        if not self._pass_jobs:
            self._wait_for_any_pending_hook_job(batch.jobs)
            return

        try:
            inputs = create_packed_input(self._pass_jobs)
            with t.inference_mode():
                self.model(**inputs, use_cache=False)
            completed_jobs = [job for job in batch.jobs if job.is_alive]
            self._finish_live_jobs(completed_jobs, status="finished")
        except AllJobsPaused:
            self._wait_for_any_pending_hook_job(batch.jobs)
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

        next_chunks_by_job_id: dict[str, t.Tensor] = {}
        hook_jobs: list[Job] = []
        for job in live_jobs:
            start = job.idx_in_batch
            end = start + job.seq_len
            module_slice = module_tensor[:, start:end, :]
            residual_slice = (
                None
                if residual_tensor is None
                else residual_tensor[:, start:end, :]
            )

            if (
                job.pending_hookpoint is not None
                and job.pending_hookpoint != hookpoint
            ):
                next_chunks_by_job_id[job.request.id] = _compose_gpt2_residual(
                    hook_kind,
                    module_slice,
                    residual_slice,
                )
                continue

            if hookpoint not in job.request.hooks:
                next_chunks_by_job_id[job.request.id] = _compose_gpt2_residual(
                    hook_kind,
                    module_slice,
                    residual_slice,
                )
                continue

            hook_jobs.append(job)
            if (
                job.pending_hookpoint == hookpoint
                and job.computed_response is not None
            ):
                continue

            job.worker.pipe.send(
                WorkerRequest(
                    request_id=job.request.id,
                    req={
                        "type": "apply_hooks",
                        "hookpoint": hookpoint,
                        "tensor": module_slice,
                    },
                ).model_dump(mode="python")
            )

        self._wait_for_hook_jobs(hook_jobs)

        for job in hook_jobs:
            start = job.idx_in_batch
            end = start + job.seq_len
            module_slice = module_tensor[:, start:end, :]
            residual_slice = (
                None
                if residual_tensor is None
                else residual_tensor[:, start:end, :]
            )
            op = job.request.hooks[hookpoint].op

            if job.computed_response is None:
                job.is_alive = False
                job.pending_hookpoint = hookpoint
                job.cached_residual_tensor = (
                    None
                    if residual_slice is None
                    else residual_slice[0].clone()
                )
                continue

            if op == "read":
                next_chunks_by_job_id[job.request.id] = _compose_gpt2_residual(
                    hook_kind,
                    module_slice,
                    residual_slice,
                )
            else:
                assert isinstance(job.computed_response.tensor, t.Tensor)
                response_tensor = job.computed_response.tensor.to(
                    device=module_tensor.device,
                    dtype=module_tensor.dtype,
                )
                if hook_kind == "resid":
                    combined = response_tensor.unsqueeze(0)
                else:
                    if (
                        job.pending_hookpoint is not None
                        and job.cached_residual_tensor is not None
                    ):
                        residual_2d = job.cached_residual_tensor.to(
                            device=module_tensor.device,
                            dtype=module_tensor.dtype,
                        )
                    else:
                        assert residual_slice is not None
                        residual_2d = residual_slice[0]
                    combined = (response_tensor + residual_2d).unsqueeze(0)

                next_chunks_by_job_id[job.request.id] = combined

            job.is_alive = True
            job.pending_hookpoint = None
            job.computed_response = None
            job.cached_residual_tensor = None

        next_chunks = [
            next_chunks_by_job_id[job.request.id]
            for job in live_jobs
            if job.request.id in next_chunks_by_job_id
        ]
        if not next_chunks:
            raise AllJobsPaused()

        next_position_ids = self._batch.packed_position_ids(alive_only=True)
        next_position_ids_tensor = t.tensor(
            [next_position_ids],
            dtype=t.long,
            device=module_tensor.device,
        )
        self.model.transformer._batched_position_ids = next_position_ids_tensor
        return t.cat(next_chunks, dim=1)

    def _wait_for_hook_jobs(self, jobs: list[Job]) -> None:
        if not jobs:
            return

        # TODO(cadentj): enforce max request runtime and recycle stuck workers.
        pending: dict[Connection, Job] = {
            job.worker.pipe: job
            for job in jobs
            if job.computed_response is None
        }
        if not pending:
            return

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

            if len(pending) < len(jobs) and not adjusted:
                deadline = time.monotonic() + self._hook_wait_s
                adjusted = True

    def _wait_for_any_pending_hook_job(self, jobs: list[Job]) -> None:
        pending: dict[Connection, Job] = {
            job.worker.pipe: job
            for job in jobs
            if job.pending_hookpoint is not None
            and job.computed_response is None
        }
        if not pending:
            return

        ready = wait(pending.keys())
        for pipe in ready:
            self._recv_hook_response(pending[pipe])

    def _recv_hook_response(self, job: Job) -> None:
        message = WorkerResponse.model_validate(job.worker.pipe.recv())
        assert message.type == "hooks_applied"
        assert message.request_id == job.request.id
        message.tensor = _normalize_worker_hook_tensor(message.tensor)
        job.computed_response = message
        if job.pending_hookpoint is not None:
            job.is_alive = True

    def _finish_live_jobs(self, jobs: list[Job], status: str) -> None:
        if not jobs:
            return

        for job in jobs:
            if job.request.status == "queued":
                job.request.status = status
            job.request.done.set()

        finished_ids = {job.request.id for job in jobs}
        with self._condition:
            self._batch.jobs = [
                job
                for job in self._batch.jobs
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
        return WorkerResponse.model_validate(worker.pipe.recv())

    def __call__(self, request: Request) -> str:
        with self._condition:
            if self._closed:
                return "wrapper_closed"
            self._pending_requests.append(request)
            self._condition.notify_all()

        request.done.wait()
        return request.status


__all__ = ["BatchedModel"]
