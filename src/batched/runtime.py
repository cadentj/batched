from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from multiprocessing.connection import Connection
import re
import threading
import time
from typing import Any

import torch as t
import torch.multiprocessing as mp

from .gpt2 import (
    patch_gpt2_blocks_for_async,
    patch_gpt2_transformer_for_trimmed_sequences,
)
from .types import QueuedRequest, Request, WorkerResponse, WorkerSlot
from .worker import _worker_main

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ModuleNotFoundError:
    AutoModelForCausalLM = Any
    AutoTokenizer = Any


_HOOKPOINT_RE = re.compile(r"^transformer\.h\.(\d+)\.(attn|mlp|resid)$")


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


@dataclass
class LiveJob:
    worker: WorkerSlot
    queued_request: QueuedRequest
    token_ids: list[int]
    seq_len: int
    pass_start_idx: int = 0
    pending_hookpoint: str | None = None
    pending_response: WorkerResponse | None = None
    cached_module_tensor: t.Tensor | None = None
    cached_residual_tensor: t.Tensor | None = None

    @property
    def request(self) -> Request:
        return self.queued_request.request


class WrapperModel:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        n: int = 1,
        batch_window_ms: int = 25,
        hook_wait_ms: int = 50,
        worker_memory_fraction: float | None = None,
        cuda_mps_pipe_directory: str | None = None,
        cuda_mps_active_thread_percentage: int | None = None,
    ):
        assert n > 0, "n must be positive"
        assert batch_window_ms >= 0, "batch_window_ms must be non-negative"
        assert hook_wait_ms >= 0, "hook_wait_ms must be non-negative"
        if worker_memory_fraction is not None:
            assert 0.0 < worker_memory_fraction <= 1.0, (
                "worker_memory_fraction must be in the range (0, 1]"
            )
        if cuda_mps_active_thread_percentage is not None:
            assert 1 <= cuda_mps_active_thread_percentage <= 100, (
                "cuda_mps_active_thread_percentage must be in the range [1, 100]"
            )

        self.model = model
        self.model.eval()
        self._validate_model_is_gpt2()

        self.batch_window_ms = batch_window_ms
        self.hook_wait_ms = hook_wait_ms
        self._hook_wait_s = hook_wait_ms / 1000.0
        self._tokenizer: AutoTokenizer = tokenizer
        self._num_layers = len(self.model.transformer.h)

        # CUDA tensor IPC between processes requires spawn-safe workers.
        self._mp_context = mp.get_context("spawn")
        self._shutdown = self._mp_context.Event()
        self._condition = threading.Condition()
        self._closed = False
        self._pending_requests: deque[QueuedRequest] = deque()
        self._live_jobs: list[LiveJob] = []
        self._pass_jobs: list[LiveJob] = []

        self._patch_model_for_async()
        self._workers = self._spawn_workers(
            n,
            worker_memory_fraction=worker_memory_fraction,
            cuda_mps_pipe_directory=cuda_mps_pipe_directory,
            cuda_mps_active_thread_percentage=cuda_mps_active_thread_percentage,
        )
        self._idle_workers: deque[WorkerSlot] = deque(self._workers)
        self._scheduler = threading.Thread(
            target=self._scheduler_loop,
            name="wrapper-model-scheduler",
            daemon=True,
        )
        self._scheduler.start()

    def _validate_model_is_gpt2(self) -> None:
        model_type = getattr(
            getattr(self.model, "config", None),
            "model_type",
            None,
        )
        assert model_type == "gpt2", "only_gpt2_supported"

    def _patch_model_for_async(self) -> None:
        patch_gpt2_blocks_for_async(
            self.model.transformer,
            self._dispatch_gpt2_hook,
        )
        patch_gpt2_transformer_for_trimmed_sequences(self.model.transformer)

    def _spawn_workers(
        self,
        num_workers: int,
        worker_memory_fraction: float | None,
        cuda_mps_pipe_directory: str | None,
        cuda_mps_active_thread_percentage: int | None,
    ) -> list[WorkerSlot]:
        workers: list[WorkerSlot] = []
        for worker_id in range(num_workers):
            parent_pipe, child_pipe = self._mp_context.Pipe()
            process = self._mp_context.Process(
                target=_worker_main,
                args=(
                    worker_id,
                    child_pipe,
                    self._shutdown,
                    worker_memory_fraction,
                    cuda_mps_pipe_directory,
                    cuda_mps_active_thread_percentage,
                ),
            )
            process.daemon = True
            process.start()
            child_pipe.close()
            self._wait_for_worker_ready(parent_pipe, worker_id)
            workers.append(
                WorkerSlot(
                    worker_id=worker_id,
                    pipe=parent_pipe,
                    process=process,
                )
            )
        return workers

    def _wait_for_worker_ready(self, pipe: Connection, worker_id: int) -> None:
        if not pipe.poll(10.0):
            raise RuntimeError(f"worker_start_timeout:{worker_id}")
        message = pipe.recv()
        if (
            message.get("type") != "worker_ready"
            or message.get("worker_id") != worker_id
        ):
            raise RuntimeError(f"unexpected_worker_ready_message:{message}")

    def _scheduler_loop(self) -> None:
        while True:
            live_jobs = self._next_cycle_jobs()
            if live_jobs is None:
                return
            if not live_jobs:
                continue

            self._run_forward_cycle(live_jobs)

    def _next_cycle_jobs(self) -> list[LiveJob] | None:
        while True:
            assignments: list[tuple[QueuedRequest, WorkerSlot]] = []
            with self._condition:
                while True:
                    assignments = self._claim_pending_requests_locked()
                    if assignments:
                        break

                    if self._closed and not self._live_jobs:
                        return None

                    if self._live_jobs:
                        return list(self._live_jobs)

                    self._condition.wait(timeout=0.05)

            new_jobs: list[LiveJob] = []
            for queued_request, worker in assignments:
                new_jobs.append(
                    self._start_live_job(
                        queued_request=queued_request,
                        worker=worker,
                    )
                )

            with self._condition:
                self._live_jobs.extend(new_jobs)
                if self._closed and not self._live_jobs:
                    return None
                if self._live_jobs:
                    return list(self._live_jobs)

    def _claim_pending_requests_locked(
        self,
    ) -> list[tuple[QueuedRequest, WorkerSlot]]:
        if self._closed:
            return []
        if not self._pending_requests or not self._idle_workers:
            return []

        deadline = time.monotonic() + (self.batch_window_ms / 1000.0)
        target_batch_size = len(self._idle_workers)
        while not self._closed and len(self._pending_requests) < target_batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            self._condition.wait(timeout=remaining)
            target_batch_size = len(self._idle_workers)
            if target_batch_size == 0:
                break

        batch_size = min(
            len(self._pending_requests),
            len(self._idle_workers),
        )
        if batch_size == 0:
            return []

        assignments: list[tuple[QueuedRequest, WorkerSlot]] = []
        for _ in range(batch_size):
            assignments.append(
                (
                    self._pending_requests.popleft(),
                    self._idle_workers.popleft(),
                )
            )
        return assignments

    def _start_live_job(
        self,
        queued_request: QueuedRequest,
        worker: WorkerSlot,
    ) -> LiveJob:
        request = queued_request.request
        self._validate_request_hooks(request)
        token_ids = self._tokenizer(
            request.prompt,
            add_special_tokens=False,
        )["input_ids"]
        assert len(token_ids) > 0, "empty_prompt_tokens"

        worker.pipe.send(
            {
                "type": "start_request",
                "request_id": request.id,
                "hooks": request.dump_hooks(),
            }
        )
        message = self._recv_request_message(worker, timeout_s=5.0)
        assert message.type == "request_started", (
            f"unexpected_worker_message:{message.type}"
        )

        return LiveJob(
            worker=worker,
            queued_request=queued_request,
            token_ids=token_ids,
            seq_len=len(token_ids),
        )

    def _validate_request_hooks(self, request: Request) -> None:
        for hookpoint, hook in request.hooks.items():
            assert hookpoint == hook.hookpoint, "hookpoint_key_mismatch"
            match = _HOOKPOINT_RE.match(hookpoint)
            assert match is not None, f"invalid_hookpoint:{hookpoint}"
            layer = int(match.group(1))
            assert 0 <= layer < self._num_layers, (
                f"invalid_layer:{layer}"
            )

    def _run_forward_cycle(self, live_jobs: list[LiveJob]) -> None:
        self._pass_jobs = list(live_jobs)
        self._assign_pass_offsets(self._pass_jobs)

        try:
            inputs = self._pack_pass_inputs(self._pass_jobs)
            with t.inference_mode():
                self.model(**inputs, use_cache=False)
            completed_jobs = list(self._pass_jobs)
            self._finish_live_jobs(completed_jobs, status="finished")
        finally:
            self._pass_jobs = []

    def _assign_pass_offsets(self, jobs: list[LiveJob]) -> None:
        cursor = 0
        for job in jobs:
            job.pass_start_idx = cursor
            cursor += job.seq_len

    def _pack_pass_inputs(self, jobs: list[LiveJob]) -> dict[str, t.Tensor]:
        input_ids: list[int] = []
        position_ids: list[int] = []
        for job in jobs:
            input_ids.extend(job.token_ids)
            position_ids.extend(range(job.seq_len))

        device = self._model_device()
        return {
            "input_ids": t.tensor([input_ids], dtype=t.long, device=device),
            "position_ids": t.tensor(
                [position_ids],
                dtype=t.long,
                device=device,
            ),
        }

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
        jobs = self._pass_jobs

        hook_jobs: list[LiveJob] = []
        for job in jobs:
            start = job.pass_start_idx
            end = start + job.seq_len
            # [1, job_seq, d_model] — one batch row, this job's token span
            module_slice = module_tensor[:, start:end, :]
            residual_slice = (
                None
                if residual_tensor is None
                else residual_tensor[:, start:end, :]
            )

            if hookpoint not in job.request.hooks:
                continue

            if (
                job.pending_hookpoint is not None
                and job.pending_hookpoint != hookpoint
            ):
                continue

            if job.pending_hookpoint is None:
                job.pending_hookpoint = hookpoint
                job.pending_response = None
                # Cache 2d activations for recombine after worker write
                job.cached_module_tensor = module_slice[0].clone()
                job.cached_residual_tensor = (
                    None
                    if residual_slice is None
                    else residual_slice[0].clone()
                )
                job.worker.pipe.send(
                    {
                        "type": "apply_hooks",
                        "request_id": job.request.id,
                        "hookpoint": hookpoint,
                        "tensor": module_slice,
                    }
                )
            else:
                assert job.pending_hookpoint == hookpoint, (
                    f"pending_hookpoint_mismatch:{job.pending_hookpoint}:{hookpoint}"
                )

            hook_jobs.append(job)

        self._wait_for_hook_jobs(hook_jobs)

        next_jobs: list[LiveJob] = []
        next_chunks: list[t.Tensor] = []
        cursor = 0
        for job in jobs:
            start = job.pass_start_idx
            end = start + job.seq_len
            module_slice = module_tensor[:, start:end, :]
            residual_slice = (
                None
                if residual_tensor is None
                else residual_tensor[:, start:end, :]
            )

            if hookpoint not in job.request.hooks:
                combined = _compose_gpt2_residual(
                    hook_kind, module_slice, residual_slice
                )
                next_chunks.append(combined)
                job.pass_start_idx = cursor
                cursor += job.seq_len
                next_jobs.append(job)
                continue

            if (
                job.pending_hookpoint is not None
                and job.pending_hookpoint != hookpoint
            ):
                combined = _compose_gpt2_residual(
                    hook_kind, module_slice, residual_slice
                )
                next_chunks.append(combined)
                job.pass_start_idx = cursor
                cursor += job.seq_len
                next_jobs.append(job)
                continue

            if job.pending_response is None:
                continue

            # pending_response.tensor is [seq, d_model] after _poll_hook_jobs normalize
            hook_tensor = job.cached_module_tensor
            if job.pending_response.tensor is not None:
                hook_tensor = job.pending_response.tensor.to(
                    device=module_tensor.device,
                    dtype=module_tensor.dtype,
                )

            if hook_kind == "resid":
                # Back to [1, seq, d_model] for concat along sequence dim
                combined = hook_tensor.unsqueeze(0)
            else:
                combined = (
                    hook_tensor + job.cached_residual_tensor
                ).unsqueeze(0)

            next_chunks.append(combined)
            job.pass_start_idx = cursor
            cursor += job.seq_len
            next_jobs.append(job)

            job.pending_hookpoint = None
            job.pending_response = None
            job.cached_module_tensor = None
            job.cached_residual_tensor = None

        if not next_chunks:
            raise RuntimeError("all_jobs_paused")

        self._pass_jobs = next_jobs
        return t.cat(next_chunks, dim=1)

    def _wait_for_hook_jobs(self, jobs: list[LiveJob]) -> None:
        if not jobs:
            return

        if not self._poll_hook_jobs(jobs):
            deadline = time.monotonic() + self._hook_wait_s
            while time.monotonic() < deadline:
                if self._poll_hook_jobs(jobs):
                    break
                time.sleep(0.001)

            if not any(job.pending_response is not None for job in jobs):
                while not self._poll_hook_jobs(jobs):
                    time.sleep(0.001)

        grace_deadline = time.monotonic() + self._hook_wait_s
        while time.monotonic() < grace_deadline:
            self._poll_hook_jobs(jobs)
            if all(job.pending_response is not None for job in jobs):
                break
            time.sleep(0.001)

    def _poll_hook_jobs(self, jobs: list[LiveJob]) -> bool:
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

    def _finish_live_jobs(self, jobs: list[LiveJob], status: str) -> None:
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
                    message.tensor = _normalize_worker_hook_tensor(message.tensor)
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

    def _model_device(self) -> t.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return t.device("cpu")

    def run_request(self, request: Request) -> str:
        self._validate_request_hooks(request)
        queued_request = QueuedRequest(request=request)

        with self._condition:
            if self._closed:
                return "wrapper_closed"
            self._pending_requests.append(queued_request)
            self._condition.notify_all()

        queued_request.done.wait()
        return queued_request.status

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            pending_requests = list(self._pending_requests)
            self._pending_requests.clear()
            active_requests = [
                job.queued_request for job in self._live_jobs
            ]
            self._condition.notify_all()

        for request in pending_requests:
            request.status = "wrapper_closed"
            request.done.set()

        for request in active_requests:
            request.status = "wrapper_closed"
            request.done.set()

        self._scheduler.join(timeout=5.0)
        self._shutdown.set()

        for worker in self._workers:
            try:
                worker.pipe.send({"type": "shutdown"})
            except (BrokenPipeError, EOFError, OSError):
                pass

            worker.process.join(timeout=1.0)
            if worker.process.is_alive():
                worker.process.terminate()
                worker.process.join(timeout=1.0)

            try:
                worker.pipe.close()
            except OSError:
                pass


__all__ = ["WrapperModel"]
