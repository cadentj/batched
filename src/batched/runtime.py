from __future__ import annotations

from collections import deque
from multiprocessing.connection import Connection
import threading
import time
from typing import Any

import torch as t
import torch.multiprocessing as mp

from .types import Job, QueuedRequest, Request, WorkerResponse, WorkerSlot
from .worker import _worker_main

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ModuleNotFoundError:
    AutoModelForCausalLM = Any
    AutoTokenizer = Any


class WrapperModel:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        n: int = 1,
        batch_window_ms: int = 25,
        worker_memory_fraction: float | None = None,
        cuda_mps_pipe_directory: str | None = None,
        cuda_mps_active_thread_percentage: int | None = None,
    ):
        assert n > 0, "n must be positive"
        assert batch_window_ms >= 0, "batch_window_ms must be non-negative"
        if worker_memory_fraction is not None and not (
            0.0 < worker_memory_fraction <= 1.0
        ):
            raise ValueError(
                "worker_memory_fraction must be in the range (0, 1]"
            )
        if cuda_mps_active_thread_percentage is not None and not (
            1 <= cuda_mps_active_thread_percentage <= 100
        ):
            raise ValueError(
                "cuda_mps_active_thread_percentage must be in the range [1, 100]"
            )

        self.model = model
        self.model.eval()
        self.batch_window_ms = batch_window_ms

        # CUDA tensor IPC between processes requires spawn-safe workers.
        self._mp_context = mp.get_context("spawn")
        self._shutdown = self._mp_context.Event()
        self._condition = threading.Condition()
        self._closed = False
        self._pending_requests: deque[QueuedRequest] = deque()
        self._active_batch: list[Job] = []
        self._tokenizer: AutoTokenizer = tokenizer
        self._last_batch_size = 0

        self._hook_handles = self._register_hooks()
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

    def _register_hooks(self) -> list[Any]:
        handles: list[Any] = []
        for hookpoint, module in self.model.named_modules():
            if not hookpoint:
                continue
            handles.append(
                module.register_forward_hook(self._make_module_hook(hookpoint))
            )
        return handles

    def _make_module_hook(self, hookpoint: str):
        def _hook(_module: Any, _args: tuple[Any, ...], output: Any) -> Any:
            batch = self._active_batch

            is_tuple = isinstance(output, tuple)
            # NOTE(cadentj): This assumes the first input is a tensor
            activation = output[0] if is_tuple else output

            activation_was_modified = False
            hook_jobs: list[tuple[Job, WorkerSlot]] = []
            for job in batch:
                worker = job.worker

                if hookpoint not in job.request.hooks:
                    continue

                activation_slice = activation[job.batch_index : job.batch_index + 1]

                worker.pipe.send(
                    {
                        "type": "apply_hooks",
                        "request_id": job.request.id,
                        "hookpoint": hookpoint,
                        "tensor": activation_slice,
                    }
                )
                hook_jobs.append((job, worker))

            for job, worker in hook_jobs:
                message = self._recv_request_message(worker)
                assert message.type == "hooks_applied", (
                    f"unexpected_worker_message:{message.type}"
                )

                if message.tensor is None:
                    continue

                message.tensor = message.tensor.to(
                    device=activation.device,
                    dtype=activation.dtype,
                )
                activation[job.batch_index : job.batch_index + 1] = message.tensor
                activation_was_modified = True

            if activation_was_modified:
                return output if not is_tuple else (activation, *output[1:])
            return None

        return _hook

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
            batch = self._dequeue_batch()
            if batch is None:
                return

            self._run_batch(batch)

    def _dequeue_batch(self) -> list[Job] | None:
        with self._condition:
            while True:
                while not self._closed and (
                    not self._pending_requests or not self._idle_workers
                ):
                    self._condition.wait(timeout=0.05)

                if self._closed and not self._pending_requests:
                    return None

                if not self._pending_requests or not self._idle_workers:
                    if self._closed:
                        return None
                    continue

                deadline = time.monotonic() + (self.batch_window_ms / 1000.0)
                target_batch_size = len(self._idle_workers)
                while (
                    not self._closed
                    and len(self._pending_requests) < target_batch_size
                ):
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
                    if self._closed:
                        return None
                    continue

                batch: list[Job] = []
                for batch_index in range(batch_size):
                    queued_request = self._pending_requests.popleft()
                    worker = self._idle_workers.popleft()
                    batch.append(
                        Job(
                            request=queued_request.request,
                            worker=worker,
                            batch_index=batch_index,
                            queued_request=queued_request,
                        )
                    )
                self._last_batch_size = len(batch)
                return batch

    def _run_batch(self, batch: list[Job]) -> None:
        self._active_batch = batch
        status = "finished"

        try:
            for job in batch:
                job.worker.pipe.send(
                    {
                        "type": "start_request",
                        "request_id": job.request.id,
                        "hooks": job.request.dump_hooks(),
                    }
                )
            for job in batch:
                message = self._recv_request_message(job.worker)
                if message.type != "request_started":
                    status = "failed"

            prompts = [job.request.prompt for job in batch]
            inputs = self._tokenizer(prompts, return_tensors="pt", padding=True)
            device = self._model_device()
            inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

            with t.inference_mode():
                self.model(**inputs, use_cache=False)

        except BaseException:
            status = "failed"
        finally:
            for job in batch:
                job.worker.pipe.send(
                    {
                        "type": "finish_request",
                        "request_id": job.request.id,
                    }
                )
            for job in batch:
                message = self._recv_request_message(job.worker)
                if message.type != "request_finished":
                    status = "failed"
            self._complete_batch(batch, status=status)

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
        queued_request = QueuedRequest(request=request)

        with self._condition:
            if self._closed:
                return "wrapper_closed"
            self._pending_requests.append(queued_request)
            self._condition.notify_all()

        queued_request.done.wait()
        return queued_request.status

    def _complete_batch(self, batch: list[Job], status: str) -> None:
        returned_workers: list[WorkerSlot] = []
        for job in batch:
            worker = job.worker
            returned_workers.append(worker)
            if job.queued_request.status == "queued":
                job.queued_request.status = status
            job.queued_request.done.set()

        with self._condition:
            self._active_batch = []
            for worker in returned_workers:
                self._idle_workers.append(worker)
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            pending_requests = list(self._pending_requests)
            self._pending_requests.clear()
            active_requests = [job.queued_request for job in self._active_batch]
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

        for handle in self._hook_handles:
            handle.remove()


__all__ = ["WrapperModel"]
