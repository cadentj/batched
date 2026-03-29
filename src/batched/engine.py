from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from abc import abstractmethod
import time
import threading
import torch as t
import torch.multiprocessing as mp
from .types import QueuedRequest, Request, WorkerResponse, WorkerSlot

from .worker import _worker_main
from .utils import warn_if_mps_daemon_inactive

@dataclass
class Job:
    worker: WorkerSlot
    queued_request: QueuedRequest
    token_ids: list[int]
    seq_len: int
    pass_start_idx: int = 0
    pending_hookpoint: str | None = None
    pending_response: WorkerResponse | None = None
    cached_module_tensor: t.Tensor | None = None
    cached_residual_tensor: t.Tensor | None = None

    start_idx: int = 0
    is_paused: bool = False

    @property
    def request(self) -> Request:
        return self.queued_request.request

@dataclass
class Batch: 
    jobs: list[Job]

    @property
    def is_empty(self) -> bool:
        return len(self.jobs) > 0

    def extend(self, jobs: list[Job]) -> None:
        self.jobs.extend(jobs)

class Engine: 
    def __init__(
        self,
        n: int = 1,
        batch_window_ms: int = 25,
        worker_memory_fraction: float | None = None,
        cuda_mps_pipe_directory: str | None = None,
        cuda_mps_active_thread_percentage: int | None = None,
    ):
        assert n > 0, "n must be positive"
        assert batch_window_ms >= 0, "batch_window_ms must be non-negative"
        if worker_memory_fraction is not None:
            assert 0.0 < worker_memory_fraction <= 1.0, (
                "worker_memory_fraction must be in the range (0, 1]"
            )
        if cuda_mps_active_thread_percentage is not None:
            assert 1 <= cuda_mps_active_thread_percentage <= 100, (
                "cuda_mps_active_thread_percentage must be in the range [1, 100]"
            )

        warn_if_mps_daemon_inactive(cuda_mps_pipe_directory)

        self._pending_requests: deque[QueuedRequest] = deque()
        self._idle_workers: deque[WorkerSlot] = deque()

        # CUDA tensor IPC between processes requires spawn-safe workers.
        self._mp_context = mp.get_context("spawn")
        self._shutdown = self._mp_context.Event()
        self._condition = threading.Condition()
        self._closed = False
        self._pending_requests: deque[QueuedRequest] = deque()
        # self._live_jobs: list[Job] = []
        # self._pass_jobs: list[Job] = []
        self._batch: Batch = Batch()    
        self.batch_window_ms = batch_window_ms

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

    def _scheduler_loop(self) -> None:
        while True:
            batch = self._next_cycle_jobs()

            # Scheduler is closed and there are no live jobs.
            if batch is None:
                return

            self._run_forward_cycle(batch)

    @abstractmethod
    def _run_forward_cycle(self, batch: Batch) -> None:
        pass

    def _next_cycle_jobs(self) -> Batch | None:
        while True:
            assignments: list[tuple[QueuedRequest, WorkerSlot]] = []
            # 1) Wait for new requests to arrive.
            with self._condition:
                while True:
                    assignments = self._wait_for_request_batch()
                    if assignments:
                        break

                    if self._closed and self._batch.is_empty:
                        return None

                    # If we didn't get any new requests in the window,
                    # and there are cached jobs, just run those.
                    if not self._batch.is_empty:
                        return self._batch

                    # Else wait for more requests to arrive.
                    self._condition.wait(timeout=0.05)

            # NOTE(cadentj): why not start jobs within the condition?

            # 2) Start new jobs.
            new_jobs: list[Job] = []
            for queued_request, worker in assignments:
                new_jobs.append(
                    self._start_live_job(
                        queued_request=queued_request,
                        worker=worker,
                    )
                )

            # 3) Add new jobs to the batch.
            with self._condition:
                self._batch.extend(new_jobs)
                if self._closed and self._batch.is_empty:
                    return None
                if not self._batch.is_empty:
                    return self._batch

    def _start_job(
        self,
        queued_request: QueuedRequest,
        worker: WorkerSlot,
    ) -> Job:
        request = queued_request.request
        token_ids = self._tokenizer(
            request.prompt,
            add_special_tokens=False,
        )["input_ids"]
        
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

        return Job(
            worker=worker,
            queued_request=queued_request,
            token_ids=token_ids,
            seq_len=len(token_ids),
        )

    def _wait_for_request_batch(
        self,
    ) -> list[tuple[QueuedRequest, WorkerSlot]]:
        """
        Wait max batch_window_ms for more requests to arrive.

        Returns:
            list[tuple[QueuedRequest, WorkerSlot]]: A list of queued requests and idle workers.
        """
        if self._closed:
            return []
        if not self._pending_requests or not self._idle_workers:
            return []

        # Wait some window of time for more requests to arrive
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

            # Wait for the worker to be ready.
            if not parent_pipe.poll(10.0):
                raise RuntimeError(f"worker_start_timeout:{worker_id}")
            message = parent_pipe.recv()
            if (
                message.get("type") != "worker_ready"
                or message.get("worker_id") != worker_id
            ):
                raise RuntimeError(f"unexpected_worker_ready_message:{message}")


            workers.append(
                WorkerSlot(
                    worker_id=worker_id,
                    pipe=parent_pipe,
                    process=process,
                )
            )
        return workers

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