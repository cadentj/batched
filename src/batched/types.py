from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.connection import Connection
import re
import threading
from typing import Any, Literal
from uuid import uuid4

import torch.multiprocessing as mp
from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    model_validator,
    field_validator,
)

_HOOKPOINT_RE = re.compile(r"^transformer\.h\.(\d+)\.(attn|mlp|resid)$")


class Hook(BaseModel):
    fn: str
    hookpoint: str
    op: Literal["read", "write"]

    @model_validator(mode="after")
    def validate_hookpoint(self) -> Hook:
        match = _HOOKPOINT_RE.match(self.hookpoint)
        assert match is not None, f"unsupported_hookpoint:{self.hookpoint}"
        return self


class Request(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    input_ids: list[int]
    hooks: dict[str, Hook] = Field(default_factory=dict)
    _done: threading.Event = PrivateAttr(default_factory=threading.Event)
    _status: str = PrivateAttr(default="queued")

    def dump_hooks(self) -> dict[str, dict[str, Any]]:
        return {
            hookpoint: hook.model_dump(mode="python")
            for hookpoint, hook in self.hooks.items()
        }

    @field_validator("input_ids", mode="after")
    @classmethod
    def has_input_ids(cls, v: list[int]) -> list[int]:
        if len(v) == 0:
            raise ValueError("empty_input_ids")
        return v

    @property
    def done(self) -> threading.Event:
        return self._done

    @property
    def status(self) -> str:
        return self._status

    @status.setter
    def status(self, value: str) -> None:
        self._status = value


class StartRequest(BaseModel):
    type: Literal["start_request"]
    hooks: dict[str, Hook]


class HookRequest(BaseModel):
    type: Literal["apply_hooks"]
    hookpoint: str
    tensor: Any


class ShutdownRequest(BaseModel):
    type: Literal["shutdown"]


class WorkerRequest(BaseModel):
    req: StartRequest | HookRequest | ShutdownRequest
    request_id: str


class WorkerResponse(BaseModel):
    type: Literal["request_started", "hooks_applied"]
    request_id: str
    # t.Tensor is not serializable, just use Any
    tensor: Any | None = None


@dataclass
class CompiledHook:
    hookpoint: str
    op: str
    fn: Callable[[Any, Any, Any], Any]


@dataclass
class WorkerRequestRuntime:
    hooks: dict[str, CompiledHook]
    scope: dict[str, Any]


@dataclass
class WorkerSlot:
    worker_id: int
    pipe: Connection
    process: mp.Process


__all__ = [
    "CompiledHook",
    "Hook",
    "Request",
    "WorkerRequest",
    "WorkerRequestRuntime",
    "WorkerResponse",
    "WorkerSlot",
]
