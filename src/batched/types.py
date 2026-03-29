from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
import re
import threading
from typing import Any, Literal
from uuid import uuid4

import torch.multiprocessing as mp
from pydantic import BaseModel, Field, model_validator, field_validator

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

    def dump_hooks(self) -> dict[str, dict[str, Any]]:
        return {
            hookpoint: hook.model_dump(mode="python")
            for hookpoint, hook in self.hooks.items()
        }

    @field_validator("prompt", mode="after")
    @classmethod
    def has_prompt(cls, v: str) -> str:
        if len(v) == 0:
            raise ValueError("empty_prompt")
        return v

    


class WorkerRequest(BaseModel):
    type: Literal["start_request", "apply_hooks", "finish_request", "shutdown"]
    request_id: str = ""
    hooks: dict[str, Hook] = Field(default_factory=dict)
    hookpoint: str = ""
    tensor: Any | None = None


class WorkerResponse(BaseModel):
    type: Literal["request_started", "hooks_applied", "request_finished"]
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


@dataclass
class QueuedRequest:
    request: Request
    status: str = "queued"
    done: threading.Event = field(default_factory=threading.Event)


__all__ = [
    "CompiledHook",
    "Hook",
    "QueuedRequest",
    "Request",
    "WorkerRequest",
    "WorkerRequestRuntime",
    "WorkerResponse",
    "WorkerSlot",
]
