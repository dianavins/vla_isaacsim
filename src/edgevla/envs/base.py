from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EnvAdapter(Protocol):
    suites: list[str]

    def reset(self, suite: str | None = None) -> object: ...

    def step(self, action) -> tuple[object, float, bool, dict]: ...

    def is_success(self) -> bool: ...
