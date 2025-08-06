from pathlib import Path
from typing import Protocol, TypedDict


class Scope(TypedDict):
    source_directory: Path
    initial_crash_log: str
    global_executable: Path


class BaseSandbox(Protocol):
    @property
    def scope(self) -> Scope: ...

    def build(self) -> tuple[int, str, str]: ...
    def reproduce(self) -> tuple[int, str]: ...
