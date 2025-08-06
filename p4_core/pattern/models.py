from __future__ import annotations

from pydantic import BaseModel


class Fragment(BaseModel):
    value: str
    start_position: int

    @property
    def end_position(self):
        return self.start_position + len(self.value)

    def __hash__(self) -> int:
        return hash(f"{self.start_position}:{self.value}")
