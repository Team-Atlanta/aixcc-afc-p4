from typing import Protocol

from .models import Fragment


class BasePattern(Protocol):
    def match(self, source: str) -> set[Fragment]: ...
