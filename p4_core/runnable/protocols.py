from typing import Protocol


class BaseRunnable[T, U, Context](Protocol):
    def run_or_none(
        self,
        x: T,
        context: Context,
    ) -> U | None:
        try:
            return self.run(x, context)
        except Exception:
            return None

    def run(
        self,
        x: T,
        context: Context,
    ) -> U: ...
