import time
from collections.abc import Callable
from dataclasses import dataclass, field, fields

from rich.table import Table


@dataclass
class Timer:
    _start: float | None = field(default=None, repr=False, compare=False)
    timings: list[float] = field(default_factory=list)

    def record(self):
        if self._start is not None:
            elapsed = time.perf_counter() - self._start
            self.timings.append(elapsed)
        self._start = time.perf_counter()

    def reset_lap(self):
        self._start = None

    def reset(self):
        self._start = None
        self.timings.clear()

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self._start is not None
        elapsed = time.perf_counter() - self._start
        self.timings.append(elapsed)

    def __call__[**P, T](self, func: Callable[P, T]) -> Callable[P, T]:
        def inner(*args: P.args, **kwargs: P.kwargs) -> T:
            with self:
                ret = func(*args, **kwargs)
            return ret

        return inner

    @property
    def count(self) -> int:
        return len(self.timings)

    @property
    def total(self) -> float:
        return sum(self.timings)

    @property
    def mean(self) -> float:
        return self.total / self.count if self.timings else 0

    def __repr__(self) -> str:
        return f'{{"total": {self.total:.2f}, "average": {sum(self.timings) / len(self.timings):.2f}}}'


@dataclass
class Timers:
    model_load: Timer = field(default_factory=Timer)
    prefill_gen: Timer = field(default_factory=Timer)
    prefill_save: Timer = field(default_factory=Timer)
    prefill_load: Timer = field(default_factory=Timer)

    token_gen: Timer = field(default_factory=Timer)

    qdrant_retrieve: Timer = field(default_factory=Timer)
    qdrant_assemble: Timer = field(default_factory=Timer)

    # timers that accumulate per generation and should be cleared between prompts
    GENERATION = ("token_gen", "qdrant_retrieve", "qdrant_assemble")

    def reset_generation(self) -> None:
        for name in self.GENERATION:
            getattr(self, name).reset()

    def __rich__(self) -> Table:
        table = Table(title="timings", title_justify="left")
        table.add_column("timer")
        table.add_column("total (s)", justify="right")
        table.add_column("mean (ms)", justify="right")
        for f in fields(self):
            t: Timer = getattr(self, f.name)
            if t.count == 0:
                continue
            table.add_row(f.name, f"{t.total:.2f}", f"{t.mean * 1e3:.1f}")
        return table


timers = Timers()
