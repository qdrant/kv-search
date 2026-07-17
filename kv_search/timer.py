import time
from collections.abc import Callable
from pydantic import BaseModel, Field


class Timer(BaseModel):
    name: str
    total: float = 0
    start: float | None = None
    timings: list[float] = Field(default_factory=list)

    def record(self):
        if self.start is not None:
            elapsed = time.perf_counter() - self.start
            self.total += elapsed
            self.timings.append(elapsed)
        self.start = time.perf_counter()

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self.start is not None
        elapsed = time.perf_counter() - self.start
        self.total += elapsed
        self.timings.append(elapsed)

    def __call__[**P, T](self, func: Callable[P, T]) -> Callable[P, T]:
        def inner(*args: P.args, **kwargs: P.kwargs) -> T:
            with self:
                ret = func(*args, **kwargs)
            return ret

        return inner

    def __repr__(self) -> str:
        return f'{{"total": {self.total:.2f}, "average": {sum(self.timings)/len(self.timings):.2f}}}'


class Timers(BaseModel):
    model_load: Timer = Timer(name="model_load")
    prefill_gen: Timer = Timer(name="prefill_gen")
    prefill_save: Timer = Timer(name="prefill_save")
    prefill_load: Timer = Timer(name="prefill_load")

    token_gen: Timer = Timer(name="token_gen")

    qdrant_retrieve: Timer = Timer(name="qdrant_retrieve")
    qdrant_assemble: Timer = Timer(name="qdrant_assemble")


timers = Timers()
