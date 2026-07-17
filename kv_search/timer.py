import time
from pydantic import BaseModel, Field


class Timer(BaseModel):
    name: str
    total: float = 0
    start: float = 0
    timings: list[float] = Field(default_factory=list)

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start
        self.total += elapsed
        self.timings.append(elapsed)


class Timers(BaseModel):
    model_load: Timer = Timer(name="model_load")
    prefill_gen: Timer = Timer(name="prefill_gen")
    prefill_load: Timer = Timer(name="prefill_load")

    token_gen_total: Timer = Timer(name="token_gen_total")

    qdrant_retrieve: Timer = Timer(name="qdrant_retrieve")
    qdrant_assemble: Timer = Timer(name="qdrant_assemble")

timers = Timers()
