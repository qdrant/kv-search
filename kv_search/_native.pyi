import numpy as np

def tailm_kernel() -> str:
    """The tailM decode-tail kernel this CPU selects: "avx2" (x86_64 with AVX2 + FMA),
    "neon" (aarch64) or "portable"."""

class NativeEdgeRetriever:
    def __init__(
        self,
        shards: list[tuple[tuple[int, int], str]],
        exact: bool = True,
        hnsw_ef: int = 128,
    ):
        """Load the shards. `exact`: full-scan top-n; otherwise HNSW search with `hnsw_ef`
        (raised to the limit by qdrant-edge)."""
    @property
    def search_mode(self) -> str:
        """`"exact"` or `"hnsw ef N"`."""
    def retrieve(
        self,
        layer_idx: int,
        q: np.ndarray[tuple[int, int, int], np.dtype[np.float32]],
        limit: int,
        scaling: float,
    ) -> tuple[
        np.ndarray[tuple[int, int, int], np.dtype[np.float32]],
        np.ndarray[tuple[int, int], np.dtype[np.float32]],
        np.ndarray[tuple[int, int], np.dtype[np.float32]],
    ]: ...
    def retrieve_tail(
        self,
        layer_idx: int,
        q: np.ndarray[tuple[int, int, int], np.dtype[np.float32]],
        limit: int,
        scaling: float,
    ) -> tuple[
        np.ndarray[tuple[int, int, int], np.dtype[np.float32]],
        np.ndarray[tuple[int, int], np.dtype[np.float32]],
        np.ndarray[tuple[int, int], np.dtype[np.float32]],
        np.ndarray[tuple[int, int, int], np.dtype[np.float32]],
        np.ndarray[tuple[int, int], np.dtype[np.float32]],
    ]:
        """`retrieve`'s (out, lse, boundary), bit for bit, plus (tail_out, tail_lse): the tailM tail
        on the heads registered with `set_tailm`, computed in each KV head's search task; other
        rows repeat out / lse exactly."""
    def set_tailm(
        self,
        layer: int,
        heads: list[int],
        packed: np.ndarray[tuple[int, int, int], np.dtype[np.uint16]],
        bias: np.ndarray[tuple[int, int], np.dtype[np.float32]],
        scale: np.ndarray[tuple[int], np.dtype[np.float32]],
        log_alpha: np.ndarray[tuple[int], np.dtype[np.float32]],
        n_keys: int,
    ) -> None:
        """Register `layer`'s tail heads (replacing the layer's earlier ones): `packed` [h, d, 2d+1]
        bf16 bits, `bias` [h, d], `scale` / `log_alpha` [h] (−inf for α = 0). ValueError on bad shapes."""
    def clear_tailm(self) -> None: ...
    def set_tailm_kernel(self, name: str) -> None:
        """Use the decode-tail kernel `name`: "avx2" / "neon" (if this CPU has it) or
        "portable"; for tests and benchmarks. ValueError otherwise."""
