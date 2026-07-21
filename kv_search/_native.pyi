import numpy as np

class NativeEdgeRetriever:
    def __init__(self, shards: list[tuple[tuple[int, int], str]]): ...
    def retrieve(
        self,
        layer_idx: int,
        head_idx: int,
        q: np.ndarray[tuple[int], np.dtype[np.float32]],
        limit: int,
    ) -> tuple[
        np.ndarray[tuple[int, int], np.dtype[np.float32]],
        np.ndarray[tuple[int, int], np.dtype[np.float32]],
    ]: ...
