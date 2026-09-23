import numpy as np
import torch


def merge_topk(
    best_v: torch.Tensor,
    best_i: torch.Tensor,
    new_v: torch.Tensor,
    new_i: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge a running top-k (values+indices [P, k]) with a new block, keep top k."""
    v = torch.cat([best_v, new_v], dim=1)
    i = torch.cat([best_i, new_i], dim=1)
    tv, ti = v.topk(k, dim=-1)
    return tv, i.gather(1, ti)


def kmeans(
    x: torch.Tensor,
    c: int,
    iters: int = 15,
    fit_n: int = 20000,
    seed: int = 0,
    chunk: int = 65536,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lloyd k-means; fits on a random subsample of fit_n rows, assigns all in chunks.
    Returns (centroids [c, d], assignment [N]); empty clusters keep their centroid."""
    n, d = x.shape
    gen = torch.Generator(device=x.device).manual_seed(seed)
    c = min(c, n)
    fit = x if n <= fit_n else x[torch.randperm(n, generator=gen, device=x.device)[:fit_n]]
    cent = fit[torch.randperm(fit.shape[0], generator=gen, device=x.device)[:c]].clone()

    def assign_all(src: torch.Tensor) -> torch.Tensor:
        out = torch.empty(src.shape[0], dtype=torch.long, device=x.device)
        for s in range(0, src.shape[0], chunk):
            out[s : s + chunk] = torch.cdist(src[s : s + chunk], cent).argmin(1)
        return out

    for _ in range(iters):
        a = assign_all(fit)
        new = torch.zeros_like(cent)
        cnt = torch.zeros(c, device=x.device)
        new.index_add_(0, a, fit)
        cnt.index_add_(0, a, torch.ones(fit.shape[0], device=x.device))
        nonempty = cnt > 0
        cent[nonempty] = new[nonempty] / cnt[nonempty].unsqueeze(1)
    return cent, assign_all(x)


def retrieved_weights(sc_ht: np.ndarray, dyn_ht: np.ndarray, k: int) -> np.ndarray:
    """Softmax over one step's retrieved top-k logits + live logits; return top-k slice."""
    logits = np.concatenate([sc_ht, dyn_ht[~np.isnan(dyn_ht)]])
    w = np.exp(logits - logits.max())
    w /= w.sum()
    return w[:k]
