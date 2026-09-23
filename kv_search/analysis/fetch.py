"""Edge-fetch simulators over recorded retrieval. V-side (returned top-k values) unless
noted; K (HNSW traversal) is measured separately via scripts/measure_k.py."""

import numpy as np
import torch

from kv_search.analysis.data import CachedData
from kv_search.analysis._util import kmeans, merge_topk, retrieved_weights


def download_curve(d: CachedData, prompt_idx: int = 0) -> tuple[np.ndarray, np.ndarray, int]:
    """Cumulative distinct retrieved positions over generation, summed over (layer,
    kv-head) shards (distinctness is per kv-head; the 4 GQA heads share a shard)."""
    layers = d.full_layer_indices
    g = d.num_key_value_groups
    cum: np.ndarray | None = None
    for layer_idx in layers:
        _, idx_t = d.indices(layer_idx, prompt_idx)
        idx = idx_t.cpu().numpy()  # [n_qheads, q_len, n]
        q_len = idx.shape[1]
        if cum is None:
            cum = np.zeros(q_len, dtype=np.int64)
        for h0 in range(0, idx.shape[0], g):
            seen: set[int] = set()
            for t in range(q_len):
                for h in range(g):
                    seen.update(idx[h0 + h, t].tolist())
                cum[t] += len(seen)
    assert cum is not None
    n_shards = len(layers) * (idx.shape[0] // g)
    return np.arange(cum.shape[0]), cum, n_shards


def weight_mass_new(d: CachedData, prompt_idx: int = 0) -> dict[int, np.ndarray]:
    """Per layer/step: fraction of a query head's softmax weight on positions the shard
    fetches for the first time that step, averaged over heads. Small => fresh fetches
    barely affect the output."""
    g = d.num_key_value_groups
    out: dict[int, np.ndarray] = {}
    for layer_idx in d.full_layer_indices:
        idx, sc, dyn = d.aligned_records(layer_idx, prompt_idx)
        n_heads, q_len, k = idx.shape
        frac = np.zeros(q_len)
        for h0 in range(0, n_heads, g):
            seen: set[int] = set()
            for t in range(q_len):
                group_pos: set[int] = set()
                for j in range(g):
                    group_pos.update(idx[h0 + j, t].tolist())
                new_pos = group_pos - seen
                step = 0.0
                for j in range(g):
                    w = retrieved_weights(sc[h0 + j, t], dyn[h0 + j, t], k)
                    mask = np.fromiter((p in new_pos for p in idx[h0 + j, t]), bool, k)
                    step += w[mask].sum()
                frac[t] += step / g
                seen |= group_pos
        out[layer_idx] = frac / (n_heads // g)
    return out


def weight_threshold_tradeoff(
    d: CachedData, prompt_idx: int = 0, taus: list[float] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Policy "fetch a newly-seen position only if its per-shard weight >= tau". Returns
    (taus, download_frac, weight_retained), download_frac relative to tau=0 (fetch all)."""
    if taus is None:
        taus = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    taus_a = np.asarray(taus, dtype=np.float64)
    g = d.num_key_value_groups

    total_ret_w = 0.0
    lost = np.zeros(len(taus_a))
    fetched = np.zeros(len(taus_a))
    for layer_idx in d.full_layer_indices:
        idx, sc, dyn = d.aligned_records(layer_idx, prompt_idx)
        n_heads, q_len, k = idx.shape
        for h0 in range(0, n_heads, g):
            cached: list[set[int]] = [set() for _ in taus_a]
            for t in range(q_len):
                head_w = []
                for j in range(g):
                    w = retrieved_weights(sc[h0 + j, t], dyn[h0 + j, t], k)
                    head_w.append(w)
                    total_ret_w += float(w.sum())
                best: dict[int, float] = {}
                for j in range(g):
                    for p, w in zip(idx[h0 + j, t].tolist(), head_w[j]):
                        if w > best.get(p, 0.0):
                            best[p] = float(w)
                for i, tau in enumerate(taus_a):
                    c = cached[i]
                    for p, bw in best.items():
                        if bw >= tau:
                            c.add(p)
                    for j in range(g):
                        for p, w in zip(idx[h0 + j, t].tolist(), head_w[j]):
                            if p not in c:
                                lost[i] += w
            for i in range(len(taus_a)):
                fetched[i] += len(cached[i])

    return taus_a, fetched / fetched[0], 1.0 - lost / total_ret_w


def download_mse_tradeoff(
    d: CachedData, prompt_idx: int = 0, taus: list[float] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Output-MSE quality axis for the weight-threshold policy (pair with
    weight_threshold_tradeoff for the matching download). Needs prefill values."""
    if taus is None:
        taus = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    taus_a = np.asarray(taus, dtype=np.float64)
    g = d.num_key_value_groups

    sse = np.zeros(len(taus_a))
    count = 0
    for layer_idx in d.full_layer_indices:
        idx, sc, dyn = d.aligned_records(layer_idx, prompt_idx)
        n_heads, q_len, k = idx.shape
        gathered = d.retrieved_values(layer_idx, idx).to(torch.float32)
        sc_t = torch.as_tensor(sc, device=gathered.device, dtype=torch.float32)
        o_full = torch.einsum("htk,htkd->htd", torch.softmax(sc_t, dim=-1), gathered)

        # per shard/step, each position's best (retrieved+live) weight, tau-independent
        shard_best: list[list[dict[int, float]]] = []
        for h0 in range(0, n_heads, g):
            steps: list[dict[int, float]] = []
            for t in range(q_len):
                best: dict[int, float] = {}
                for j in range(g):
                    w = retrieved_weights(sc[h0 + j, t], dyn[h0 + j, t], k)
                    for p, wv in zip(idx[h0 + j, t].tolist(), w):
                        if wv > best.get(p, 0.0):
                            best[p] = float(wv)
                steps.append(best)
            shard_best.append(steps)

        for i, tau in enumerate(taus_a):
            mask = np.zeros((n_heads, q_len, k), dtype=bool)
            for si, h0 in enumerate(range(0, n_heads, g)):
                cache: set[int] = set()
                for t in range(q_len):
                    for p, bw in shard_best[si][t].items():
                        if bw >= tau:
                            cache.add(p)
                    for j in range(g):
                        mask[h0 + j, t] = np.fromiter(
                            (p in cache for p in idx[h0 + j, t]), bool, k
                        )
            mask_t = torch.as_tensor(mask, device=gathered.device)
            w = torch.softmax(sc_t.masked_fill(~mask_t, float("-inf")), dim=-1)
            o = torch.einsum("htk,htkd->htd", torch.nan_to_num(w), gathered)
            sse[i] += float(((o_full - o) ** 2).sum().item())
        count += n_heads * q_len * gathered.shape[-1]

    return taus_a, sse / count


def tail_meanfield_mse(
    d: CachedData, prompt_idx: int = 0, head_sizes: list[int] | None = None
) -> dict[str, list[float | int]]:
    """MSE vs full top-k when the low-weight tail is dropped (head_renorm) or replaced by
    its mean: in-step (meanfield_inst), causal running (meanfield_causal), or windowed
    (meanfield_window). head_renorm >> meanfield => only the tail's mean matters, not its
    identity. Needs prefill values."""
    if head_sizes is None:
        head_sizes = [1, 2, 4, 8, 16, 32]
    g = d.num_key_value_groups
    window = 32
    names = ("head_renorm", "meanfield_inst", "meanfield_causal", "meanfield_window")
    acc = {n: np.zeros(len(head_sizes)) for n in names}
    count = 0
    for layer_idx in d.full_layer_indices:
        idx, sc, _ = d.aligned_records(layer_idx, prompt_idx)
        n_heads, q_len, k = idx.shape
        gathered = d.retrieved_values(layer_idx, idx).to(torch.float32)
        w = torch.softmax(
            torch.as_tensor(sc, device=gathered.device, dtype=torch.float32), dim=-1
        )  # [H, q_len, K]
        o_full = torch.einsum("htk,htkd->htd", w, gathered)
        all_val_sum = gathered.sum(dim=2)
        n_shards = n_heads // g
        for hi, hs in enumerate(head_sizes):
            _, topi = torch.topk(w, hs, dim=-1)
            head_mask = torch.zeros_like(w, dtype=torch.bool).scatter_(-1, topi, True)
            w_head = (w * head_mask).sum(-1)
            w_tail = (1.0 - w_head).unsqueeze(-1)
            o_head_w = torch.einsum("htk,htkd->htd", w * head_mask, gathered)
            head_val_sum = torch.einsum("htk,htkd->htd", head_mask.float(), gathered)
            tail_val_sum = all_val_sum - head_val_sum
            tail_cnt = float(k - hs)

            o_hr = o_head_w / w_head.unsqueeze(-1).clamp_min(1e-9)
            acc["head_renorm"][hi] += float(((o_full - o_hr) ** 2).sum())

            o_mfi = o_head_w + w_tail * (tail_val_sum / tail_cnt)
            acc["meanfield_inst"][hi] += float(((o_full - o_mfi) ** 2).sum())

            tvs = tail_val_sum.reshape(n_shards, g, q_len, -1).sum(1)
            run_sum = torch.cumsum(tvs, dim=1)
            run_cnt = (
                torch.arange(1, q_len + 1, device=gathered.device) * (g * tail_cnt)
            ).view(1, q_len, 1)
            run_mean = (run_sum / run_cnt).repeat_interleave(g, dim=0)
            o_mfc = o_head_w + w_tail * run_mean
            acc["meanfield_causal"][hi] += float(((o_full - o_mfc) ** 2).sum())

            shifted = torch.zeros_like(run_sum)
            shifted[:, window:] = run_sum[:, :-window]
            win_sum = run_sum - shifted
            win_steps = torch.arange(1, q_len + 1, device=gathered.device).clamp_max(window)
            win_cnt = (win_steps * (g * tail_cnt)).view(1, q_len, 1)
            win_mean = (win_sum / win_cnt).repeat_interleave(g, dim=0)
            o_mfw = o_head_w + w_tail * win_mean
            acc["meanfield_window"][hi] += float(((o_full - o_mfw) ** 2).sum())
        count += n_heads * q_len * gathered.shape[-1]
    return {n: (v / count).tolist() for n, v in acc.items()} | {"head_sizes": head_sizes}


def centroid_substitution(
    d: CachedData,
    n_positions: int = 128,
    n_centroids: list[int] | None = None,
    head_sizes: list[int] | None = None,
    nprobes: list[int] | None = None,
    k: int = 128,
    fit_n: int = 20000,
    kmeans_iters: int = 15,
    ctx_chunk: int = 65536,
    layers: list[int] | None = None,
    seed: int = 0,
) -> dict:
    """Coarse-to-fine centroid sim, keys included. Self-contained on prefill q/k/v (uses
    the last n_positions prefill queries, not decode queries). Per (layer, kv-head):
    k-means the keys into C centroids (cluster-mean values). recall_weight[C][nprobe] =
    top-k attention weight whose cluster ranks in the top nprobe by q.centroid (the
    scoring/K side). mse_centroid[C][h] keeps the true top-h exact and replaces the tail
    by centroids (weight from centroid key, value from centroid), vs mse_drop / mse_mean
    (in-step mean) baselines. Needs prefill."""
    assert d.prefill is not None, "needs load_prefill=True"
    if n_centroids is None:
        n_centroids = [256, 1024, 4096]
    if head_sizes is None:
        head_sizes = [8, 16, 32, 64]
    if nprobes is None:
        nprobes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    g = d.num_key_value_groups
    dev = d.device
    layer_ids = layers if layers is not None else d.full_layer_indices

    sse = {c: {h: 0.0 for h in head_sizes} for c in n_centroids}
    rec = {c: {npr: 0.0 for npr in nprobes} for c in n_centroids}
    sse_drop = {h: 0.0 for h in head_sizes}
    sse_mean = {h: 0.0 for h in head_sizes}
    n_q = 0
    dim = 0

    for layer_idx in layer_ids:
        keys, values = d.prefill_kv(layer_idx)
        n_kv, ctx, dim = keys.shape
        q_all = d.queries(layer_idx, slice(-n_positions, None)).to(dev, torch.float32)[0]
        for s in range(n_kv):
            ks, vs = keys[s], values[s]
            # exact top-k per query head (C-independent) + drop/mean baselines
            exact: list[tuple[torch.Tensor, ...]] = []
            for j in range(g):
                q = q_all[s * g + j]
                P = q.shape[0]
                e_v = torch.full((P, k), float("-inf"), device=dev)
                e_i = torch.zeros((P, k), dtype=torch.long, device=dev)
                for b in range(0, ctx, ctx_chunk):
                    e = b + min(ctx_chunk, ctx - b)
                    lb = (q @ ks[b:e].T) * d.scaling
                    ids = torch.arange(b, e, device=dev).expand(P, e - b)
                    e_v, e_i = merge_topk(e_v, e_i, lb, ids, k)
                val_top = vs[e_i]
                w_full = torch.softmax(e_v, dim=-1)
                o_full = torch.einsum("pk,pkd->pd", w_full, val_top)
                for h in head_sizes:
                    o_drop = torch.einsum(
                        "pk,pkd->pd", torch.softmax(e_v[:, :h], dim=-1), val_top[:, :h]
                    )
                    sse_drop[h] += float(((o_full - o_drop) ** 2).sum())
                    w_head = w_full[:, :h].sum(-1, keepdim=True)
                    o_head = torch.einsum("pk,pkd->pd", w_full[:, :h], val_top[:, :h])
                    o_mean = o_head + (1.0 - w_head) * val_top[:, h:].mean(1)
                    sse_mean[h] += float(((o_full - o_mean) ** 2).sum())
                exact.append((q, e_v, e_i, val_top, o_full))
                n_q += P
            for c in n_centroids:
                cent_k, assign = kmeans(
                    ks, c, iters=kmeans_iters, fit_n=fit_n, seed=seed, chunk=ctx_chunk
                )
                cent_v = torch.zeros(c, dim, device=dev)
                cnt = torch.zeros(c, device=dev)
                cent_v.index_add_(0, assign, vs)
                cnt.index_add_(0, assign, torch.ones(ctx, device=dev))
                cent_v /= cnt.clamp_min(1.0).unsqueeze(1)
                for q, e_v, e_i, val_top, o_full in exact:
                    qc = (q @ cent_k.T) * d.scaling
                    w_full = torch.softmax(e_v, dim=-1)
                    cl = assign[e_i]
                    cent_logit = torch.einsum("pd,pkd->pk", q, cent_k[cl]) * d.scaling
                    cent_val = cent_v[cl]
                    for h in head_sizes:
                        lg = e_v.clone()
                        lg[:, h:] = cent_logit[:, h:]
                        vl = val_top.clone()
                        vl[:, h:] = cent_val[:, h:]
                        o = torch.einsum("pk,pkd->pd", torch.softmax(lg, dim=-1), vl)
                        sse[c][h] += float(((o_full - o) ** 2).sum())
                    cl_rank = qc.argsort(-1, descending=True).argsort(-1)
                    top_cl_rank = cl_rank.gather(1, cl)
                    for npr in nprobes:
                        rec[c][npr] += float((w_full * (top_cl_rank < npr)).sum())
                del cent_k, assign, cent_v

    n_el = n_q * dim
    return {
        "n_centroids": n_centroids,
        "head_sizes": head_sizes,
        "nprobes": nprobes,
        "k": k,
        "n_queries": n_q,
        "recall_weight": {c: {npr: rec[c][npr] / n_q for npr in nprobes} for c in n_centroids},
        "mse_centroid": {c: {h: sse[c][h] / n_el for h in head_sizes} for c in n_centroids},
        "mse_drop": {h: sse_drop[h] / n_el for h in head_sizes},
        "mse_mean": {h: sse_mean[h] / n_el for h in head_sizes},
    }


def cached_reuse_substitution(
    d: CachedData, prompt_idx: int = 0, head_sizes: list[int] | None = None
) -> dict:
    """Reuse-the-cached-set sim (V-side, no HNSW). Fetch the top-h positions each step
    (they join the shard cache), reuse cached positions exactly, and substitute uncached
    tail positions with the nearest cached value (by key). mse_cached is that policy;
    mse_drop / mse_mean (in-step mean) are baselines; download_frac vs fetch-all. Needs
    prefill."""
    if head_sizes is None:
        head_sizes = [8, 16, 32]
    g = d.num_key_value_groups
    dev = d.device
    names = ("mse_drop", "mse_mean", "mse_cached")
    sse = {n: {h: 0.0 for h in head_sizes} for n in names}
    fetched = {h: 0 for h in head_sizes}
    distinct_all = 0
    n_el = 0
    for layer_idx in d.full_layer_indices:
        idx, sc, _ = d.aligned_records(layer_idx, prompt_idx)
        keys_p, vals_p = d.prefill_kv(layer_idx)
        n_kv, _, dim = keys_p.shape
        _, q_len, K = idx.shape
        idx_t = torch.as_tensor(idx, device=dev, dtype=torch.long)
        w_t = torch.softmax(torch.as_tensor(sc, device=dev, dtype=torch.float32), dim=-1)
        for s in range(n_kv):
            hs = slice(s * g, (s + 1) * g)
            distinct_all += int(torch.unique(idx_t[hs].reshape(-1)).numel())
        for h in head_sizes:
            for s in range(n_kv):
                hs = slice(s * g, (s + 1) * g)
                is_cached = torch.zeros(keys_p.shape[1], dtype=torch.bool, device=dev)
                cache_k = torch.empty(0, dim, device=dev)
                cache_v = torch.empty(0, dim, device=dev)
                for t in range(q_len):
                    pos_g = idx_t[hs, t]  # [g, K]
                    w_g = w_t[hs, t]
                    toph = pos_g.gather(1, w_g.topk(min(h, K), dim=1).indices)
                    cand = torch.unique(toph)
                    newp = cand[~is_cached[cand]]
                    if newp.numel():
                        is_cached[newp] = True
                        cache_k = torch.cat([cache_k, keys_p[s, newp]])
                        cache_v = torch.cat([cache_v, vals_p[s, newp]])
                    val_g = vals_p[s, pos_g]
                    o_full = torch.einsum("gk,gkd->gd", w_g, val_g)
                    cmask = is_cached[pos_g]
                    unc = ~cmask
                    wd = w_g * cmask
                    o_drop = torch.einsum(
                        "gk,gkd->gd", wd / wd.sum(1, keepdim=True).clamp_min(1e-9), val_g
                    )
                    sse["mse_drop"][h] += float(((o_full - o_drop) ** 2).sum())
                    cnt = unc.sum(1, keepdim=True).clamp_min(1)
                    mean_g = (val_g * unc.unsqueeze(-1)).sum(1) / cnt
                    v_mean = torch.where(unc.unsqueeze(-1), mean_g.unsqueeze(1), val_g)
                    o_mean = torch.einsum("gk,gkd->gd", w_g, v_mean)
                    sse["mse_mean"][h] += float(((o_full - o_mean) ** 2).sum())
                    v_sub = val_g.clone()
                    if cache_k.shape[0] and bool(unc.any()):
                        nn = torch.cdist(keys_p[s, pos_g[unc]], cache_k).argmin(1)
                        v_sub[unc] = cache_v[nn]
                    o_sub = torch.einsum("gk,gkd->gd", w_g, v_sub)
                    sse["mse_cached"][h] += float(((o_full - o_sub) ** 2).sum())
                fetched[h] += int(is_cached.sum())
        n_el += idx_t.shape[0] * q_len * dim
    return {
        "head_sizes": head_sizes,
        "download_frac": {h: fetched[h] / distinct_all for h in head_sizes},
        **{n: {h: sse[n][h] / n_el for h in head_sizes} for n in names},
    }
