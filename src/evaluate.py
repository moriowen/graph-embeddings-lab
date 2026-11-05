"""Metrics, both quality definitions (PLAN section 3.1), defined before results.

Held-out links: P/R/NDCG/hit@K over nodes with >= 1 held-out edge; candidates exclude the query and known edges.
Label agreement: fraction of top-K sharing any label with the query, compared with that query's base rate.
"""
import numpy as np
import scipy.sparse as sp


def link_metrics(top, queries, rel, ks):
    """top: (len(queries), Kmax) ranked ids, -1 padded. rel: sparse n x n of held-out edges.
    Returns {k: {metric: per-query array}} and each query's held-out count."""
    R = rel[queries].tocsr()
    n_rel = np.asarray(R.sum(1)).ravel().astype(int)
    hits = np.zeros(top.shape, dtype=bool)
    for i in range(len(queries)):
        rs = R.indices[R.indptr[i]:R.indptr[i + 1]]
        hits[i] = np.isin(top[i], rs) & (top[i] >= 0)
    disc = 1.0 / np.log2(np.arange(2, top.shape[1] + 2))
    out = {}
    for k in ks:
        h = hits[:, :k]
        dcg = (h * disc[:k]).sum(1)
        ideal_n = np.minimum(k, n_rel)
        cum = np.r_[0.0, np.cumsum(disc[:k])]
        out[k] = dict(p=h.sum(1) / k, r=h.sum(1) / np.maximum(n_rel, 1), ndcg=dcg / np.maximum(cum[ideal_n], 1e-12),
                      hit=h.any(1).astype(float), cap=ideal_n / k)
    return out, n_rel


def label_matrix(n, labels):
    """n x n boolean: do two nodes share a label? Unlabelled rows are all False."""
    names = sorted({x for v in labels.values() for x in v})
    pos = {x: i for i, x in enumerate(names)}
    r, c = zip(*[(u, pos[x]) for u, v in labels.items() for x in v])
    L = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(n, len(names)))
    return (L @ L.T).astype(bool).toarray()


def label_metrics(top, queries, share, ks):
    """Per-query agreement@K and base rate. Base rate: share of the other n-1 nodes sharing a label."""
    n = share.shape[0]
    base = (share[queries].sum(1) - share[queries, queries]) / (n - 1)
    out = {}
    for k in ks:
        t = top[:, :k]
        ok = t >= 0
        got = np.take_along_axis(share[queries], np.where(ok, t, 0), axis=1) & ok
        out[k] = got.sum(1) / np.maximum(ok.sum(1), 1)
    return out, base


def macro(out, ks):
    return {str(k): {m: float(v.mean()) for m, v in out[k].items()} for k in ks}
