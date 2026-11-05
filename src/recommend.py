"""The one place candidates are filtered and ranked, so evaluation and the report cannot disagree."""
import numpy as np
import scipy.sparse as sp


class EmbeddingScorer:
    """score(queries) -> dense (len(queries), n). metric: cosine (default), dot or euclidean."""

    def __init__(self, X, metric="cosine"):
        self.metric, self.X = metric, X
        if metric == "cosine":
            norm = np.linalg.norm(X, axis=1, keepdims=True)
            self.Y = X / np.where(norm == 0, 1, norm)
        else:
            self.Y = X

    def __call__(self, q):
        if self.metric == "euclidean":
            d = (self.X[q] ** 2).sum(1, keepdims=True) - 2 * self.X[q] @ self.X.T + (self.X ** 2).sum(1)
            return -d
        return self.Y[q] @ self.Y.T


def rank(score, queries, k, exclude=None, batch=512):
    """Top-k candidates per query, best first. The query itself is never a candidate; `exclude` (sparse n x n,
    nonzero = excluded) removes more. Rows with fewer than k candidates are padded with -1."""
    queries = np.asarray(queries)
    out = np.full((len(queries), k), -1, dtype=np.int32)
    for i in range(0, len(queries), batch):
        q = queries[i:i + batch]
        S = np.array(score(q), dtype=np.float64)
        S[np.arange(len(q)), q] = -np.inf
        if exclude is not None:
            r, c = exclude[q].nonzero()
            S[r, c] = -np.inf
        idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
        vals = np.take_along_axis(S, idx, axis=1)
        o = np.argsort(-vals, axis=1, kind="stable")
        idx, vals = np.take_along_axis(idx, o, axis=1), np.take_along_axis(vals, o, axis=1)
        idx[np.isneginf(vals)] = -1
        out[i:i + len(q)] = idx
    return out


def known(n, *edge_sets):
    """Sparse mask of edges already known, for excluding them from candidates."""
    from src.graphs import adj
    e = np.vstack([x for x in edge_sets if len(x)]) if edge_sets else np.empty((0, 2), int)
    return adj(n, e).astype(bool).tocsr() if len(e) else sp.csr_matrix((n, n), dtype=bool)
