"""Non-embedding recommenders, scored with the same rank() and candidates as the embeddings.

Each adds tiny seeded noise, so ties (very common in CN, total in popularity among equal degrees) break at
random instead of by node id. All use training edges only.
"""
import numpy as np

from src.graphs import adj

EPS = 1e-9


class _Noise:
    def __init__(self, n, seed):
        self.n, self.seed = n, seed

    def noise(self, q):
        rng = np.random.default_rng([self.seed, int(q[0])])
        return EPS * rng.random((len(q), self.n))


class Random(_Noise):
    def __call__(self, q):
        return self.noise(q) / EPS


class Popularity(_Noise):
    def __init__(self, n, train, seed):
        super().__init__(n, seed)
        self.deg = np.asarray(adj(n, train).sum(1), dtype=np.float64).ravel()

    def __call__(self, q):
        return np.tile(self.deg, (len(q), 1)) + self.noise(q)


class CommonNeighbours(_Noise):
    def __init__(self, n, train, seed):
        super().__init__(n, seed)
        self.A = adj(n, train).astype(np.float64)

    def __call__(self, q):
        return np.asarray((self.A[q] @ self.A).todense()) + self.noise(q)


class AdamicAdar(_Noise):
    def __init__(self, n, train, seed):
        super().__init__(n, seed)
        A = adj(n, train).astype(np.float64)
        d = np.asarray(A.sum(1)).ravel()
        w = np.divide(1.0, np.log(d), out=np.zeros(n), where=d > 1)
        self.A, self.w = A, w

    def __call__(self, q):
        import scipy.sparse as sp
        return np.asarray((self.A[q] @ sp.diags(self.w) @ self.A).todense()) + self.noise(q)


def make(name, n, train, seed):
    if name == "base_random":
        return Random(n, seed)
    return {"base_popularity": Popularity, "base_cn": CommonNeighbours, "base_aa": AdamicAdar}[name](n, train, seed)
