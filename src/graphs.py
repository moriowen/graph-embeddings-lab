"""Load a built graph and split it. Shared by every entry point."""
import json
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from config import DATA


@dataclass
class Graph:
    name: str
    titles: list          # index i is node id i + 1
    edges: np.ndarray     # (m, 2) 0-based, a < b
    labels: dict          # 0-based index -> list of labels

    @property
    def n(self):
        return len(self.titles)

    def index(self, title):
        return self.titles.index(title)


def load(name, scheme="labels"):
    d = DATA / name
    titles = [l.split("\t", 1)[1] for l in (d / "nodes.tsv").read_text(encoding="utf8").splitlines()]
    edges = np.loadtxt(d / "edges.tsv", dtype=int, ndmin=2) - 1
    raw = json.loads((d / "labels.json").read_text())
    labels = {int(k) - 1: v[scheme] for k, v in raw.items() if v.get(scheme)}
    return Graph(name, titles, edges, labels)


def adj(n, edges):
    """Symmetric CSR adjacency from (m, 2) edges."""
    if len(edges) == 0:
        return sp.csr_matrix((n, n), dtype=np.int32)
    r = np.r_[edges[:, 0], edges[:, 1]]
    c = np.r_[edges[:, 1], edges[:, 0]]
    return sp.csr_matrix((np.ones(len(r), dtype=np.int32), (r, c)), shape=(n, n))
