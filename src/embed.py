"""One entry point for every embedding config: train edges + params -> vectors, phase timings, peak RSS.

The walks come from the vendored repos (handout requirement). Both go through gensim's Word2Vec with explicit
hs/negative, so a row's loss is exactly what its config says.
"""
import random
import resource
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "vendor"))

from deepwalk import graph as dw_graph  # noqa: E402
from gensim.models import Word2Vec  # noqa: E402
from node2vec import Node2Vec  # noqa: E402


class TimedNode2Vec(Node2Vec):
    """eliorc's constructor runs precompute and walks together; this only times them separately."""

    def _precompute_probabilities(self):
        t = time.perf_counter()
        super()._precompute_probabilities()
        self.precompute_s = time.perf_counter() - t

    def _generate_walks(self):
        t = time.perf_counter()
        walks = super()._generate_walks()
        self.walk_s = time.perf_counter() - t
        return walks


def peak_rss_mb():
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v / 1e6 if sys.platform == "darwin" else v / 1e3


def to_nx(n, edges):
    """Node ids run from 1 because vendored DeepWalk's `if start:` treats node 0 as no start."""
    G = nx.Graph()
    G.add_nodes_from(range(1, n + 1))
    G.add_edges_from((int(a) + 1, int(b) + 1) for a, b in edges)
    return G


def train(n, edges, c, seed, workers=1):
    """Returns (vectors in node order, timings, walks). Nodes with no walk appearance get a zero vector."""
    G = to_nx(n, edges)
    t = {"precompute_s": 0.0}
    w2v = dict(vector_size=c["d"], window=c["w"], min_count=0, sg=1, hs=c["hs"], negative=c["negative"],
               sample=c["sample"], epochs=c["epochs"], workers=workers, seed=seed)
    if c["algo"] == "dw":
        t0 = time.perf_counter()
        DG = dw_graph.from_networkx(G, undirected=True)
        walks = dw_graph.build_deepwalk_corpus(DG, num_paths=c["r"], path_length=c["l"], alpha=0,
                                               rand=random.Random(seed))
        t["walk_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        model = Word2Vec(walks, **w2v)
    else:
        nv = TimedNode2Vec(G, dimensions=c["d"], walk_length=c["l"], num_walks=c["r"], p=c["p"], q=c["q"],
                           workers=workers, quiet=True, seed=seed)
        walks = nv.walks
        t["precompute_s"], t["walk_s"] = nv.precompute_s, nv.walk_s
        t0 = time.perf_counter()
        w2v.pop("vector_size")
        model = nv.fit(**w2v)
    t["train_s"] = time.perf_counter() - t0
    X = np.zeros((n, c["d"]), dtype=np.float32)
    for i in range(n):
        k = str(i + 1)
        if k in model.wv:
            X[i] = model.wv[k]
    return X, t, walks
