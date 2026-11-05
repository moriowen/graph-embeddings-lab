"""Acceptance checks, written before any results were read. python -m src.check [--small-only]

1 size bands  2 held-out edges absent from training walks  3 query never in its own list
4 metrics match a hand count on a toy graph  5 same seed gives the same embedding with workers=1
6 split is connected, disjoint, and covers every edge  7 rank() honours exclusions and pads
"""
import numpy as np

from config import GRAPHS, QUERIES, resolve
from src import baselines
from src.embed import train
from src.evaluate import label_matrix, label_metrics, link_metrics
from src.graphs import adj, load
from src.recommend import EmbeddingScorer, known, rank
from src.split import load_split, make_split

FAST = dict(d=16, l=20, r=3, w=3, epochs=1, sample=1e-3, hs=0, negative=5)


def ok(msg):
    print("PASS", msg)


def check_toy_metrics():
    # Query 0. Ranked 1, 2, 3, 4. Held-out neighbours {2, 4}.
    rel = adj(6, np.array([[0, 2], [0, 4]])).tocsr()
    top = np.array([[1, 2, 3, 4]])
    m, n_rel = link_metrics(top, np.array([0]), rel, (2, 4))
    # K=2: 1 hit of 2. DCG = 1/log2(3). IDCG = 1 + 1/log2(3).
    d2, d3, d5 = 1.0, 1 / np.log2(3), 1 / np.log2(5)
    assert n_rel[0] == 2
    assert np.isclose(m[2]["p"][0], 0.5) and np.isclose(m[2]["r"][0], 0.5) and m[2]["hit"][0] == 1
    assert np.isclose(m[2]["ndcg"][0], d3 / (d2 + d3))
    assert np.isclose(m[2]["cap"][0], 1.0)
    # K=4: 2 hits. Hits at ranks 2 and 4. p=0.5, r=1, cap=2/4.
    assert np.isclose(m[4]["p"][0], 0.5) and np.isclose(m[4]["r"][0], 1) and np.isclose(m[4]["cap"][0], 0.5)
    assert np.isclose(m[4]["ndcg"][0], (d3 + d5) / (d2 + d3))
    # Labels: 0,1,2 share "a"; 3 has "b"; 2 also "b". Base for 0 = 2/4 with n=5.
    share = label_matrix(5, {0: ["a"], 1: ["a"], 2: ["a", "b"], 3: ["b"]})
    a, base = label_metrics(np.array([[1, 3, -1]]), np.array([0]), share, (2, 3))
    assert np.isclose(base[0], 0.5) and np.isclose(a[2][0], 0.5) and np.isclose(a[3][0], 0.5)
    ok("metrics match the hand count on a toy graph")


def check_rank():
    X = np.eye(4, dtype=np.float32) + 0.1
    excl = known(4, np.array([[0, 1]]))
    top = rank(EmbeddingScorer(X), np.array([0, 1]), 3, excl)
    assert 0 not in top[0] and 1 not in top[0] and 1 not in top[1] and 0 not in top[1]
    assert top[0, 2] == -1, "padded when fewer candidates than k"
    ok("rank excludes the query and known edges, and pads")


def check_baselines():
    # Path 0-1-2 and 0-3-2 give nodes 0 and 2 two common neighbours; 1 has degree 2.
    e = np.array([[0, 1], [1, 2], [0, 3], [2, 3], [3, 4]])
    cn = baselines.make("base_cn", 5, e, 0)
    aa = baselines.make("base_aa", 5, e, 0)
    assert np.isclose(cn(np.array([0]))[0, 2], 2, atol=1e-6)
    want = 1 / np.log(2) + 1 / np.log(3)  # via node 1 (deg 2), node 3 (deg 3)
    assert np.isclose(aa(np.array([0]))[0, 2], want, atol=1e-6)
    pop = baselines.make("base_popularity", 5, e, 0)
    assert pop(np.array([0]))[0].argmax() == 3
    ok("CN and Adamic-Adar match a hand count; popularity picks the highest degree")


def check_graph(name, algos=("dw_ns", "n2v_ns")):
    g = load(name)
    lo, hi = GRAPHS[name]
    assert lo <= g.n <= hi
    s = load_split(name)
    got = {tuple(e) for k in s for e in s[k].tolist()}
    assert len(got) == len(g.edges) == sum(len(v) for v in s.values()), "disjoint and complete"
    A = adj(g.n, s["train"])
    from scipy.sparse.csgraph import connected_components
    assert connected_components(A, directed=False)[0] == 1
    held = {tuple(sorted(e)) for k in ("val", "test") for e in s[k].tolist()}
    for cfg in algos:
        c = resolve(cfg) | FAST
        X, _, walks = train(g.n, s["train"], c, 7)
        pairs = {tuple(sorted((int(a) - 1, int(b) - 1))) for w in walks for a, b in zip(w, w[1:])}
        assert not pairs & held, "held-out edge appears in a walk"
        assert {int(x) for w in walks for x in w} == set(range(1, g.n + 1)), "every node walked"
        Y, _, walks2 = train(g.n, s["train"], c, 7)
        assert walks == walks2 and np.array_equal(X, Y), "same seed, same embedding"
        Z, _, _ = train(g.n, s["train"], c, 8)
        assert not np.array_equal(X, Z), "different seed, different embedding"
        top = rank(EmbeddingScorer(X), np.arange(g.n), 20)
        assert not (top == np.arange(g.n)[:, None]).any(), "query in its own list"
        ok(f"{name} {cfg}: no held-out edge in {len(walks)} walks; reproducible; query never in own list")
    if name == "small":
        assert all(q in g.titles for q in QUERIES)


def main():
    import sys
    check_toy_metrics()
    check_rank()
    check_baselines()
    e = load("small").edges
    s = make_split(222, e, 1)
    assert len(s["test"]) == round(0.2 * len(e)) and len(s["val"]) == round(0.1 * len(e))
    ok("split sizes are 20% and 10%")
    check_graph("small")
    if "--small-only" not in sys.argv:
        check_graph("large")


if __name__ == "__main__":
    main()
