"""Seeded train/val/test edge split. python -m src.split

A random spanning tree is protected, so training stays connected and every node keeps a walkable neighbour.
Test and validation edges come from the rest. Result: data/<graph>/split_<k>.npz holding 0-based edges.
Consequence, reported: nodes with few edges are over-protected, so held-out metrics lean towards higher-degree nodes.
"""
import numpy as np

from config import DATA, GRAPHS, SPLIT_TEST, SPLIT_VAL, SPLITS
from src.graphs import load


def make_split(n, edges, seed, f_test=SPLIT_TEST, f_val=SPLIT_VAL):
    rng = np.random.default_rng(seed)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    order = rng.permutation(len(edges))
    tree, rest = [], []
    for i in order:
        a, b = find(edges[i, 0]), find(edges[i, 1])
        if a == b:
            rest.append(i)
        else:
            parent[a] = b
            tree.append(i)
    assert len(tree) == n - 1, "graph must be connected"
    m = len(edges)
    n_test, n_val = round(f_test * m), round(f_val * m)
    assert n_test + n_val <= len(rest), "not enough non-tree edges"
    test, val = rest[:n_test], rest[n_test:n_test + n_val]
    train = tree + rest[n_test + n_val:]
    return {k: edges[np.sort(v)] for k, v in dict(train=train, val=val, test=test).items()}


def load_split(name, k=0):
    with np.load(DATA / name / f"split_{k}.npz") as z:
        return {x: z[x] for x in ("train", "val", "test")}


def main():
    for name in GRAPHS:
        g = load(name)
        for k in SPLITS:
            s = make_split(g.n, g.edges, 6220 + k)
            np.savez_compressed(DATA / name / f"split_{k}.npz", **s)
        print(name, g.n, {x: len(v) for x, v in s.items()})


if __name__ == "__main__":
    main()
