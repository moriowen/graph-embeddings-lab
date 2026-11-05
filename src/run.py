"""One config, one seed, one split -> results/run_<graph>_<config>_split<k>_s<seed>.json

    python -m src.run --graph small --config n2v_ns --seed 0

The record holds parameters, machine, commits, phase timings, peak RSS, query latency, both quality measures
and the query nodes' lists. Per-node arrays go to results/nodes/ for paired tests; vectors to embeddings/.
"""
import argparse
import json
import os
import platform
import time

import gensim
import numpy as np

from config import BASELINES, COMMITS, EMB, KMAX, KS_ALL, MACHINE, NODES, QUERIES, RESULTS, resolve
from src import baselines
from src.embed import peak_rss_mb, train
from src.evaluate import label_matrix, label_metrics, link_metrics, macro
from src.graphs import adj, load
from src.recommend import EmbeddingScorer, known, rank
from src.split import load_split

N_LATENCY = 200


def latency(score, n, k, exclude):
    rng = np.random.default_rng(0)
    q = rng.integers(0, n, N_LATENCY)
    ts = []
    for u in q:
        t = time.perf_counter()
        rank(score, [u], k, exclude)
        ts.append(time.perf_counter() - t)
    ts = np.array(ts) * 1e3
    return dict(mean_ms=float(ts.mean()), p95_ms=float(np.percentile(ts, 95)), n=N_LATENCY, k=k)


def evaluate(g, sp_, score, seed_tag=None):
    """Everything computed from a scorer. Returns (record parts, per-node arrays)."""
    n, tr, va, te = g.n, sp_["train"], sp_["val"], sp_["test"]
    all_q = np.arange(n)
    rec, arrays = {}, {}

    # Held-out: test excludes train + val edges; val excludes train edges only.
    for name, rel_e, excl in (("test", te, known(n, tr, va)), ("val", va, known(n, tr))):
        rel = adj(n, rel_e).tocsr()
        q = np.flatnonzero(np.asarray(rel.sum(1)).ravel() > 0)
        top = rank(score, q, KMAX, excl)
        m, n_rel = link_metrics(top, q, rel, KS_ALL)
        rec[name] = dict(n_nodes=len(q), metrics=macro(m, KS_ALL))
        if name == "test":
            arrays.update(test_q=q.astype(np.int32), test_top=top[:, :20].astype(np.int16),
                          ndcg5=m[5]["ndcg"].astype(np.float32), ndcg20=m[20]["ndcg"].astype(np.float32))

    # Labels: "literal" is the handout's top-K nearest (only the query excluded); "novel" hides training neighbours.
    share = label_matrix(n, g.labels)
    lq = np.array(sorted(g.labels))
    lit = rank(score, all_q, KMAX)
    for name, top in (("literal", lit[lq]), ("novel", rank(score, lq, KMAX, known(n, tr)))):
        a, base = label_metrics(top, lq, share, KS_ALL)
        rec["label_" + name] = dict(n_nodes=len(lq), base=float(base.mean()),
                                    agree={str(k): float(a[k].mean()) for k in KS_ALL},
                                    lift={str(k): float(a[k].mean() / base.mean()) for k in KS_ALL})
        if name == "literal":
            arrays.update(label_q=lq.astype(np.int32), agree5=a[5].astype(np.float32),
                          agree20=a[20].astype(np.float32), base=base.astype(np.float32))
    arrays["literal_top"] = lit[:, :20].astype(np.int16)

    # Required lists: the five query nodes, literal mode, top 20 (K=5 is its prefix).
    tr_m, te_m, va_m = known(n, tr), known(n, te), known(n, va)
    lists = {}
    for t in QUERIES:
        if t not in g.titles:
            continue
        u = g.index(t)
        lists[t] = [dict(title=g.titles[v], train=bool(tr_m[u, v]), held_out=bool(te_m[u, v] or va_m[u, v]),
                         label=bool(share[u, v])) for v in lit[u, :20] if v >= 0]
    rec["lists"] = lists
    return rec, arrays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    stem = f"{a.tag + '_' if a.tag else ''}{a.graph}_{a.config}_split{a.split}_s{a.seed}"
    if a.workers != 1:
        stem += f"_w{a.workers}"
    path = RESULTS / f"run_{stem}.json"
    if path.exists() and not a.force:
        print("exists", path.name)
        return
    g, sp_ = load(a.graph), load_split(a.graph, a.split)
    t0 = time.perf_counter()
    if a.config in BASELINES:
        c = dict(algo="baseline")
        score = baselines.make(a.config, g.n, sp_["train"], a.seed)
        timings = dict(precompute_s=0.0, walk_s=0.0, train_s=time.perf_counter() - t0)
        X = None
    else:
        c = resolve(a.config)
        tr = sp_["train"]
        if c.get("hubdrop"):   # S8: the top fraction of nodes by training degree never enter a walk
            deg = np.asarray(adj(g.n, tr).sum(1)).ravel()
            hubs = np.argsort(-deg)[:max(1, round(c["hubdrop"] * g.n))]
            tr = tr[~np.isin(tr, hubs).any(1)]
        X, timings, _ = train(g.n, tr, c, a.seed, a.workers)
        score = EmbeddingScorer(X)
    timings["total_s"] = time.perf_counter() - t0
    rec = dict(graph=a.graph, config=a.config, params=c, seed=a.seed, split=a.split, workers=a.workers,
               machine=MACHINE, commits=COMMITS, gensim=gensim.__version__, python=platform.python_version(),
               hashseed=os.environ.get("PYTHONHASHSEED"), timings=timings)
    rec["latency"] = latency(score, g.n, 20, None)
    parts, arrays = evaluate(g, sp_, score)
    rec.update(parts)
    rec["peak_rss_mb"] = peak_rss_mb()
    NODES.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(NODES / f"{stem}.npz", **arrays)
    if X is not None:
        EMB.mkdir(exist_ok=True)
        np.save(EMB / f"{stem}.npy", X)
    path.write_text(json.dumps(rec, indent=1))
    t = rec["test"]["metrics"]["20"]
    print(f"{stem}: ndcg20={t['ndcg']:.4f} p20={t['p']:.4f} lift20={rec['label_literal']['lift']['20']:.2f} "
          f"{timings['total_s']:.1f}s {rec['peak_rss_mb']:.0f}MB")


if __name__ == "__main__":
    main()
