"""Turn results/run_*.json + results/nodes/*.npz into results/summary.json and results/tuning.json.

    python -m src.stats

Reads only result files. Timing tables use only records from the dedicated timing job (tag timing_), quality uses only
the grid. Paired tests compare per-node test NDCG@20 averaged over seeds first, so seeds are not treated as independent
nodes. Selection of p and q reads validation scores only.
"""
import json
import re
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

from config import BASELINES, KS_ALL, MAIN_GRAPHS, MATRIX, NODES, PQ_GRID, RESULTS, SEEDS, TUNE_SEEDS

RUN = re.compile(r"run_(?:(?P<tag>timing|dev)_)?(?P<graph>small|large|fb_small|fb_large)_(?P<config>.+)_split(?P<split>\d+)_s(?P<seed>\d+)(?:_w(?P<w>\d+))?\.json")


def load(tag=None, include_workers=False):
    out = []
    for p in sorted(RESULTS.glob("run_*.json")):
        m = RUN.fullmatch(p.name)
        if not m or m["tag"] != tag or (m["w"] and not include_workers):
            continue
        r = json.loads(p.read_text())
        r["_stem"] = p.stem[4:]
        out.append(r)
    return out


def ms(xs):
    xs = np.array(xs, dtype=float)
    return dict(mean=float(xs.mean()), sd=float(xs.std(ddof=1)) if len(xs) > 1 else 0.0, n=len(xs))


def get_nodes(stem):
    with np.load(NODES / f"{stem}.npz") as z:
        return {k: z[k] for k in z.files}


def quality(runs):
    """graph -> config -> split -> metric mean +- sd over seeds."""
    g = defaultdict(list)
    for r in runs:
        if r["seed"] in TUNE_SEEDS:   # selection runs never appear in a reported row
            continue
        g[(r["graph"], r["config"], r["split"])].append(r)
    out = defaultdict(lambda: defaultdict(dict))
    for (gr, c, sp), rs in g.items():
        e = dict(n_seeds=len(rs), seeds=sorted(r["seed"] for r in rs), test={}, val={}, label={}, novel={},
                 base=rs[0]["label_literal"]["base"], n_test=rs[0]["test"]["n_nodes"], n_label=rs[0]["label_literal"]["n_nodes"])
        for k in map(str, KS_ALL):
            for part in ("test", "val"):
                e[part][k] = {m: ms([r[part]["metrics"][k][m] for r in rs]) for m in ("p", "r", "ndcg", "hit", "cap")}
            e["label"][k] = dict(agree=ms([r["label_literal"]["agree"][k] for r in rs]),
                                 lift=ms([r["label_literal"]["lift"][k] for r in rs]))
            e["novel"][k] = dict(agree=ms([r["label_novel"]["agree"][k] for r in rs]),
                                 lift=ms([r["label_novel"]["lift"][k] for r in rs]))
        out[gr][c][str(sp)] = e
    return out


def select_pq(runs):
    """Mean validation NDCG@20 over the tuning seeds, per graph. Ties go to p=q=1, then to the earlier grid cell."""
    out = {}
    for g in MAIN_GRAPHS:
        cells = {}
        for p in PQ_GRID:
            for q in PQ_GRID:
                c = f"n2v_p{p:g}_q{q:g}"
                rs = [r for r in runs if r["graph"] == g and r["config"] == c and r["seed"] in TUNE_SEEDS and r["split"] == 0]
                if len(rs) == len(TUNE_SEEDS):
                    cells[c] = dict(p=p, q=q, val=ms([r["val"]["metrics"]["20"]["ndcg"] for r in rs]),
                                    test_not_used=ms([r["test"]["metrics"]["20"]["ndcg"] for r in rs]),
                                    lift_not_used=ms([r["label_literal"]["lift"]["20"] for r in rs]))
        if len(cells) < len(PQ_GRID) ** 2:
            continue
        best = max(cells, key=lambda c: (round(cells[c]["val"]["mean"], 6), c == "n2v_p1_q1"))
        out[g] = dict(config=best, p=cells[best]["p"], q=cells[best]["q"], cells=cells, seeds=list(TUNE_SEEDS),
                      criterion="validation NDCG@20")
    return out


def seed_means(graph, config, split=0, seeds=SEEDS, key="ndcg20"):
    """Per-seed arrays over the same nodes, plus the node index."""
    a, idx = [], None
    q = "test_q" if key.startswith("ndcg") else "label_q"
    for s in seeds:
        p = NODES / f"{graph}_{config}_split{split}_s{s}.npz"
        if p.exists():
            with np.load(p) as z:
                a.append(z[key].astype(float))
                idx = z[q]
    return a, idx


def per_node(graph, config, split=0, seeds=SEEDS, key="ndcg20"):
    a, idx = seed_means(graph, config, split, seeds, key)
    return (np.mean(a, axis=0), idx, len(a)) if a else (None, None, 0)


def paired(graph, c1, c2, key="ndcg20", split=0, n_boot=2000):
    """Paired per-node difference c1 - c2: mean, bootstrap 95% CI over nodes, Wilcoxon p."""
    a, ia, na = per_node(graph, c1, split, key=key)
    b, ib, nb = per_node(graph, c2, split, key=key)
    if a is None or b is None or not np.array_equal(ia, ib):
        return None
    d = a - b
    rng = np.random.default_rng(6220)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    nz = d[d != 0]
    # A CI over nodes treats the seed-averaged embeddings as fixed. These show the noise it leaves out: the spread of
    # the graph-level score across seeds, and the difference in units of its standard error (Welch).
    sa = np.array([x.mean() for x in seed_means(graph, c1, split, key=key)[0]])
    sb = np.array([x.mean() for x in seed_means(graph, c2, split, key=key)[0]])
    se = float(np.sqrt(sa.var(ddof=1) / len(sa) + sb.var(ddof=1) / len(sb))) if min(len(sa), len(sb)) > 1 else None
    return dict(seed_sd_a=float(sa.std(ddof=1)) if len(sa) > 1 else None, seed_sd_b=float(sb.std(ddof=1)) if len(sb) > 1 else None,
                seed_se_diff=se, seed_t=float(d.mean() / se) if se else None, a=c1, b=c2, mean_a=float(a.mean()), mean_b=float(b.mean()), diff=float(d.mean()),
                ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                wilcoxon_p=float(wilcoxon(nz).pvalue) if len(nz) else 1.0,
                wins=int((d > 0).sum()), losses=int((d < 0).sum()), ties=int((d == 0).sum()), n=len(d), seeds=[na, nb])


def jaccard(graph, config, split=0, seeds=SEEDS):
    """Mean top-K Jaccard across all seed pairs, over all nodes (literal lists)."""
    tops = []
    for s in seeds:
        p = NODES / f"{graph}_{config}_split{split}_s{s}.npz"
        if p.exists():
            with np.load(p) as z:
                tops.append(z["literal_top"])
    if len(tops) < 2:
        return None
    out = {}
    for k in (5, 20):
        vals = []
        for i in range(len(tops)):
            for j in range(i + 1, len(tops)):
                inter = (tops[i][:, :k, None] == tops[j][:, None, :k]).sum((1, 2))
                vals.append(inter / (2 * k - inter))
        v = np.mean(vals, axis=0)
        out[str(k)] = dict(mean=float(v.mean()), p10=float(np.percentile(v, 10)), p90=float(np.percentile(v, 90)), per_node_sd=float(v.std()))
    return out


def timing(runs):
    g = defaultdict(list)
    for r in runs:
        g[(r["graph"], r["config"], r["workers"])].append(r)
    out = defaultdict(lambda: defaultdict(dict))
    for (gr, c, w), rs in g.items():
        e = {m: ms([r["timings"][m] for r in rs]) for m in ("precompute_s", "walk_s", "train_s", "total_s")}
        e.update(latency_mean_ms=ms([r["latency"]["mean_ms"] for r in rs]), latency_p95_ms=ms([r["latency"]["p95_ms"] for r in rs]),
                 peak_rss_mb=ms([r["peak_rss_mb"] for r in rs]), machine=sorted({r["machine"] for r in rs}))
        out[gr][c][str(w)] = e
    return out


def main():
    runs = load()
    tun = select_pq(runs)
    (RESULTS / "tuning.json").write_text(json.dumps(tun, indent=1))
    Q = json.loads(json.dumps(quality(runs)))
    def tuned(g):
        c = tun.get(g, {}).get("config")
        return "n2v_ns" if c == "n2v_p1_q1" else c
    for g in MAIN_GRAPHS:
        t = tuned(g)
        if t and t in Q.get(g, {}):
            Q[g]["n2v_tuned"] = Q[g][t]
    pairs = {}
    for g in MAIN_GRAPHS:
        t = tuned(g)
        spec = [("dw_hsns", "n2v_ns"), ("dw_ns", "n2v_ns"), ("dw_hs", "n2v_hs"), ("dw_hsns", "n2v_hsns"),  # A/A + headline
                ("dw_hs", "dw_ns"), ("dw_hsns", "dw_ns"), ("n2v_hs", "n2v_ns"), ("n2v_hsns", "n2v_ns"),      # loss
                ("base_aa", "dw_hsns"), ("base_aa", "n2v_ns"), ("base_cn", "dw_hsns"), ("base_aa", "base_cn"), ("base_popularity", "dw_hsns")]
        if t:
            spec += [(t, "n2v_ns"), (t, "dw_hsns"), ("base_aa", t)]
        for key in ("ndcg20", "ndcg5"):
            for a, b in spec:
                r = paired(g, a, b, key)
                if r:
                    pairs[f"{g}:{key}:{a}-{b}"] = r
        for key in ("agree20", "agree5"):
            for a, b in spec:
                r = paired(g, a, b, key)
                if r:
                    pairs[f"{g}:{key}:{a}-{b}"] = r
    jac = {g: {c: jaccard(g, c) for c in list(MATRIX) + ([tuned(g)] if tuned(g) else [])} for g in MAIN_GRAPHS}
    out = dict(quality=Q, pairs=pairs, jaccard=jac, timing=json.loads(json.dumps(timing(load("timing", True)))), n_runs=len(runs))
    (RESULTS / "summary.json").write_text(json.dumps(out, indent=1))
    print(f"{len(runs)} runs")
    for g in Q:
        print(g)
        for c in sorted(Q[g], key=lambda c: -Q[g][c].get("0", {}).get("test", {}).get("20", {}).get("ndcg", {}).get("mean", 0)):
            e = Q[g][c].get("0")
            if not e or re.match(r"n2v_p", c):
                continue
            t, l = e["test"]["20"]["ndcg"], e["label"]["20"]["lift"]
            print(f"  {c:18s} n={e['n_seeds']} ndcg20 {t['mean']:.4f}±{t['sd']:.4f}  lift20 {l['mean']:.3f}±{l['sd']:.3f}")
    for g, t in tun.items():
        print(g, "selected", t["config"], t["cells"][t["config"]]["val"])


if __name__ == "__main__":
    main()
