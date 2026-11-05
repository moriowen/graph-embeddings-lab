"""Part 2 analyses that need vectors, not just run records -> results/part2_<graph>.json and results/part2_cross.json

    python -m src.part2 small          # on the Mac
    python -m src.part2 large          # on ICE; embeddings live there
    python -m src.part2 cross          # S5, needs small and large embeddings

Every block answers one Part 2 question and uses src.recommend.rank, so candidates match the required results.
"""
import json
import sys

import numpy as np
from scipy.stats import spearmanr

from config import EMB, RESULTS, SEEDS, SPLITS
from src import baselines
from src.evaluate import label_matrix, label_metrics, link_metrics
from src.graphs import adj, load
from src.recommend import EmbeddingScorer, known, rank
from src.split import load_split

MAIN = ("dw_hsns", "dw_hs", "n2v_ns", "n2v_hs")
BASE = ("base_cn", "base_aa", "base_popularity")


def vec(g, c, s, split=0):
    p = EMB / f"{g}_{c}_split{split}_s{s}.npy"
    return np.load(p) if p.exists() else None


def tuned_cfg(g):
    t = json.loads((RESULTS / "tuning.json").read_text())
    return t[g]["config"] if g in t else None


def mean_over(fn, seeds):
    vals = [fn(s) for s in seeds]
    vals = [v for v in vals if v is not None]
    return dict(mean=float(np.mean(vals)), sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, n=len(vals)) if vals else None


def scorers(gname, g, tr, configs, seeds=SEEDS):
    for c in configs:
        for s in seeds:
            if c.startswith("base_"):
                yield c, s, baselines.make(c, g.n, tr, s)
            else:
                X = vec(gname, c, s)
                if X is not None:
                    yield c, s, EmbeddingScorer(X)


def strata(gname):
    """S3: who wins where. By query training degree, and on edges with vs without a common neighbour."""
    g, sp = load(gname), load_split(gname)
    tr, te = sp["train"], sp["test"]
    A = adj(g.n, tr).astype(np.int64).tocsr()
    deg = np.asarray(A.sum(1)).ravel()
    excl = known(g.n, tr, sp["val"])
    rel = adj(g.n, te).tocsr()
    q = np.flatnonzero(np.asarray(rel.sum(1)).ravel() > 0)
    cn2 = (A @ A).tocsr()
    two_hop = np.array([cn2[a, b] > 0 for a, b in te])
    # Held-out edges a query could only reach beyond two hops: CN and AA score them as ties at zero.
    rel_far = adj(g.n, te[~two_hop]).tocsr()
    rel_near = adj(g.n, te[two_hop]).tocsr()
    bins = np.quantile(deg[q], [0, .25, .5, .75, 1])
    out = dict(two_hop_share=float(two_hop.mean()), n_far=int((~two_hop).sum()), n_test=len(te),
               deg_bins=[float(x) for x in bins], table={})
    tcfg = tuned_cfg(gname)
    cfgs = list(MAIN) + list(BASE) + ([tcfg] if tcfg else [])
    per = {}
    for c, s, sc in scorers(gname, g, tr, cfgs):
        top = rank(sc, q, 20, excl)
        m, _ = link_metrics(top, q, rel, (20,))
        nd = m[20]["ndcg"]
        hits_far = np.array([np.isin(top[i], rel_far[q[i]].indices).sum() for i in range(len(q))])
        hits_near = np.array([np.isin(top[i], rel_near[q[i]].indices).sum() for i in range(len(q))])
        slots_2hop = np.mean([(cn2[q[i]].toarray().ravel()[top[i]] > 0).mean() for i in range(0, len(q), max(1, len(q) // 400))])
        e = per.setdefault(c, dict(ndcg_by_bin=[], far=[], near=[], top_in_2hop=[]))
        qd = np.digitize(deg[q], bins[1:-1])
        e["ndcg_by_bin"].append([float(nd[qd == b].mean()) for b in range(4)])
        e["far"].append(float(hits_far.sum()))
        e["near"].append(float(hits_near.sum()))
        e["top_in_2hop"].append(float(slots_2hop))
    for c, e in per.items():
        out["table"][c] = dict(ndcg_by_bin=np.mean(e["ndcg_by_bin"], 0).tolist(), far_hits=float(np.mean(e["far"])),
                               near_hits=float(np.mean(e["near"])), top_in_2hop=float(np.mean(e["top_in_2hop"])), n_seeds=len(e["far"]))
    return out


def hubs(gname):
    """S8: do hubs drive the lists? Share of slots taken, degree of what's recommended, and scores with hubs out."""
    g, sp = load(gname), load_split(gname)
    tr, te = sp["train"], sp["test"]
    deg = np.asarray(adj(g.n, tr).sum(1)).ravel()
    n_h = max(1, round(0.01 * g.n))
    hub = np.argsort(-deg)[:n_h]
    is_hub = np.isin(np.arange(g.n), hub)
    excl = known(g.n, tr, sp["val"])
    # Hub-free evaluation: hubs leave the candidates and the held-out set.
    excl_nh = excl.tolil(copy=True)
    excl_nh[:, hub] = True
    excl_nh = excl_nh.tocsr()
    te_nh = te[~np.isin(te, hub).any(1)]
    rel, rel_nh = adj(g.n, te).tocsr(), adj(g.n, te_nh).tocsr()
    q = np.flatnonzero(np.asarray(rel_nh.sum(1)).ravel() > 0)
    allq = np.arange(g.n)
    out = dict(hubs=[g.titles[i] for i in hub[:10]], n_hubs=n_h, edge_share_touching_hubs=float(np.isin(te, hub).any(1).mean()),
               mean_degree=float(deg.mean()), table={})
    tcfg = tuned_cfg(gname)
    for c in list(MAIN) + list(BASE) + ([tcfg] if tcfg else []):
        rows = []
        for cc, s, sc in scorers(gname, g, tr, [c]):
            lit = rank(sc, allq, 20)
            filt = rank(sc, q, 20, excl)
            filt_nh = rank(sc, q, 20, excl_nh)
            m, _ = link_metrics(filt, q, rel, (20,))
            mn, _ = link_metrics(filt_nh, q, rel_nh, (20,))
            freq = np.bincount(lit[:, :20].ravel(), minlength=g.n)
            top1 = np.sort(freq)[::-1]
            rows.append(dict(hub_slots=float(is_hub[lit[:, :20]].mean()), hub_slots_filtered=float(is_hub[filt[filt >= 0]].mean()),
                             rec_degree=float(deg[lit[:, :20]].mean()), ndcg=float(m[20]["ndcg"].mean()), ndcg_nohub=float(mn[20]["ndcg"].mean()),
                             gini=float(1 - 2 * np.cumsum(np.sort(freq)).sum() / (g.n * freq.sum()) + 1 / g.n), never=float((freq == 0).mean()),
                             spearman_deg=float(spearmanr(freq, deg).statistic), max_in=int(top1[0])))
        if rows:
            out["table"][c] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    return out


def distance(gname):
    """S9: cosine, dot and Euclidean on the same vectors."""
    g, sp = load(gname), load_split(gname)
    tr, te = sp["train"], sp["test"]
    deg = np.asarray(adj(g.n, tr).sum(1)).ravel()
    excl = known(g.n, tr, sp["val"])
    rel = adj(g.n, te).tocsr()
    q = np.flatnonzero(np.asarray(rel.sum(1)).ravel() > 0)
    share = label_matrix(g.n, g.labels)
    lq = np.array(sorted(g.labels))
    out = {}
    for c in ("dw_hsns", "n2v_ns"):
        rows = {m: [] for m in ("cosine", "dot", "euclidean")}
        for s in SEEDS:
            X = vec(gname, c, s)
            if X is None:
                continue
            base_top = rank(EmbeddingScorer(X, "cosine"), lq, 20)
            nrm = np.linalg.norm(X, axis=1)
            for metric in rows:
                sc = EmbeddingScorer(X, metric)
                top = rank(sc, q, 20, excl)
                m, _ = link_metrics(top, q, rel, (5, 20))
                lit = rank(sc, lq, 20)
                a, base = label_metrics(lit, lq, share, (5, 20))
                ov = (lit[:, :, None] == base_top[:, None, :]).sum((1, 2))
                rows[metric].append(dict(ndcg20=float(m[20]["ndcg"].mean()), ndcg5=float(m[5]["ndcg"].mean()),
                                         lift20=float(a[20].mean() / base.mean()), lift5=float(a[5].mean() / base.mean()),
                                         jaccard_cos=float((ov / (40 - ov)).mean()), rec_degree=float(deg[lit].mean()),
                                         norm_deg_spearman=float(spearmanr(nrm, deg).statistic), rec_norm=float(nrm[lit].mean() / nrm.mean())))
        out[c] = {m: {k: float(np.mean([r[k] for r in rs])) for k in rs[0]} | {"n": len(rs)} for m, rs in rows.items() if rs}
    return out


def cross():
    """S5: same articles, 20x the graph. Physics-only ranking on the large graph vs the small graph's ranking."""
    S, L = load("small"), load("large")
    sub = {t: i for i, t in enumerate(S.titles)}
    li = np.array([L.index(t) for t in S.titles if t in L.titles])
    si = np.array([sub[t] for t in S.titles if t in L.titles])
    Lp = load("large", "phys")
    is_phys = np.isin(np.arange(L.n), li)
    non_phys = np.flatnonzero(~is_phys)
    from scipy.sparse import csr_matrix
    # Mask non-physics candidates so that the large graph's ranking covers the same 222 articles.
    r = np.repeat(li, len(non_phys))
    c = np.tile(non_phys, len(li))
    mask = csr_matrix((np.ones(len(r), bool), (r, c)), shape=(L.n, L.n))
    share_s = label_matrix(S.n, S.labels)
    # Subcategory agreement on the shared label scheme, on both graphs.
    share_l = label_matrix(L.n, {u: v for u, v in Lp.labels.items() if u in set(li.tolist())})
    out = {}
    sdeg, ldeg = np.asarray(adj(S.n, load_split("small")["train"]).sum(1)).ravel(), np.asarray(adj(L.n, load_split("large")["train"]).sum(1)).ravel()
    out["deg_ratio_median"] = float(np.median(ldeg[li] / np.maximum(sdeg[si], 1)))
    out["physics_edge_share_of_degree"] = float(np.mean(sdeg[si] / np.maximum(ldeg[li], 1)))
    configs = {}
    for c_ in ("dw_hsns", "n2v_ns"):
        rows, noise = [], []
        keep_s, keep_l = {}, {}
        for s in SEEDS:
            XS, XL = vec("small", c_, s), vec("large", c_, s)
            if XS is None or XL is None:
                continue
            ts = rank(EmbeddingScorer(XS), si, 20)
            tl_all = rank(EmbeddingScorer(XL), li, 20)
            tl = rank(EmbeddingScorer(XL), li, 20, mask)
            # Translate large-graph ids to small-graph ids so the lists can be intersected.
            to_s = -np.ones(L.n, int)
            to_s[li] = si
            tl_s = to_s[tl]
            r_ = {}
            for k in (5, 20):
                inter = (ts[:, :k, None] == tl_s[:, None, :k]).sum((1, 2))
                r_[f"jaccard{k}"] = float((inter / (2 * k - inter)).mean())
                ag_s = np.take_along_axis(share_s[si], ts[:, :k], 1).mean(1)
                ag_l = np.take_along_axis(share_l[li], tl[:, :k], 1).mean(1)
                ag_lall = np.take_along_axis(share_l[li], tl_all[:, :k], 1).mean(1)
                r_[f"agree_small{k}"], r_[f"agree_large_phys{k}"], r_[f"agree_large_all{k}"] = float(ag_s.mean()), float(ag_l.mean()), float(ag_lall.mean())
                r_[f"phys_share{k}"] = float(is_phys[tl_all[:, :k]].mean())
            rows.append(r_)
            keep_s[s], keep_l[s] = ts, tl_s
            if s == 0:
                ex = {}
                for t in ("HD_28185", "Integrated_circuit", "90377_Sedna", "Andromeda_Galaxy", "Special_relativity"):
                    i = list(si).index(sub[t])
                    ex[t] = dict(small=[S.titles[v] for v in ts[i, :5]], large_all=[L.titles[v] for v in tl_all[i, :5]],
                                 large_phys_share20=float(is_phys[tl_all[i]].mean()))
                configs.setdefault(c_, {})["examples"] = ex
        if rows:
            # Noise floor: the same graph across seeds. Cross-graph overlap only means something above this.
            seeds = sorted(keep_s)
            for name, store in (("small", keep_s), ("large", keep_l)):
                v = []
                for a in range(len(seeds)):
                    for b in range(a + 1, len(seeds)):
                        inter = (store[seeds[a]][:, :20, None] == store[seeds[b]][:, None, :20]).sum((1, 2))
                        v.append((inter / (40 - inter)).mean())
                configs[c_][f"seed_jaccard20_{name}"] = float(np.mean(v))
            configs[c_]["metrics"] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
            configs[c_]["n_seeds"] = len(rows)
    out["configs"] = configs
    out["base_small"] = float(np.mean([(share_s[i].sum() - 1) / (S.n - 1) for i in si]))
    return out


def main():
    arg = sys.argv[1]
    if arg == "cross":
        r = cross()
        (RESULTS / "part2_cross.json").write_text(json.dumps(r, indent=1))
    else:
        r = dict(strata=strata(arg), hubs=hubs(arg), distance=distance(arg))
        (RESULTS / f"part2_{arg}.json").write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=0)[:3500])


if __name__ == "__main__":
    main()
