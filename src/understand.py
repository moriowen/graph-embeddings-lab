"""Analyses over run records that need no vectors -> results/understand.json. python -m src.understand

Required lists (D4, D6, D7) with per-item stability; K behaviour; S2 selection audit; S6 subsampling arithmetic.
"""
import json

import numpy as np
from scipy.stats import spearmanr

from config import QUERIES, RESULTS, SEEDS, SHARED
from src.graphs import adj, load
from src.split import load_split

ALGOS = {"DeepWalk": "dw_hsns", "node2vec": "n2v_p2_q0.5"}


def run(g, c, s, split=0):
    return json.loads((RESULTS / f"run_{g}_{c}_split{split}_s{s}.json").read_text())


def lists(g):
    out = {}
    rec = {a: [run(g, c, s) for s in SEEDS] for a, c in ALGOS.items()}
    for t in QUERIES:
        e = {}
        for a, rs in rec.items():
            per_seed = [r["lists"][t] for r in rs]
            e[a] = {}
            for k in (5, 20):
                sets = [{x["title"] for x in l[:k]} for l in per_seed]
                items = [dict(x, seeds=sum(x["title"] in st for st in sets)) for x in per_seed[0][:k]]
                jac = [len(sets[i] & sets[j]) / len(sets[i] | sets[j]) for i in range(5) for j in range(i + 1, 5)]
                e[a][str(k)] = dict(items=items, seed_jaccard=float(np.mean(jac)),
                                    label=float(np.mean([x["label"] for x in items])),
                                    train=float(np.mean([x["train"] for x in items])),
                                    held=float(np.mean([x["held_out"] for x in items])))
        e["between"] = {}
        for k in (5, 20):
            v = []
            for s in range(5):
                A = {x["title"] for x in rec["DeepWalk"][s]["lists"][t][:k]}
                B = {x["title"] for x in rec["node2vec"][s]["lists"][t][:k]}
                v.append(len(A & B) / len(A | B))
            e["between"][str(k)] = float(np.mean(v))
        out[t] = e
    return out


def selection():
    t = json.loads((RESULTS / "tuning.json").read_text())
    out = {}
    for g, v in t.items():
        c = v["cells"]
        names = sorted(c)
        val = np.array([c[n]["val"]["mean"] for n in names])
        test = np.array([c[n]["test_not_used"]["mean"] for n in names])
        lift = np.array([c[n]["lift_not_used"]["mean"] for n in names])
        P, Q = np.array([c[n]["p"] for n in names]), np.array([c[n]["q"] for n in names])
        out[g] = dict(selected=v["config"], val_best=float(val.max()), val_uniform=c["n2v_p1_q1"]["val"]["mean"], val_sd_selected=c[v["config"]]["val"]["sd"],
                      val_sd_median=float(np.median([c[n]["val"]["sd"] for n in names])), val_range=float(val.max() - val.min()),
                      spearman_val_test=float(spearmanr(val, test).statistic), spearman_val_lift=float(spearmanr(val, lift).statistic),
                      oracle_test=names[int(test.argmax())], oracle_test_score=float(test.max()), selected_test_tuneseeds=c[v["config"]]["test_not_used"]["mean"],
                      uniform_test_tuneseeds=c["n2v_p1_q1"]["test_not_used"]["mean"],
                      q_lt1_mean=float(val[Q < 1].mean()), q_gt1_mean=float(val[Q > 1].mean()), p_lt1_mean=float(val[P < 1].mean()), p_gt1_mean=float(val[P > 1].mean()))
    return out


def subsampling():
    """Word2vec keeps a token with probability min(1, (sqrt(f/s) + 1) * s/f). Visit frequency follows degree in a uniform walk."""
    out = {}
    for g in ("small", "large"):
        G, sp = load(g), load_split(g)
        d = np.asarray(adj(G.n, sp["train"]).sum(1)).ravel().astype(float)
        f = d / d.sum()
        s = SHARED["sample"]
        keep = np.minimum(1, (np.sqrt(f / s) + 1) * s / f)
        out[g] = dict(n=G.n, kept_share=float((f * keep).sum()), nodes_downsampled=float((keep < 1).mean()), tokens=G.n * SHARED["r"] * SHARED["l"],
                      effective_tokens=float((f * keep).sum() * G.n * SHARED["r"] * SHARED["l"]), max_f=float(f.max()))
    out["token_ratio"] = out["large"]["tokens"] / out["small"]["tokens"]
    out["effective_ratio"] = out["large"]["effective_tokens"] / out["small"]["effective_tokens"]
    return out


def main():
    out = dict(lists={g: lists(g) for g in ("small", "large")}, selection=selection(), subsampling=subsampling())
    (RESULTS / "understand.json").write_text(json.dumps(out, indent=1))
    for g in out["selection"]:
        print(g, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out["selection"][g].items()})
    print({k: v for k, v in out["subsampling"].items()})
    for g in ("small", "large"):
        for t, e in out["lists"][g].items():
            print(g, t, {a: {k: (round(e[a][k]["seed_jaccard"], 2), round(e[a][k]["label"], 2), round(e[a][k]["train"], 2)) for k in ("5", "20")} for a in ALGOS}, {k: round(v, 2) for k, v in e["between"].items()})
            print("   DW:", [x["title"] for x in e["DeepWalk"]["5"]["items"]])
            print("   N2V:", [x["title"] for x in e["node2vec"]["5"]["items"]])


if __name__ == "__main__":
    main()
