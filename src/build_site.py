"""Build public/index.html, the web write-up, from results/*.json. No number is typed here.

    python -m src.build_site    # then: npx vercel --prod

Same pattern as HW3 and HW4: one self-contained page, data embedded as JSON and drawn by the inline script, findings
first, no course framing (the coursework version is report/report.html). The page is public, so it carries no GTID, no
handout text and no raw graph data beyond article titles in the recommendation lists.
"""
import json

from config import DATA, QUERIES, RESULTS, ROOT, SHARED

OUT = ROOT / "public" / "index.html"
TEMPLATE = ROOT / "src" / "site_template.html"
GRAPHS = ("small", "large", "fb_small", "fb_large")


def J(name):
    return json.loads((RESULTS / name).read_text())


S, U, TUN, X = J("summary.json"), J("understand.json"), J("tuning.json"), J("part2_cross.json")
P2 = {g: J(f"part2_{g}.json") for g in ("small", "large")}
PROV = json.loads((DATA / "provenance.json").read_text())


def quality(g):
    out = {}
    for c, splits in S["quality"][g].items():
        e = splits["0"]
        out[c] = dict(n=e["n_seeds"], sd=e["test"]["20"]["ndcg"]["sd"], lsd=e["label"]["20"]["lift"]["sd"], base=e["base"],
                      k={k: dict(p=e["test"][k]["p"]["mean"], cap=e["test"][k]["cap"]["mean"], r=e["test"][k]["r"]["mean"],
                                 ndcg=e["test"][k]["ndcg"]["mean"], hit=e["test"][k]["hit"]["mean"], lift=e["label"][k]["lift"]["mean"])
                         for k in e["test"]},
                      splits={sp: v["test"]["20"]["ndcg"]["mean"] for sp, v in splits.items()})
    return out


def runs():
    n, secs = 0, 0.0
    for p in RESULTS.glob("run_*.json"):
        if "run_dev" in p.name or "run_timing" in p.name:
            continue
        n += 1
        secs += json.loads(p.read_text())["timings"]["total_s"]
    return n, secs / 3600


def pair(g, a, b):
    """A paired comparison as a minus b; summary.json stores each pair in one direction only."""
    if f"{g}:ndcg20:{a}-{b}" in S["pairs"]:
        p = S["pairs"][f"{g}:ndcg20:{a}-{b}"]
        return dict(diff=p["diff"], ci=p["ci"], seed_t=p["seed_t"], n=p["n"])
    p = S["pairs"][f"{g}:ndcg20:{b}-{a}"]
    return dict(diff=-p["diff"], ci=[-p["ci"][1], -p["ci"][0]], seed_t=-p["seed_t"], n=p["n"])


def data():
    t = TUN["large"]["config"]
    n, hours = runs()
    pairs = {f"{g}:{a}-{b}": pair(g, a, b) for g in ("small", "large") for a, b in
             (("dw_ns", "n2v_ns"), ("dw_hs", "n2v_hs"), ("n2v_hs", "n2v_ns"), ("n2v_hsns", "n2v_ns"), (t, "n2v_ns"), ("dw_hsns", t))}
    return dict(
        tuned=t, tp=TUN["large"]["p"], tq=TUN["large"]["q"], queries=list(QUERIES), shared=SHARED, n_runs=n, core_hours=hours,
        n_timing=len(list(RESULTS.glob("run_timing_*.json"))),
        prov={g: {k: PROV[g][k] for k in ("nodes", "edges", "max_degree", "labelled", "rule")} for g in GRAPHS},
        degrees=PROV["query_degrees"],
        q={g: quality(g) for g in GRAPHS}, pairs=pairs, timing={g: S["timing"][g] for g in ("small", "large")},
        jaccard={g: S["jaccard"][g] for g in ("small", "large")},
        cells={g: {c: dict(p=v["p"], q=v["q"], val=v["val"]["mean"], sd=v["val"]["sd"], test=v["test_not_used"]["mean"], lift=v["lift_not_used"]["mean"])
                   for c, v in TUN[g]["cells"].items()} for g in ("small", "large")},
        selection=U["selection"], subsampling=U["subsampling"], lists=U["lists"],
        hubs={g: P2[g]["hubs"] for g in P2}, strata={g: P2[g]["strata"] for g in P2}, distance={g: P2[g]["distance"]["dw_hsns"] for g in P2},
        cross=dict(metrics=X["configs"]["dw_hsns"]["metrics"], examples=X["configs"]["dw_hsns"]["examples"], seedj=X["configs"]["dw_hsns"]["seed_jaccard20_small"],
                   physics_share=X["physics_edge_share_of_degree"], base=X["base_small"]),
    )


def main():
    payload = json.dumps(data(), separators=(",", ":")).replace("</", "<\\/")
    page = TEMPLATE.read_text().replace("/*__DATA__*/null", payload)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(page)
    print(f"{OUT.relative_to(ROOT)} {len(page) // 1024} KB")


if __name__ == "__main__":
    main()
