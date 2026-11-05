"""Build report/report.html from the result files (the only source of numbers).

    python -m src.stats && python -m src.understand && python -m src.ai_rounds && python -m src.build_report

Same shape as HW3 (graded "Top submission") and HW4: cover + At a glance + glossary + tool table, Questions 1-8 (handout D1-D8), Part 2
"Understanding the results", references, Appendices A (prompts), B (extended tables), C (compute). The 4b critiques are drafted by the
assistant; the student reads them for voice.
"""
import json
from pathlib import Path

from config import DATA, KS_REQUIRED, QUERIES, REPORT, RESULTS, ROOT, SHARED
from src import figures as F
from src import report_lib as R
from src.prompts import PROMPTS, ROUND_INSTRUCTIONS

S = json.loads((RESULTS / "summary.json").read_text())
U = json.loads((RESULTS / "understand.json").read_text())
TUN = json.loads((RESULTS / "tuning.json").read_text())
P2 = {g: json.loads((RESULTS / f"part2_{g}.json").read_text()) for g in ("small", "large")}
X = json.loads((RESULTS / "part2_cross.json").read_text())
PROV = json.loads((DATA / "provenance.json").read_text())
T = TUN["large"]["config"]
assert T == TUN["small"]["config"]
TPQ = f"p={TUN['large']['p']:g}, q={TUN['large']['q']:g}"
CSS_EXTRA = """
pre.brief { white-space: pre-wrap; font-size: 6.6pt; line-height: 1.25; background: #f6f6f4; padding: 6px 8px; }
table.lists td { font-size: 7.6pt; line-height: 1.25; }
sup { font-size: 5.6pt; color: #555; }
.step2 { font-size: 9.4pt; }
.ok { font-weight: 600; }
"""


def Q(g, c, k="20", m="ndcg", part="test", sp="0"):
    return S["quality"][g][c][sp][part][k][m]["mean"]


def Qsd(g, c, k="20", m="ndcg", sp="0"):
    return S["quality"][g][c][sp]["test"][k][m]["sd"]


def L(g, c, k="20", sp="0"):
    return S["quality"][g][c][sp]["label"][k]["lift"]["mean"]


def Lsd(g, c, k="20"):
    return S["quality"][g][c]["0"]["label"][k]["lift"]["sd"]


def PR(g, a, b, key="ndcg20"):
    return S["pairs"][f"{g}:{key}:{a}-{b}"]


def TM(g, c, key="total_s", w="1"):
    return S["timing"][g][c][w][key]["mean"]


def pm(g, c, f=3, k="20", m="ndcg"):
    return f"{Q(g, c, k, m):.{f}f} ± {Qsd(g, c, k, m):.{f}f}"


def pml(g, c, k="20"):
    return f"{L(g, c, k):.2f} ± {Lsd(g, c, k):.2f}"


def ci(p):
    return f"{p['diff']:+.4f} [{p['ci'][0]:+.4f}, {p['ci'][1]:+.4f}]"


def pv(p):
    return "&lt;0.0001" if p < 1e-4 else f"{p:.2g}"


TABLES = [0]


def table(head, rows, caption, num_cols=(), cls=""):
    TABLES[0] += 1
    return R.table(head, rows, f"Table {TABLES[0]}. {caption}", num_cols, cls)


FIGS = [0]


def fig(svg, caption):
    FIGS[0] += 1
    return f"<figure>{svg}<figcaption>Figure {FIGS[0]}. {caption}</figcaption></figure>"


def quote(pid):
    p = next(x for x in PROMPTS if x[0] == pid)
    return f"<blockquote>[{p[0]}] {R.esc(p[3])}</blockquote>"


def rquote(i):
    r = ROUND_INSTRUCTIONS[i]
    return f"<blockquote>[{r[0]}, {r[1]}] {R.esc(r[2])}</blockquote>"


def brief(start, end):
    """Verbatim lines of PROMPT.md between two markers."""
    t = (ROOT / "PROMPT.md").read_text()
    a = t.index(start)
    b = t.index(end, a) if end else len(t)
    return f"<blockquote>[PROMPT.md, the brief P1 produced] {R.esc(t[a:b].strip())}</blockquote>"


def step2(prompts, rounds, a, b):
    rows = "".join(f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td></tr>" for r in rounds)
    return f"""<h3>Step 2: AI interaction record</h3><div class="step2">
<p><b>1) Tool:</b> Table 1. <b>2) Prompts (exact):</b></p>{prompts}
<p><b>3) Workflow:</b></p>
<table><thead><tr><th>Round</th><th>Context supplied</th><th>Query</th><th>Outcome</th></tr></thead><tbody>{rows}</tbody></table>
<p><b>4a) Strategy.</b> {a}</p>
<p><b>4b) Critique.</b> <i>Requirements met:</i> {b[0]} <i>Overlooked, oversimplified or miscalculated:</i> {b[1]} <i>Refined synthesis:</i> {b[2]}</p></div>"""


def lst(items):
    out = []
    for x in items:
        t = R.esc(x["title"].replace("_", " "))
        mark = "●" if x["train"] else ("✓" if x["held_out"] else "")
        name = f"<b>{t}</b>" if x["label"] else t
        out.append(f"{name}{mark}<sup>{x['seeds']}</sup>")
    return ", ".join(out)


def list_rows(k):
    rows = []
    for g in ("small", "large"):
        for t in QUERIES:
            e = U["lists"][g][t]
            for a in ("DeepWalk", "node2vec"):
                rows.append([g, R.esc(t.replace("_", " ")), a, lst(e[a][k]["items"])])
    return rows


def q_rows():
    rows = []
    for g in ("small", "large"):
        for t in QUERIES:
            e = U["lists"][g][t]
            d = PROV["query_degrees"][t][f"{g}_degree"]
            rows.append([g, R.esc(t.replace("_", " ")), d] + [f"{e[a][k]['label']:.2f}" for a in ("DeepWalk", "node2vec") for k in ("5", "20")]
                        + [f"{e[a][k]['seed_jaccard']:.2f}" for a in ("DeepWalk", "node2vec") for k in ("5",)] + [f"{e['between'][k]:.2f}" for k in ("5", "20")])
    return rows


def compute():
    ms = {}
    for p in RESULTS.glob("run_*.json"):
        if "run_dev" in p.name or "run_timing" in p.name:
            continue
        r = json.loads(p.read_text())
        a = ms.setdefault(r["machine"], [0, 0.0])
        a[0] += 1
        a[1] += r["timings"]["total_s"]
    return ms


MS = compute()
NRUNS = sum(v[0] for v in MS.values())
CORE_H = sum(v[1] for v in MS.values()) / 3600
SEL = U["selection"]
SUB = U["subsampling"]
HB = {g: P2[g]["hubs"] for g in P2}
ST = {g: P2[g]["strata"] for g in P2}
DI = {g: P2[g]["distance"] for g in P2}
XC = X["configs"]
HAS_NEG = "n2v_ns+negative=15" in S["quality"]["large"]
E_RATIO = PROV["large"]["edges"] / PROV["small"]["edges"]
N_RATIO = PROV["large"]["nodes"] / PROV["small"]["nodes"]
D_RATIO = PROV["large"]["sum_deg2"] / PROV["small"]["sum_deg2"]
BASE_ROWS = ("base_aa", "base_cn", "base_popularity", "base_random")
NAME = {"dw_hsns": "DeepWalk (as written, HS+NS)", "n2v_tuned": f"node2vec tuned ({TPQ}, NS)", "base_aa": "Adamic-Adar", "base_cn": "Common neighbours",
        "base_popularity": "Popularity", "base_random": "Random", "n2v_ns": "node2vec p=q=1, NS", "n2v_hs": "node2vec p=q=1, HS", "n2v_hsns": "node2vec p=q=1, HS+NS",
        "dw_ns": "DeepWalk walks, NS", "dw_hs": "DeepWalk walks, HS", "dw_repo": "DeepWalk repo defaults", "n2v_repo": "node2vec repo defaults"}
DW0 = "dw_hsns"
ROUNDS_ALL = {
    "P1": ("P1", "Handout PDF, HW4 PROMPT.md and setup", "P1", "PROMPT.md: tiers, design, vetoes"),
    "P2": ("P2", "PROMPT.md, HW3 and HW4 builders, rubric skill", "P2", "Repo smoke test, PLAN.md, stop for approval"),
    "P3": ("P3", "PLAN.md, the smoke-test findings", "P3", "Plan approved; repos cleared to run; VPN on"),
    "B": ("Build", "Approved plan, acceptance checks", "P2 continued", "Code, graphs, 512 grid runs"),
    "R1": ("R1", "Required rows of summary.json only", "R1", "Claims C1 to C9"),
    "R2": ("R2", "All of results/", "R2", "Five of nine claims fail; the other four need rewording"),
}


def R_(*ids):
    return [ROUNDS_ALL[i] for i in ids]


def cover():
    return f"""<div class="cover">
<p><b>COURSE TITLE:</b> CS6220 Big Data Systems and Analytics, Fall 2026, cs6220-A</p>
<p><b>HW5:</b> Programming [X] Reading Critique [ ]</p>
<p><b>Student Name:</b> Atharva Mohite</p>
<p><b>Student ID:</b> {R.SID_SLOT}</p>
<p>Code, run records and logs: in the submitted zip, <code>HW5_P_Mohite_Atharva.zip</code>.</p></div>
<h1>Graph embedding for node recommendation: DeepWalk vs node2vec</h1>
<p><b>At a glance.</b> With one shared configuration, DeepWalk and node2vec with p=q=1 tie on the large graph (difference {PR('large','dw_ns','n2v_ns')['diff']:+.4f} on NDCG@20, Table 10).
The gap you see between the two repos ({Q('large',DW0):.3f} vs {Q('large','n2v_tuned'):.3f}) comes mostly from the loss. DeepWalk's code trains hierarchical softmax and negative sampling together, and node2vec's default trains negative sampling only.
Tuning p and q adds {PR('large',T,'n2v_ns')['diff']:+.4f} on the large graph and nothing measurable on the small one. Adamic-Adar, a one-line heuristic, beats every embedding on held-out links under cosine distance on both graphs ({Q('small','base_aa'):.3f} and {Q('large','base_aa'):.3f}); embeddings beat it on label lift (Figure 1).
All of this replicates in shape on ego-Facebook, except that on its small graph the heuristics win on both measures (Part 2, S10).</p>
<p><b>Limitations.</b> I could not read the Towards Data Science blog: it returns HTTP 403 to automated fetches, so Question 1 rests on the node2vec paper. Recommendation quality is my definition, since the handout gives none (Question 5). Timing ran on 8 cores of one CPU model where other users' jobs may share the node, not on an exclusive node, which stayed pending for nearly 2 hours. Quality runs used two CPU models ({', '.join(str(v[0]) for v in MS.values())} runs). Lift is not comparable between the two graphs because their label schemes differ (S5). Sweeps and hub tests use 3 seeds, other rows 5. The Euclidean result in S9 was found after looking at test scores and is a lead, not a selected result.</p>
<p><b>Handout anomalies, recorded rather than resolved silently.</b> Step 2 says "either DeepWalk or Node2Vec", while 4a and 4c say "each of the two algorithms", and 4e makes the comparison a bonus: I ran both. 4e says "the above two learning tasks", but step 4 describes one: I read it as the 4a queries plus the 4b runtime and quality results. "Recommendation quality" is undefined. There is no naming line, so the filename follows the course convention. The handout says posted Oct 26, while Canvas shows Oct 16; the course calendar records both.</p>
<p><b>Glossary.</b> <i>Skip-gram:</i> word2vec's model, here predicting nearby nodes in a walk. <i>HS (hierarchical softmax) and NS (negative sampling):</i> two approximations of its loss. <i>p, q:</i> node2vec's walk biases; low p favours returning, low q favours moving outward. <i>NDCG@K:</i> rank-weighted share of held-out neighbours found in the top K, where 1 is a perfect order. <i>Lift:</i> how many times more often the top K shares a label than a random node would. <i>Adamic-Adar, common neighbours:</i> count shared neighbours, Adamic-Adar weighting rare ones higher. <i>A/A check:</i> comparing two things that should be equal, to measure the noise and catch bugs.</p>
""" + table(["Tool", "Role", "Used for", "URL"], [["Claude Code (Claude Opus 5.5)", "Assistant", "Brief, plan, code, analysis rounds, report drafts, 4b drafts", "https://claude.ai/code"]], "Step 2.1 for all questions: tool used.")


def q1():
    body = f"""<h2 style="break-before: page">Question 1: Read the comparison of DeepWalk and node2vec</h2>
<h3>Step 1: Final answer</h3>
<p>Both learn node vectors by feeding random walks to skip-gram. They differ in the walk: DeepWalk steps uniformly, while node2vec biases each step by where the walk just came from. The paper calls DeepWalk's sampling "a special case of node2vec with p = 1 and q = 1" (Grover and Leskovec 2016, Section 4.2), and low q pushes the walk outward while low p keeps it close.
The blog could not be read (HTTP 403), so this answer takes nothing from it.</p>
""" + table(["Paper's claim or setup", "What it predicts here", "What happened"], [
        ["node2vec gains over DeepWalk: Macro-F1 up to 22.3% (Section 4.3, its Table 2); link-prediction AUC up to 3.8% over DeepWalk (Section 4.7)", "Tuned node2vec beats DeepWalk", f"Not as the repos ship. Same loss: tie or {PR('large',T,'n2v_ns')['diff']:+.4f} (Table 10)"],
        ["\"DeepWalk uses hierarchical sampling\"; the authors switch it to NS, since \"hierarchical softmax is inefficient\" (Section 4.2)", "NS is the fair common loss and cheaper", f"HS is both faster and better here ({TM('large','dw_hs'):.0f} s vs {TM('large','dw_ns'):.0f} s, Table 7)"],
        ["Embeddings beat heuristics on link prediction: AUC 0.968 vs 0.888 on Facebook (Section 4.7)", "Embeddings beat Adamic-Adar", f"The reverse, under direct ranking. The paper trains a classifier on edge features and scores AUC; we rank by distance (S3)"],
        ["Settings d=128, r=10, l=80, k=10, one epoch; p, q chosen from {0.25, 0.5, 1, 2, 4}", "Use these as the shared config", f"Used, except 5 epochs. One epoch scores far lower on the large graph ({Q('large','n2v_ns+epochs=1'):.3f}, S7)"],
    ], "What the paper predicts for our graphs, and the outcome.")
    return body + step2(quote("P1") + quote("P2"), R_("P1", "P2", "R1", "R2"),
                        "I had the AI turn the paper into testable predictions before any run, so each became a row in a table I could check. When the blog fetch failed I had it report the failure instead of describing the article from memory, which keeps Question 1 honest about its one source.",
                        ("the difference between the algorithms and the prediction are stated, and the paper's parameters were read from its text, not recalled.",
                         f"the first brief said DeepWalk \"hard-codes sg=1, hs=1 (hierarchical softmax)\". The code at <code>deepwalk/__main__.py:75</code> leaves gensim's <code>negative=5</code> on, and <code>word2vec_inner.pyx</code> then runs both losses. The plan also called the shared config \"the paper's\" although the paper trains one epoch. The handout's blog remains unread.",
                         "the paper's advantage rests on a classifier, AUC and a corrected loss, and none of those hold when you run the two repos with distance ranking. The paper is right that the walk is the only algorithmic difference. It does not follow that the repos differ only in the walk."))


def q2():
    defaults = [["Dimension d", 64, 128], ["Walk length l", 40, 80], ["Walks per node r", 10, 10], ["Window w", 5, "5 (gensim default)"], ["Loss as called", "HS+NS (<code>hs=1</code>, <code>negative=5</code>)", "NS"],
                ["Walk engine", "uniform, Python", "second-order, Python, precomputed table"], [f"Repo-default NDCG@20, large", pm("large", "dw_repo"), pm("large", "n2v_repo")], [f"Repo-default NDCG@20, small", pm("small", "dw_repo"), pm("small", "n2v_repo")]]
    body = f"""<h2>Question 2: Download and evaluate DeepWalk and node2vec</h2>
<h3>Step 1: Final answer</h3>
<p>I used both handout repos, pinned at <code>phanein/deepwalk</code> 6e6dff2 and <code>eliorc/node2vec</code> 872af47 and vendored with a small reviewable patch (<code>vendor/patch.diff</code>). Neither imports as shipped on Python 3.11 with gensim 4.3.3: DeepWalk fails on <code>collections.Iterable</code> and <code>Mapping</code>, and node2vec fails on missing package metadata when vendored. The patch fixes those imports, renames <code>size=</code> to <code>vector_size=</code> and passes a seed to word2vec, which DeepWalk's CLI never did.</p>
""" + table(["", "DeepWalk", "node2vec"], defaults, "The two repos differ in more than the algorithm.") + f"""
<p>Comparing repo defaults would credit DeepWalk with a {Q('large','dw_repo')-Q('large','n2v_repo'):.3f} win on the large graph. Moving from NS to HS+NS accounts for {PR('large','n2v_hsns','n2v_ns')['diff']:.3f} at the shared settings, and the rest comes from the other defaults; Question 8 and S7 separate the effects. Reading the code turned up three more traps. DeepWalk's <code>if start:</code> (<code>graph.py:130</code>) treats node 0 as "no start", so I number nodes from 1. eliorc's <code>fit()</code> inherits <code>min_count=5</code> and <code>window=5</code> unless passed. And its unweighted path raises and catches an exception per neighbour during the precompute, which inflates that phase. Same seed gives the same vectors with one worker, for both repos (acceptance checks, Appendix C), and the one row I compared across machines gave the same score on the Mac and on PACE.</p>"""
    return body + step2(quote("P2") + quote("P3"), R_("P2", "P3", "B"),
                        "The brief required a smoke test before the plan, because that result could change the design, and it did. When a permission check blocked running the cloned code, I had the AI first audit both repos by reading them and grepping for network, subprocess or exec calls, then asked me for an explicit decision. The patch stays a diff so a reader can review it.",
                        ("both repos ran at pinned commits with the smallest patch, disclosed here.",
                         "the brief said <code>from gensim.models.word2vec import Vocab</code> breaks on gensim 4. It doesn't: <code>word2vec.Vocab</code> still exists in 4.3.3. The real import failures are <code>Mapping</code> and <code>Iterable</code>. The brief named only <code>Mapping</code> and missed the node2vec metadata failure entirely. More seriously, it read <code>hs=1</code> as pure HS, which would have given the wrong control design.",
                         "the fair evaluation of these repos treats loss, defaults and walk as three separate variables, because DeepWalk as written is a hybrid of two losses that neither its paper nor its flags describe."))


def q3():
    rows = []
    for g, role, lab in (("small", "Small", "5 physics subcategories"), ("large", "Large", "15 top-level subjects"), ("fb_small", "Replication small", "circles"), ("fb_large", "Replication large", "circles (ego/circle)")):
        p = PROV[g]
        rows.append([role, R.esc(p["rule"]), f"{p['nodes']:,}", f"{p['edges']:,}", p["max_degree"], lab, f"{p['labelled']:,}"])
    body = f"""<h2>Question 3: Two graphs from the same entities</h2>
<h3>Step 1: Final answer</h3>
<p>Both main graphs are Wikispeedia, from SNAP: Wikipedia articles as nodes, hyperlinks as edges. The small graph is nested inside the large one, which meets "the same set of entities" literally and lets me run the same five query articles on both. Article names also let a reader judge a recommendation by eye.</p>
""" + table(["Role", "Rule", "Nodes", "Edges", "Max degree", "Labels", "Labelled"], rows, "Graphs as built (data/provenance.json). Sizes fall in the handout's bands, which the build asserts.", (2, 3, 4, 6)) + f"""
<p><b>Rules.</b> Links are directed, so I symmetrise them; I drop {PROV['large']['self_links']} self-links and URL-decode titles. The large graph keeps its largest component, dropping {len(PROV['large']['dropped_nodes'])} articles. Articles with no category stay in the graph and leave the label measure only; just <code>{PROV['large']['no_category_kept'][0]}</code> is affected. "Relevant" means sharing any label, because 598 articles have several categories. For ego-Facebook, SNAP's <code>0.edges</code> omits the ego node, so I use the largest component of what it lists ({PROV['fb_small']['nodes_before_lcc']} listed, {PROV['fb_small']['nodes']} kept).</p>"""
    return body + step2(brief("**Datasets. I checked", "**Design:**"), R_("P1", "P3", "B"),
                        "The brief listed the known data traps before any code, and <code>src/data.py</code> turns the size bands and the query rule into assertions. That way a wrong build fails loudly and doesn't appear as a strange result later.",
                        ("two nested graphs in the required size bands, with the same entities and citable labels.",
                         f"the brief's table gave the large graph's 106,537 edges before taking the largest component, beside its 4,589 nodes. The graph used has {PROV['large']['nodes']:,} and {PROV['large']['edges']:,}. It also listed ego net 0 as 333 nodes and 2,519 edges, while the graph used has {PROV['fb_small']['nodes']} and {PROV['fb_small']['edges']:,}, which I rebuilt from the data. It also implied one set of labels; the two graphs use different schemes, so lift isn't comparable across them.",
                         "the graphs are right for the handout, with one caveat that carries through the report: hyperlinks are chosen by editors. A link is partly a judgement about what is worth linking, so \"held-out link\" and \"good recommendation\" are not the same thing."))


def q4():
    cases = [U["lists"][g][t] for g in ("small", "large") for t in QUERIES]
    BETW = sum(e["between"]["5"] for e in cases) / len(cases)
    SEEDJ = sum(e[a]["5"]["seed_jaccard"] for e in cases for a in ("DeepWalk", "node2vec")) / (2 * len(cases))
    head = ["Graph", "Query", "Degree", "DW lab@5", "DW lab@20", "n2v lab@5", "n2v lab@20", "DW seed J@5", "n2v seed J@5", "DW vs n2v J@5", "J@20"]
    body = f"""<h2>Question 4: Recommendation by embedding distance, five queries</h2>
<h3>Step 1: Final answer</h3>
<p>For a query node, I rank all other nodes by cosine similarity of their vectors and return the top K. The five queries come from a rule fixed before any embedding existed: sort the small graph by degree, split into five equal bins, draw one per bin with a fixed seed. Nothing was swapped afterwards. That gives 5 queries × 2 algorithms × 2 graphs = 20 query runs, in <code>src/recommend.py</code>, the one function evaluation and report share. The vectors come from training edges only. "DeepWalk" means the repo's loss as written, and "node2vec" means tuned p and q. Question 6 lists the articles.</p>
""" + table(head, q_rows(), "The 20 query runs. lab@K: share of the seed-0 list sharing a label with the query. Seed J: Jaccard overlap between seeds. DW vs n2v J: overlap between algorithms, mean over seeds.", (2, 3, 4, 5, 6, 7, 8, 9, 10)) + f"""
<p>Two things stand out. A single list is weak evidence: top-5 lists overlap only about half between seeds, so most individual differences between algorithms are within seed noise. Across these 10 cases the two algorithms overlap {BETW:.2f} with each other at K=5, against {SEEDJ:.2f} for one algorithm with itself across seeds, so most of the difference between their lists is the size of ordinary seed noise (S4). And <code>Integrated_circuit</code> is the small graph's weak query: only {U['lists']['small']['Integrated_circuit']['DeepWalk']['5']['label']:.0%} of its top 5 share a subcategory, because the articles it really relates to sit outside physics (S5).</p>"""
    return body + step2(brief("6. **Query nodes by a rule", "7. **Runtime"), R_("P1", "P2", "B"),
                        "Fixing the query rule in the brief, before any vectors, removes the temptation to pick five articles whose lists look good. Having one ranking function in one file means the tables in this report can't disagree with the metrics.",
                        ("20 runs on 5 distinct nodes, both algorithms, both graphs.",
                         f"the quintile rule draws no true hub: the highest query has degree {PROV['query_degrees']['Special_relativity']['small_degree']} on the small graph, while <code>Sun</code> has {PROV['small']['max_degree']}. The rule was followed anyway, and hubs are covered by S8. The rule also ignores the large graph, where <code>Andromeda_Galaxy</code> has degree only {PROV['query_degrees']['Andromeda_Galaxy']['large_degree']}.",
                         "these five queries are examples, and the conclusions come from the all-node metrics in Question 5. Reading algorithm differences off five lists would mostly be reading seed noise."))


def q5():
    rows = []
    for g in ("small", "large"):
        for c in (DW0, "n2v_tuned") + BASE_ROWS:
            rows.append([g, NAME[c], pm(g, c), pm(g, c, 3, "5"), f"{Q(g, c, '20', 'p'):.3f}", f"{Q(g, c, '20', 'r'):.3f}", f"{Q(g, c, '20', 'hit'):.3f}", pml(g, c), pml(g, c, "5")])
    tr = []
    for g in ("small", "large"):
        for c, n in (("dw_hsns", "DeepWalk as written"), (T, "node2vec tuned"), ("dw_ns", "DeepWalk walks, NS"), ("dw_hs", "DeepWalk walks, HS"), ("n2v_ns", "node2vec p=q=1, NS"), ("base_aa", "Adamic-Adar")):
            x = S["timing"][g][c]["1"]
            tr.append([g, n, f"{x['precompute_s']['mean']:.1f}", f"{x['walk_s']['mean']:.1f}", f"{x['train_s']['mean']:.1f}", f"{x['total_s']['mean']:.1f} ± {x['total_s']['sd']:.1f}", f"{x['latency_mean_ms']['mean']:.2f}", f"{x['latency_p95_ms']['mean']:.2f}", f"{x['peak_rss_mb']['mean']:.0f}"])
    body = f"""<h2>Question 5: Runtime and recommendation quality</h2>
<h3>Step 1: Final answer</h3>
<p><b>Quality is my definition,</b> fixed before results, with two independent measures over all eligible nodes. <i>Held-out links:</i> a seeded spanning tree is protected, then 20% test and 10% validation edges are removed before any walk. Candidates exclude the query and known edges, and metrics run over the {S['quality']['small'][DW0]['0']['n_test']} and {S['quality']['large'][DW0]['0']['n_test']:,} nodes with a held-out edge. <i>Label agreement:</i> the share of the top K sharing a label, as lift over each node's base rate ({S['quality']['small'][DW0]['0']['base']:.2f} small, {S['quality']['large'][DW0]['0']['base']:.2f} large), with only the query excluded. NDCG@20 is the primary metric.</p>
""" + table(["Graph", "Method", "NDCG@20", "NDCG@5", "P@20", "R@20", "Hit@20", "Lift@20", "Lift@5"], rows, "Quality on test, split 0, 5 seeds, mean ± sd across seeds.", (2, 3, 4, 5, 6, 7, 8)) + fig(F.tradeoff(), "The two quality measures disagree. Heuristics find more held-out links, and embeddings find more articles on the query's topic.") + table(["Graph", "Row", "Precompute s", "Walks s", "Word2vec s", "Total s", "Query ms", "p95 ms", "Peak MB"], tr, "Runtime, one worker, 8-core job on Xeon Gold 6226, 3 repeats. Query is one top-20 query by brute force. Baselines have no training to speak of.", (2, 3, 4, 5, 6, 7, 8)) + f"""
<p>Every row in Table {TABLES[0]} ran sequentially in one job. A query takes under a millisecond on either graph, so training dominates the cost. Adamic-Adar needs no training and scores higher on links, so an embedding has to justify {TM('large',DW0):.0f} s on something else, which here is topical lift. In some rows the mean query time exceeds its p95, which means a few very slow calls in 200, not a typical slowdown. S6 explains how the cost scales.</p>"""
    return body + step2(brief("1. **Quality has two", "2. **Baselines"), R_("P1", "P2", "B", "R1", "R2"),
                        "The handout doesn't define quality, so I had the AI define it before any result, with two measures that could disagree, plus baselines. That is what let me see the disagreement in Figure 1 at all. Lists that merely look sensible would have shown nothing. Timing runs one job at a time, so runs can't slow each other, and it sits apart from the contended quality grid.",
                        ("runtime broken into phases plus latency and memory; quality on two measures with variation shown.",
                         f"round 1 (C3) concluded embeddings aren't worth their cost for link recommendation. That holds only for cosine, and S9 and S8 show it is sensitive to metric and to hubs. C6 said tuned node2vec costs more than DeepWalk; measured, it costs less ({TM('large',T):.0f} s vs {TM('large',DW0):.0f} s) because DeepWalk as written pays for two losses. The plan's estimate of 10 to 15 minutes per large run and about 25 core-hours was also wrong: runs took 100 to 500 s and the whole {NRUNS}-run grid took {CORE_H:.0f} core-hours, and the brief's warning about precompute memory doesn't hold at this size (peak {S['timing']['large']['n2v_ns']['1']['peak_rss_mb']['mean']:.0f} MB).",
                         "on these graphs, ranking by Adamic-Adar is the better way to predict a hyperlink, and embeddings are the better way to find articles on the same topic. Which one wins depends on what \"quality\" means, so the handout's undefined term decides the answer."))


def q6():
    body = f"""<h2>Question 6: Top-K lists and visualisation</h2>
<h3>Step 1: Final answer</h3>
<p>Table {TABLES[0]+1} gives all 20 top-5 lists from seed 0, and Appendix B gives the 20 top-20 lists. <b>Bold</b> shares a label with the query, ● is a training neighbour, ✓ is a held-out neighbour, and the superscript counts how many of 5 seeds include that article in the same top K. These are the literal handout lists, so training neighbours are allowed.</p>
""" + table(["Graph", "Query", "Algorithm", "Top 5"], list_rows("5"), "All 20 top-5 lists (seed 0). Labels are physics subcategories on the small graph and top-level subjects on the large one.", cls="lists") + fig(F.query_panels(), "Top-20 for each query on the small graph, on one fixed layout. Recommendations cluster around the query. Open circles are close in the graph but outside the query's subcategory.") + f"""
<p>By eye most lists are reasonable, which is exactly why looking isn't enough. <code>HD_28185</code> and <code>Andromeda_Galaxy</code> differ most between the algorithms on the small graph (Jaccard {U['lists']['small']['HD_28185']['between']['5']:.2f} and {U['lists']['small']['Andromeda_Galaxy']['between']['5']:.2f}), but their superscripts show that much of each list isn't stable under the same algorithm either. Lists from the large graph tend to be better for the articles with links outside physics, as with <code>Integrated_circuit</code> (S5).</p>"""
    return body + step2(quote("P2"), R_("P1", "P2", "B"),
                        "The handout allows a figure or tables, and the brief asked for both. I had each list carry its evidence inline (label, edge status, seed count), so a reader can check any item against ground truth and its stability without leaving the table.",
                        ("tables for all query and algorithm pairs, and figures for the small graph.",
                         "the figure covers only the small graph because a 4,589-node layout would be unreadable. Showing seed 0 only would also overstate how settled a list is, so I added the seed count, which the first design didn't have. The list marks use training and held-out status from one split, so ● and ✓ depend on that split.",
                         "read a list together with its superscripts. An article in the top 5 for 5 of 5 seeds is a property of the graph, and one that appears for 1 of 5 seeds is noise."))


def q7():
    rows = []
    for g in ("small", "large"):
        for c in (DW0, "n2v_tuned", "base_aa"):
            rows.append([g, NAME[c]] + [f"{Q(g, c, k, 'p'):.3f}" for k in ("5", "20")] + [f"{Q(g, c, k, 'cap'):.3f}" for k in ("5", "20")] + [f"{Q(g, c, k, 'r'):.3f}" for k in ("5", "20")] + [f"{Q(g, c, k, 'hit'):.3f}" for k in ("5", "20")] + [f"{L(g, c, k):.2f}" for k in ("5", "20")])
    j = S["jaccard"]
    body = f"""<h2>Question 7: Results under K=5 and K=20</h2>
<h3>Step 1: Final answer</h3>
<p>As K grows from 5 to 20, precision falls and recall rises, as expected. The fall in precision is partly arithmetic: a node with 2 held-out neighbours can't exceed P@20 = 0.10, so raw precision is capped. Table {TABLES[0]+1} reports that ceiling, and it is what makes the two graphs comparable.</p>
""" + table(["Graph", "Method", "P@5", "P@20", "Cap@5", "Cap@20", "R@5", "R@20", "Hit@5", "Hit@20", "Lift@5", "Lift@20"], rows, "K=5 against K=20. Cap: highest possible precision given each node's held-out count.", (2, 3, 4, 5, 6, 7, 8, 9, 10, 11)) + fig(F.k_curves(), "Precision against its ceiling, and NDCG, across K. The ceiling falls quickly with K, so precision increasingly measures how many held-out neighbours nodes have.") + f"""
<p>On the small graph P@20 ({Q('small',DW0,'20','p'):.3f}) is nearly half its ceiling ({Q('small',DW0,'20','cap'):.3f}), against {Q('small',DW0,'5','p')/Q('small',DW0,'5','cap'):.0%} at K=5, so raw P@K makes K=20 look worse than it is. Lift falls with K for every method because the closest few articles are the most topical. And top-K overlap across seeds barely changes ({j['large'][DW0]['5']['mean']:.2f} at K=5, {j['large'][DW0]['20']['mean']:.2f} at K=20 on the large graph), so a larger K isn't a route to more stable lists. K=5 suits a short list shown to a person, while K=20 suits a candidate set for a later stage, where recall matters.</p>"""
    return body + step2(brief("8. **Two K settings", "9. **Distance"), R_("P1", "P2", "B"),
                        "The brief asked for K to be explained, not just tabulated, and named the precision cap in advance. I had the metric code report the cap with every K so it couldn't be left out later.",
                        ("two K settings for every list and metric, plus K=1, 10, 50 for the curve.",
                         "the first framing was that precision falls and recall rises with K. That is true but says nothing about these methods, because most of the fall is the ceiling dropping. NDCG, the normalised metric, is not monotonic in K either: it dips and then rises (Figure 3), so it doesn't simply reward large K. I can't fully explain the dip from K=1 to K=5.",
                         "use recall or NDCG to compare across K, and always read precision against its ceiling. Adamic-Adar leads on links and embeddings on lift at both K settings. Only the small-graph tie between DeepWalk and tuned node2vec changes order, within noise. So K changes the numbers and not the conclusion."))


def q8():
    rows = []
    for g in ("small", "large"):
        for name, a, b in (("Walk, same loss (A/A): NS", "dw_ns", "n2v_ns"), ("Walk, same loss (A/A): HS", "dw_hs", "n2v_hs"), ("Walk, same loss (A/A): HS+NS", "dw_hsns", "n2v_hsns"),
                           ("Loss: HS over NS", "n2v_hs", "n2v_ns"), ("Loss: HS+NS over NS", "n2v_hsns", "n2v_ns"), (f"Biased walk: tuned over p=q=1, NS", T, "n2v_ns"), ("Headline: DeepWalk over tuned", DW0, T)):
            try:
                p = PR(g, a, b)
            except KeyError:
                p = PR(g, b, a)
                p = dict(p, diff=-p["diff"], ci=[-p["ci"][1], -p["ci"][0]], seed_t=-p["seed_t"])
            rows.append([g, name, ci(p), pv(p["wilcoxon_p"]), f"{p['seed_t']:+.1f}"])
    body = f"""<h2>Question 8 (bonus): DeepWalk against node2vec on both graphs</h2>
<h3>Step 1: Final answer</h3>
<p>Run as shipped with their losses, DeepWalk beats node2vec on the large graph and ties on the small. That headline misleads. Once both use the same loss, the two codebases tie, and the DeepWalk advantage is a loss effect. Biased walks add a little on the large graph and nothing measurable on the small one.</p>
""" + table(["Graph", "Comparison", "Diff in NDCG@20 [node bootstrap 95% CI]", "Wilcoxon p", "Seed t"], rows, "Paired per-node comparisons on test, averaged over 5 seeds first. Seed t: the difference in units of its standard error across seeds, which the node CI leaves out.", (3, 4)) + f"""
<p><b>Inversions.</b> DeepWalk, the older and simpler method, beats tuned node2vec on the large graph ({Q('large',DW0):.3f} vs {Q('large','n2v_tuned'):.3f}), the opposite of the paper's ordering. The paper put both on NS, so its comparison has no loss gap. Cost also inverts: DeepWalk as written is the slowest common row after HS+NS node2vec ({TM('large',DW0):.0f} s vs {TM('large',T):.0f} s for tuned node2vec), since training two losses roughly doubles word2vec time. <b>Size:</b> the loss effect holds on both graphs, and on the large one it is about {PR('large','n2v_hs','n2v_ns')['diff']/PR('large',T,'n2v_ns')['diff']:.0f} times the walk effect. On the small graph one A/A row has a node CI that excludes zero ({ci(PR('small','dw_ns','n2v_ns'))}), but its seed t is {PR('small','dw_ns','n2v_ns')['seed_t']:+.1f} on 5 seeds, and the large graph ties (S4). <b>Recommendation:</b> if you must pick one of the two repos for link-style recommendation on such a graph, use the faster uniform walk with HS. Neither beats Adamic-Adar on links under cosine.</p>"""
    return body + step2(quote("P3") + brief("3. **Isolate the algorithm", "4. **Tune p and q"), R_("P1", "P3", "B", "R1", "R2"),
                        "The control I care most about is the p=q=1 row crossed with each loss. It turns \"which algorithm is better\" into three separate questions: walk, loss and codebase. Round 1 was limited on purpose to the required rows, so I could see what a report built on those rows would conclude and then test it against the controls in round 2. The assistant had already seen the control rows, so round 1 is a restricted reading rather than a blind one.",
                        ("both algorithms on both graphs, with paired tests and seed variation.",
                         f"round 1's C1 said uniform walks beat biased walks, with p = {PR('large',T,DW0)['wilcoxon_p']:.0e}. The statistic is real and the conclusion wrong: it confused loss with walk. C2 read that as a size effect. A tiny p here also reflects {PR('large',T,DW0)['n']:,} nodes, and it says nothing about what caused the gap. The node-level CI also misses seed noise, which is why I added seed t after the small A/A row looked significant.",
                         "the walk bias is a real but small effect on these graphs. The loss is larger, and the choice of metric or heuristic is larger still. The claim I would defend is narrower than \"DeepWalk beats node2vec\": gensim's HS beats its NS at these settings, and DeepWalk happens to ship with HS turned on."))


def part2():
    h = """<h2>Part 2. Understanding the results (beyond the handout)</h2>
<p>Each section asks one question, shows evidence and explains it. None of this is needed to answer the handout.</p>"""
    # S1
    r = [[n] + [f"{Q(g, c):.3f}" for g in ("small", "large")] + [f"{L(g, c):.2f}" for g in ("small", "large")] for c, n in (("dw_ns", "DeepWalk walks, NS"), ("n2v_ns", "node2vec p=q=1, NS"), ("n2v_tuned", "node2vec tuned, NS"), ("dw_hs", "DeepWalk walks, HS"), ("n2v_hs", "node2vec p=q=1, HS"), ("dw_hsns", "DeepWalk walks, HS+NS (as written)"), ("n2v_hsns", "node2vec p=q=1, HS+NS"), ("dw_repo", "DeepWalk repo defaults"), ("n2v_repo", "node2vec repo defaults"))]
    h += "<h3>S1. How much of the gap is the walk, the loss, or the codebase?</h3>" + table(["Row", "NDCG@20 small", "large", "Lift@20 small", "large"], r, "The control matrix on the shared config. Reading across rows with the same loss is the A/A check.", (1, 2, 3, 4)) + f"""
<p>On the large graph the codebase accounts for about {abs(PR('large','dw_ns','n2v_ns')['diff']):.4f}, the walk bias {PR('large',T,'n2v_ns')['diff']:.4f}, and moving from NS to HS {PR('large','n2v_hs','n2v_ns')['diff']:.4f}. Repo defaults widen the gap ({Q('large','dw_repo'):.3f} vs {Q('large','n2v_repo'):.3f}): moving to them costs NS node2vec {Q('large','n2v_ns')-Q('large','n2v_repo'):.3f} and DeepWalk's hybrid {Q('large','dw_hsns')-Q('large','dw_repo'):.3f}, though the two rows change different settings, so this can't be pinned on one of them (S7). Why HS beats NS isn't settled by this table. """
    if HAS_NEG:
        nr = [[n] + [f"{Q(g, c):.3f}" for g in ("small", "large")] + [f"{L(g, c):.2f}" for g in ("small", "large")] + [f"{S['quality']['large'][c]['0']['n_seeds']}"] for c, n in (("n2v_ns+negative=1", "NS, 1 negative"), ("n2v_ns", "NS, 5 negatives"), ("n2v_ns+negative=15", "NS, 15 negatives"), ("n2v_ns+negative=30", "NS, 30 negatives"), ("n2v_hs", "HS"))]
        best = max(("n2v_ns+negative=1", "n2v_ns+negative=15", "n2v_ns+negative=30"), key=lambda c: Q("large", c))
        h += "I tested one candidate, that 5 negatives are too few, directly:</p>" + table(["Loss", "NDCG@20 small", "large", "Lift@20 small", "large", "Seeds"], nr, "Is it the number of negative samples? Same walks, node2vec p=q=1.", (1, 2, 3, 4, 5)) + f"""<p>The best NS row on the large graph ({Q('large',best):.3f}) compares with HS at {Q('large','n2v_hs'):.3f}. {NEG_TEXT(best)}</p>"""
    else:
        h += "</p>"
    # S2
    h += "<h3>S2. Was tuning p and q worth it, and do BFS-like or DFS-like settings win?</h3>" + fig(F.heatmaps(), f"Validation NDCG@20 over the paper's grid. The selected cell is bold, {TPQ} on both graphs. Scores rise toward low q and high p.") + table(["Graph", "Val selected", "Val p=q=1", "Test, selection seeds", "Test, fresh seeds", "Test p=q=1, fresh", "Spearman val, test", "Spearman val, lift"],
        [[g, f"{SEL[g]['val_best']:.3f}", f"{SEL[g]['val_uniform']:.3f}", f"{SEL[g]['selected_test_tuneseeds']:.3f}", f"{Q(g,'n2v_tuned'):.3f}", f"{Q(g,'n2v_ns'):.3f}", f"{SEL[g]['spearman_val_test']:.2f}", f"{SEL[g]['spearman_val_lift']:.2f}"] for g in ("small", "large")], "Selection audit. Test scores on the selection seeds were never used to select; they measure how much the winner flatters itself.", (1, 2, 3, 4, 5, 6, 7)) + f"""
<p>Validation picked the same cell on both graphs, and validation tracks test well across the grid, so selection works. But the small graph shows the winner's curse. The selected cell led uniform on validation by {SEL['small']['val_best']-SEL['small']['val_uniform']:.3f}, scored {SEL['small']['selected_test_tuneseeds']:.3f} on test with its selection seeds, and {Q('small','n2v_tuned'):.3f} with fresh seeds, level with uniform. Low q (outward, DFS-like) beats high q (mean validation {SEL['large']['q_lt1_mean']:.3f} vs {SEL['large']['q_gt1_mean']:.3f} on the large graph), and high p beats low p ({SEL['large']['p_gt1_mean']:.3f} vs {SEL['large']['p_lt1_mean']:.3f}), so walks that don't double back do better. The gain is small compared with the {SEL['large']['val_range']:.3f} range across the whole grid, because uniform is already near the top. Tuning does nothing for lift (Spearman {SEL['large']['spearman_val_lift']:.2f}). So tuning cost 75 runs per graph and gave at most {PR('large',T,'n2v_ns')['diff']:+.4f}: not worth it here, compared with switching the loss.</p>"""
    # S3
    t3 = []
    for g in ("small", "large"):
        for c, n in (("base_aa", "Adamic-Adar"), ("base_cn", "Common neighbours"), ("dw_hsns", "DeepWalk"), ("dw_hs", "DeepWalk walks, HS"), (T, "node2vec tuned")):
            e = ST[g]["table"][c]
            t3.append([g, n] + [f"{x:.3f}" for x in e["ndcg_by_bin"]] + [f"{e['near_hits']:.0f}", f"{e['far_hits']:.1f}", f"{e['top_in_2hop']:.0%}"])
    h += "<h3>S3. Do embeddings beat common neighbours?</h3>" + table(["Graph", "Method", "NDCG@20, degree Q1", "Q2", "Q3", "Q4", "Hits within 2 hops", "Hits beyond", "Slots within 2 hops"], t3, f"Who wins where. Q1 to Q4 are quartiles of query degree. Beyond two hops, Adamic-Adar scores zero by construction. Test edges beyond 2 hops: {ST['small']['n_far']} of {ST['small']['n_test']} small, {ST['large']['n_far']:,} of {ST['large']['n_test']:,} large.", (2, 3, 4, 5, 6, 7, 8)) + f"""
<p><b>No, on links, in any degree quartile.</b> This inverts the paper's result, which is AUC from a classifier trained on edge features. Here I rank by distance directly and score the top K. The obvious argument for embeddings, that walks reach further than two hops, fails on this data: {ST['large']['two_hop_share']:.0%} of large-graph test edges are within two hops, embeddings spend {ST['large']['table']['dw_hsns']['top_in_2hop']:.0%} of their slots there anyway, and they find about {ST['large']['table']['dw_hsns']['far_hits']:.0f} of the {ST['large']['n_far']:,} farther edges in total. These are dense graphs with mean degree {HB['large']['mean_degree']:.1f}, and the local signal is already strong. Embeddings do win on lift. A likely reason is that heuristics lean on hubs, which link everywhere (S8), but I haven't isolated that cause.</p>"""
    # S4
    t4 = []
    for g in ("small", "large"):
        for c in (DW0, "n2v_ns"):
            j = S["jaccard"][g][c]
            t4.append([g, NAME[c], f"{j['5']['mean']:.2f}", f"{j['20']['mean']:.2f}", f"{j['20']['p10']:.2f}", f"{j['20']['p90']:.2f}", f"{Qsd(g, c):.4f}", f"{Q(g, c, sp='0'):.3f} / {Q(g, c, sp='1'):.3f} / {Q(g, c, sp='2'):.3f}"])
        t4.append([g, "Adamic-Adar", "", "", "", "", f"{Qsd(g, 'base_aa'):.4f}", f"{Q(g, 'base_aa', sp='0'):.3f} / {Q(g, 'base_aa', sp='1'):.3f} / {Q(g, 'base_aa', sp='2'):.3f}"])
    h += "<h3>S4. Is the difference noise?</h3>" + table(["Graph", "Row", "Seed Jaccard@5", "@20", "p10 @20", "p90 @20", "Seed sd NDCG@20", "NDCG@20 on splits 0 / 1 / 2"], t4, "Variation from seeds and from the edge split.", (2, 3, 4, 5, 6, 7)) + f"""
<p>Graph-level scores are stable: seed sd is about {Qsd('large',DW0):.4f} on the large graph, so its loss and heuristic gaps are far outside noise. Individual lists aren't. Two seeds share only about half of a top-5 list, so a single list shouldn't be cited as evidence. On the small graph the split moves the score more than seeds do (DeepWalk {Q('small',DW0,sp='0'):.3f} to {Q('small',DW0,sp='1'):.3f}), so the small graph's A/A gap of {abs(PR('small','dw_ns','n2v_ns')['diff']):.3f} is well inside what a different split would do. I treat it as noise: the same comparison on the large graph is {PR('large','dw_ns','n2v_ns')['diff']:+.4f} with seed t {PR('large','dw_ns','n2v_ns')['seed_t']:+.1f}. Adamic-Adar's seed sd is almost zero since its seed only breaks ties.</p>"""
    # S5
    m = XC["dw_hsns"]["metrics"]
    ex = XC["dw_hsns"]["examples"]
    h += "<h3>S5. What happens to a node's recommendations when its graph grows 20×?</h3>" + table(["Measure, 222 physics articles, DeepWalk as written", "K=5", "K=20"], [
        ["Overlap of small-graph list with large-graph list restricted to physics (Jaccard)", f"{m['jaccard5']:.2f}", f"{m['jaccard20']:.2f}"],
        ["Share of large-graph list that is physics", f"{m['phys_share5']:.2f}", f"{m['phys_share20']:.2f}"],
        ["Subcategory agreement: small graph", f"{m['agree_small5']:.2f}", f"{m['agree_small20']:.2f}"],
        ["Subcategory agreement: large graph, restricted to physics", f"{m['agree_large_phys5']:.2f}", f"{m['agree_large_phys20']:.2f}"],
        ["Subcategory agreement: large graph, unrestricted", f"{m['agree_large_all5']:.2f}", f"{m['agree_large_all20']:.2f}"]], f"Same labels on both graphs, which raw lift is not. Base rate on the small graph: {X['base_small']:.2f}. Seed-to-seed Jaccard@20 restricted to physics: {XC['dw_hsns']['seed_jaccard20_small']:.2f} small, {XC['dw_hsns']['seed_jaccard20_large']:.2f} large.", (1, 2)) + f"""
<p>The same article gets a noticeably different list: overlap is {m['jaccard20']:.2f} at K=20, well below the seed noise floor of about {XC['dw_hsns']['seed_jaccard20_small']:.2f}. So the difference is real, not seed noise. Links inside physics make up only {X['physics_edge_share_of_degree']:.0%} of an article's large-graph degree, on average. Restricted to the same 222 candidates, the large graph ranks them slightly better by subcategory ({m['agree_large_phys20']:.2f} vs {m['agree_small20']:.2f}), so outside context helps a little. The clearest case is <code>Integrated_circuit</code>. Physics-only, its top three are {', '.join(ex['Integrated_circuit']['small'][:3])}, which makes little sense. On the large graph they are {', '.join(ex['Integrated_circuit']['large_all'][:3])}, and only {ex['Integrated_circuit']['large_phys_share20']:.0%} of its top 20 is physics. Inducing a subgraph removes context the node depends on, and that hurts most for articles that sit between topics.</p>"""
    # S6
    rat = lambda c, k: TM('large', c, k) / TM('small', c, k)
    h += "<h3>S6. What does it cost, and where does the time go?</h3>" + fig(F.runtime(), "Large graph phases. Word2vec dominates. Precompute and walks are node2vec's only extra cost, and two losses cost about the sum of each.") + table(["Phase", "Small s", "Large s", "Ratio"], [
        ["node2vec precompute", f"{TM('small','n2v_ns','precompute_s'):.2f}", f"{TM('large','n2v_ns','precompute_s'):.1f}", f"{rat('n2v_ns','precompute_s'):.0f}×"],
        ["node2vec walks", f"{TM('small','n2v_ns','walk_s'):.2f}", f"{TM('large','n2v_ns','walk_s'):.1f}", f"{rat('n2v_ns','walk_s'):.0f}×"],
        ["DeepWalk walks", f"{TM('small','dw_ns','walk_s'):.2f}", f"{TM('large','dw_ns','walk_s'):.1f}", f"{rat('dw_ns','walk_s'):.0f}×"],
        ["Word2vec NS", f"{TM('small','dw_ns','train_s'):.2f}", f"{TM('large','dw_ns','train_s'):.1f}", f"{rat('dw_ns','train_s'):.0f}×"],
        ["Word2vec HS", f"{TM('small','dw_hs','train_s'):.2f}", f"{TM('large','dw_hs','train_s'):.1f}", f"{rat('dw_hs','train_s'):.0f}×"],
        ["Peak memory, node2vec (MB)", f"{TM('small','n2v_ns','peak_rss_mb'):.0f}", f"{TM('large','n2v_ns','peak_rss_mb'):.0f}", f"{rat('n2v_ns','peak_rss_mb'):.1f}×"]],
        f"Scaling from small to large. Graph ratios: nodes {N_RATIO:.1f}×, edges {E_RATIO:.1f}×, Σdegree² {D_RATIO:.0f}×.", (1, 2, 3)) + fig(F.workers(), "Workers on the large graph, 3 repeats for 1 worker, one run otherwise. Word2vec scales well; node2vec's total flattens.") + f"""
<p>Each phase follows the quantity its design predicts. Precompute grows {rat('n2v_ns','precompute_s'):.0f}×, close to Σdegree² ({D_RATIO:.0f}×), because it visits every pair of a node's neighbours. DeepWalk walks grow {rat('dw_ns','walk_s'):.0f}×, near the node ratio. node2vec's walks grow more, {rat('n2v_ns','walk_s'):.0f}×, because the library calls <code>random.choices</code> with the full weight list, not the paper's alias method, and walks spend their time at hubs. So its per-step cost grows with degree, not O(1) as the paper's design promises.
<b>Word2vec is the surprise:</b> it grows {rat('dw_ns','train_s'):.0f}× while the corpus grows only {SUB['token_ratio']:.1f}×. Part of that I can explain. Gensim's <code>sample=1e-3</code> discards frequent tokens, and on a 222-node graph every common node counts as frequent: it keeps {SUB['small']['kept_share']:.0%} of small-graph tokens but {SUB['large']['kept_share']:.0%} of large-graph tokens, so effective tokens grow {SUB['effective_ratio']:.1f}×. The remaining factor of about {rat('dw_ns','train_s')/SUB['effective_ratio']:.1f} is unexplained; a larger vocabulary missing cache more often is my guess, untested. That finding also shows the small graph is the faster one <i>per effective token</i>, so per-edge comparisons need care.
<b>Workers:</b> word2vec goes from {TM('large','dw_ns','train_s'):.0f} s to {TM('large','dw_ns','train_s','8'):.0f} s at 8 workers, a {TM('large','dw_ns','train_s')/TM('large','dw_ns','train_s','8'):.1f}× speedup. node2vec's walks best it at 4 workers ({TM('large','n2v_ns','walk_s','4'):.0f} s) and slow down at 8 ({TM('large','n2v_ns','walk_s','8'):.0f} s), probably from copying the probability table to each process, and its precompute is serial, so its total only improves {TM('large','n2v_ns')/TM('large','n2v_ns','total_s','8'):.1f}×. Multi-worker runs also lose reproducibility, which is why every quality run used one worker. HS is cheaper than NS here ({TM('large','dw_hs','train_s'):.0f} s vs {TM('large','dw_ns','train_s'):.0f} s), against the paper's remark that HS is inefficient.</p>"""
    # S7
    h += "<h3>S7. Do dimension, walk length, window and epochs matter?</h3>" + fig(F.sweeps(), "One setting moved at a time from the shared config (node2vec p=q=1, NS, 3 seeds). On the large graph each setting moves NDCG@20 more than p and q do, and several trade it against lift.") + f"""
<p>Yes, and more than p and q. On the large graph the window takes NDCG@20 from {Q('large','n2v_ns+w=5'):.3f} (w=5) to {Q('large','n2v_ns+w=20'):.3f} (w=20), and walk length from {Q('large','n2v_ns+l=20'):.3f} (l=20) to {Q('large','n2v_ns'):.3f} (l=80). <b>Inversions:</b> a bigger dimension hurts on both graphs (d=256 gives {Q('large','n2v_ns+d=256'):.3f}, d=32 gives {Q('large','n2v_ns+d=32'):.3f}), probably because 5 epochs over this corpus are too few to fit that many parameters, though I haven't tested that directly. Lower dimension also raises lift to {L('large','n2v_ns+d=32'):.2f}. And the paper's one epoch gives the best lift among the epoch settings ({L('large','n2v_ns+epochs=1'):.2f}) with the worst NDCG ({Q('large','n2v_ns+epochs=1'):.3f}). So less training keeps vectors broadly topical, and more training pulls in specific neighbours. That trade-off is the same one as in Figure 1. The window explains much of the repo-defaults gap in Question 2, since eliorc inherits gensim's w=5.</p>"""
    # S8
    t8 = []
    for g in ("small", "large"):
        for c, n in (("base_aa", "Adamic-Adar"), ("base_cn", "Common neighbours"), ("dw_hs", "DeepWalk walks, HS"), ("dw_hsns", "DeepWalk"), (T, "node2vec tuned")):
            e = HB[g]["table"][c]
            t8.append([g, n, f"{e['hub_slots']:.1%}", f"{e['rec_degree']:.0f}", f"{e['gini']:.2f}", f"{e['never']:.0%}", f"{e['ndcg']:.3f}", f"{e['ndcg_nohub']:.3f}"])
    h += "<h3>S8. How much do hubs drive this?</h3>" + table(["Graph", "Method", "Top-1% slots", "Mean degree recommended", "Gini of recommendations", "Never recommended", "NDCG@20", "NDCG@20, hubs out"], t8, f"Hubs here are the top 1% by training degree ({HB['large']['n_hubs']} on the large graph, led by {', '.join(HB['large']['hubs'][:3])}; {HB['large']['edge_share_touching_hubs']:.0%} of its test edges touch one). Hubs out: removed from candidates and from the held-out set.", (2, 3, 4, 5, 6, 7)) + f"""
<p>Hubs drive the heuristics, not the embeddings. Adamic-Adar gives {HB['large']['table']['base_aa']['hub_slots']:.0%} of its slots to the top 1% of nodes and never recommends {HB['large']['table']['base_aa']['never']:.0%} of articles, while DeepWalk gives them {HB['large']['table']['dw_hsns']['hub_slots']:.1%} and recommends every article at least once. Taking hubs out lowers Adamic-Adar ({HB['large']['table']['base_aa']['ndcg']:.3f} to {HB['large']['table']['base_aa']['ndcg_nohub']:.3f}) and raises the embeddings (HS {HB['large']['table']['dw_hs']['ndcg']:.3f} to {HB['large']['table']['dw_hs']['ndcg_nohub']:.3f}). The HS and HS+NS rows then pass common neighbours ({HB['large']['table']['base_cn']['ndcg_nohub']:.3f}), while tuned node2vec on NS doesn't ({HB['large']['table'][T]['ndcg_nohub']:.3f}), and none pass Adamic-Adar. The training-side test agrees: dropping hubs from the walks costs little ({Q('large','n2v_ns'):.3f} to {Q('large','n2v_ns+hubdrop=0.01'):.3f}), and <code>sample=0</code> changes nothing on the large graph ({Q('large','n2v_ns+sample=0'):.3f}) but lowers the small one ({Q('small','n2v_ns'):.3f} to {Q('small','n2v_ns+sample=0'):.3f}), where it matters because subsampling is active there (S6). For a recommender, the embeddings' wider coverage may matter as much as the score.</p>"""
    # S9
    t9 = []
    for g in ("small", "large"):
        for m_ in ("cosine", "dot", "euclidean"):
            e = DI[g]["dw_hsns"][m_]
            t9.append([g, m_, f"{e['ndcg20']:.3f}", f"{e['ndcg5']:.3f}", f"{e['lift20']:.2f}", f"{e['lift5']:.2f}", f"{e['jaccard_cos']:.2f}", f"{e['rec_degree']:.0f}"])
    h += "<h3>S9. Does the distance metric change the lists?</h3>" + table(["Graph", "Metric", "NDCG@20", "NDCG@5", "Lift@20", "Lift@5", "Jaccard with cosine", "Mean degree recommended"], t9, f"Same DeepWalk vectors, three metrics, 5 seeds. Spearman between vector norm and degree: {DI['small']['dw_hsns']['cosine']['norm_deg_spearman']:.2f} small, {DI['large']['dw_hsns']['cosine']['norm_deg_spearman']:.2f} large.", (2, 3, 4, 5, 6, 7)) + f"""
<p>Yes, a lot. Lists share only about half their members with cosine. The brief expected dot product to favour high-norm hubs, and <b>the reverse is true:</b> norm falls with degree. The likely reason, which I haven't tested, is that a hub's vector has to sit among many unrelated contexts, which keeps it short. So dot product favours low-degree nodes (mean degree {DI['large']['dw_hsns']['dot']['rec_degree']:.0f}), while Euclidean, which rewards being near the origin, favours hubs ({DI['large']['dw_hsns']['euclidean']['rec_degree']:.0f}). Since test edges lean toward hubs, Euclidean scores best on links, and on the large graph it passes Adamic-Adar ({DI['large']['dw_hsns']['euclidean']['ndcg20']:.3f} vs {Q('large','base_aa'):.3f}) while keeping higher lift ({DI['large']['dw_hsns']['euclidean']['lift20']:.2f} vs {L('large','base_aa'):.2f}). That is the only case where an embedding beats the heuristic on both measures. I would not claim it as a result: I found it after looking at test, it doesn't hold on the small graph ({DI['small']['dw_hsns']['euclidean']['ndcg20']:.3f} vs {Q('small','base_aa'):.3f}), and metric was never selected on validation. "Embedding distance" in the handout hides a real choice.</p>"""
    # S10
    t10 = []
    for g in ("fb_small", "fb_large"):
        for c in ("base_aa", "base_cn", "dw_hsns", "n2v_ns", "n2v_hs", "n2v_p0.25_q4", "n2v_p4_q0.25"):
            e = S["quality"][g][c]["0"]
            t10.append([g, NAME.get(c, c.replace("n2v_p", "node2vec p=").replace("_q", ", q=")), f"{e['test']['20']['ndcg']['mean']:.3f} ± {e['test']['20']['ndcg']['sd']:.3f}", f"{e['label']['20']['lift']['mean']:.2f}", f"{e['n_seeds']}"])
    fs, fl = S["quality"]["fb_small"], S["quality"]["fb_large"]
    h += "<h3>S10. Does it replicate on ego-Facebook?</h3>" + table(["Graph", "Row", "NDCG@20", "Lift@20", "Seeds"], t10, f"A different kind of graph: anonymised friendships, with circles as labels. Base rates {fs['dw_hsns']['0']['base']:.2f} and {fl['dw_hsns']['0']['base']:.2f}. No full p, q tuning, only the two extreme cells.", (2, 3, 4)) + f"""
<p>The shape replicates. Adamic-Adar beats embeddings on links on both graphs. HS beats NS strongly on the large one ({fl['n2v_hs']['0']['test']['20']['ndcg']['mean']:.3f} vs {fl['n2v_ns']['0']['test']['20']['ndcg']['mean']:.3f}), so the loss effect isn't specific to Wikipedia. Extreme p and q don't help. I didn't run the same-loss DeepWalk rows here, so the A/A check has no replication. The HS rows lead on lift on the large graph ({fl['dw_hsns']['0']['label']['20']['lift']['mean']:.1f} vs {fl['base_aa']['0']['label']['20']['lift']['mean']:.1f}), and NS node2vec doesn't ({fl['n2v_ns']['0']['label']['20']['lift']['mean']:.1f}), so here the lift advantage depends on the loss. <b>One inversion:</b> on the small ego net Adamic-Adar wins on lift as well ({fs['base_aa']['0']['label']['20']['lift']['mean']:.2f} vs {fs['dw_hsns']['0']['label']['20']['lift']['mean']:.2f}), so the embeddings' topical advantage isn't universal. My explanation, untested: in one person's friend network, circles are themselves tight clusters of friends, so shared neighbours already find them. That is unlike Wikipedia, where a topic spreads over many articles that don't link to each other. These runs use 3 seeds and only {fs['dw_hsns']['0']['n_label']} labelled nodes on the small graph.</p>"""
    # S11
    h += f"""<h3>S11. Which bugs produced wrong conclusions?</h3>
<p>Three were caught, each by looking at a number that seemed wrong.</p>
<p><b>Tuning seeds in the reported row.</b> <code>stats.py</code> grouped runs by config, so the tuned row at first averaged the seeds that selected it. The small graph showed tuned node2vec at {SEL['small']['selected_test_tuneseeds']:.3f}, a gain of about {SEL['small']['selected_test_tuneseeds']-Q('small','n2v_ns'):.3f} over uniform. With fresh seeds it is {Q('small','n2v_tuned'):.3f}, a gain of {Q('small','n2v_tuned')-Q('small','n2v_ns'):.4f}. The wrong conclusion would have been "tuning pays off on the small graph". Noticing that the row had n=3 where every other had n=5 gave it away. The fix excludes selection seeds from all reported rows.</p>
<p><b>Replication silently missing.</b> The record pattern read <code>fb_small</code> as tag <code>fb</code> plus graph <code>small</code> and skipped those runs without an error. A KeyError when printing the table exposed it. Had I built the report from the first summary, S10 would have been absent and I might have read that as the runs not having finished.</p>
<p><b>The grid would have starved.</b> Not a result bug, but it would have cost the deadline: the first array exceeded the 50-job QOS limit, and later only 2 cores served the Tier 2 list. Working out an ETA for me exposed the second.</p>
<h3>S12. What is still uncertain?</h3>
<ul><li><b>Why HS beats NS.</b> It holds on every graph and I measured the size, but the mechanism isn't established (S1).</li>
<li><b>The blog.</b> I couldn't read it, so any comparison it makes is not in this report.</li>
<li><b>Lift's cause.</b> Embeddings' lift advantage may be mostly hub avoidance. A direct test would re-score Adamic-Adar lift with hubs removed from candidates.</li>
<li><b>Generalisation of Euclidean (S9).</b> It needs selection on validation and more graphs.</li>
<li><b>Word2vec's remaining scaling factor (S6).</b> Unexplained beyond subsampling. A cache-miss count would settle it.</li>
<li><b>Hyperlinks as ground truth.</b> A held-out link is what an editor chose to link, not necessarily what a reader wants. Wikispeedia's human navigation paths are in the same download and would be a better relevance signal to test against.</li>
<li><b>Timing noise.</b> Other jobs may have shared the node. Repeats were tight (sd {S['timing']['large']['n2v_ns']['1']['total_s']['sd']:.1f} s on {TM('large','n2v_ns'):.0f} s), but workers rows beyond 1 are single runs.</li></ul>"""
    h += step2(rquote(0) + rquote(1) + quote("P11") + quote("P13"), R_("R1", "R2", "B"),
               "I kept extras apart from the required answers, as HW3's feedback encouraged. Each section starts from something that broke an expectation. The two recorded rounds are a claim, then a test of that claim against everything else, so the AI's first reading is on the record and its errors can be counted. Status questions like P11 also did real work: answering it exposed the starved job list.",
               (f"each section has evidence and an explanation, and each inversion is explained or marked untested. Round 2 tested all nine round-1 claims.",
                "C5 (walks reach further), C8 (hubs drive embeddings) and C9 (cosine is neutral) all sound right and all failed. The AI's statuses also had slips: it reported \"39 of 39\" for a sweep count that was wrong, wrote that a job had waited 3 hours when it was under 2, and printed \"synced\" after an rsync failed, because the script echoed success unconditionally. I read my one-word reply \"Okay\" (P14) as picking the recommended option, which was a guess it should have flagged more clearly. In S6, the explanation for node2vec's walk cost comes from reading the code; I didn't measure the per-step cost by degree separately.",
                f"on the large graph, moving window or walk length changes NDCG@20 by about {Q('large','n2v_ns+w=20')-Q('large','n2v_ns+w=5'):.2f}, the metric by {DI['large']['dw_hsns']['euclidean']['ndcg20']-DI['large']['dw_hsns']['cosine']['ndcg20']:.2f}, a heuristic against the best embedding by {Q('large','base_aa')-Q('large','n2v_hs'):.2f}, the loss by {PR('large','n2v_hs','n2v_ns')['diff']:.2f}, and p and q by {PR('large',T,'n2v_ns')['diff']:.3f}. The handout's framing puts the two algorithms first, and their difference is the smallest effect on that list."))
    return h


NEG_TEXT = lambda best: (f"More negatives close some of the gap, so too few negatives is part of the explanation." if Q("large", best) > Q("large", "n2v_ns") + 3 * Qsd("large", "n2v_ns") else "More negatives do not close the gap, so the count of negatives isn't the explanation.") + (" NS still falls short of HS, so the rest remains open." if Q("large", best) < Q("large", "n2v_hs") else " With enough negatives, NS reaches HS on this graph.")


def refs():
    items = ["Adamic, L. A., and Adar, E. 2003. Friends and neighbors on the Web. <i>Social Networks</i> 25(3), 211-230.",
             "Grover, A., and Leskovec, J. 2016. node2vec: Scalable feature learning for networks. In <i>Proc. KDD</i>.",
             "Leskovec, J., and Krevl, A. 2014. SNAP Datasets: Stanford large network dataset collection. https://snap.stanford.edu/data",
             "Liben-Nowell, D., and Kleinberg, J. 2007. The link-prediction problem for social networks. <i>JASIST</i> 58(7), 1019-1031.",
             "McAuley, J., and Leskovec, J. 2012. Learning to discover social circles in ego networks. In <i>Proc. NeurIPS</i>.",
             "Mikolov, T., Sutskever, I., Chen, K., Corrado, G., and Dean, J. 2013. Distributed representations of words and phrases and their compositionality. In <i>Proc. NeurIPS</i>.",
             "Perozzi, B., Al-Rfou, R., and Skiena, S. 2014. DeepWalk: Online learning of social representations. In <i>Proc. KDD</i>.",
             "Qiu, J., Dong, Y., Ma, H., Li, J., Wang, K., and Tang, J. 2018. Network embedding as matrix factorization. In <i>Proc. WSDM</i>.",
             "West, R., and Leskovec, J. 2012. Human wayfinding in information networks. In <i>Proc. WWW</i>.",
             "West, R., Pineau, J., and Precup, D. 2009. Wikispeedia: An online game for inferring semantic distances between concepts. In <i>Proc. IJCAI</i>.",
             "Software: phanein/deepwalk at 6e6dff2, https://github.com/phanein/deepwalk. eliorc/node2vec at 872af47, https://github.com/eliorc/node2vec. Řehůřek, R., and Sojka, P. gensim 4.3.3.",
             "Handout blog, not read (HTTP 403 on automated fetch): <i>Exploring Graph Embeddings: DeepWalk and Node2Vec</i>, Towards Data Science. https://towardsdatascience.com/exploring-graph-embeddings-deepwalk-and-node2vec-ee12c4c0d26d"]
    return "<h2>Step 3: References</h2><ol>" + "".join(f"<li>{i}</li>" for i in items) + "</ol>"


def appendices():
    a = "<h2>Step 4: Appendices</h2><h3>Appendix A. Full prompt log</h3><p>Every prompt typed to the AI for HW5, verbatim, typos kept, times US Eastern on Nov 5. P3 is a menu answer. R1 and R2 are instructions the assistant wrote for its own analysis rounds, not my typing. Automated background-task notices aren't prompts and are left out.</p>"
    a += table(["ID", "Time", "Kind", "Text"], [[p[0], p[1], p[2], R.esc(p[3])] for p in PROMPTS] + [[r[0], "", r[1], R.esc(r[2])] for r in ROUND_INSTRUCTIONS], "Prompt log.")
    a += "<p>The brief P1 produced, verbatim. It was the main context for every round after P2:</p><pre class='brief'>" + R.esc((ROOT / "PROMPT.md").read_text()) + "</pre>"
    a += "<h3>Appendix B. Extended tables</h3>" + table(["Graph", "Query", "Algorithm", "Top 20"], list_rows("20"), "All 20 top-20 lists, seed 0, in the notation of Question 6. Each top-5 list in Question 6 is the start of its top-20 list here; the superscript counts seeds with that article in the same K.", cls="lists")
    cells = []
    for g in ("small", "large"):
        for c in ("dw_ns", "dw_hs", "dw_hsns", "n2v_ns", "n2v_hs", "n2v_hsns", "n2v_tuned", "dw_repo", "n2v_repo") + BASE_ROWS:
            e = S["quality"][g][c]["0"]
            cells.append([g, NAME.get(c, c)] + [f"{e['test'][k][m]['mean']:.3f}" for k in ("5", "20") for m in ("p", "r", "ndcg", "hit")] + [f"{e['novel'][k]['lift']['mean']:.2f}" for k in ("5", "20")])
    a += table(["Graph", "Row", "P@5", "R@5", "NDCG@5", "Hit@5", "P@20", "R@20", "NDCG@20", "Hit@20", "Novel lift@5", "@20"], cells, "All rows, test, 5 seeds. Novel lift hides training neighbours from the list.", tuple(range(2, 12)))
    ml = "; ".join(f"{k.replace('pace-ice-', '').replace('(R)', '').replace('-grid', '').replace('-', ' ')}: {v[0]} runs" for k, v in MS.items())
    a += f"""<h3>Appendix C. Compute environment and reproduction</h3>
<p><b>Compute.</b> PACE ICE, partition <code>ice-cpu</code>, no GPU, because neither word2vec nor the walks use one. Python 3.11.9, gensim 4.3.3, numpy 1.26.4, networkx 3.4.2. Quality grid: {NRUNS} runs, {CORE_H:.0f} core-hours, one worker each, <code>PYTHONHASHSEED=0</code> ({ml}). Timing: one 8-core job on Xeon Gold 6226, one run at a time. Local work ran on the Mac; no reported number comes from it.</p>
<p><b>Acceptance checks, written before results (<code>src/check.py</code>), all pass.</b> Graph sizes in band; held-out edges absent from every training walk on both graphs; query never in its own list; metrics, common neighbours and Adamic-Adar match hand counts on toy graphs; same seed gives the same embedding with one worker, and a different seed gives a different one.</p>
<p><b>Reproduce.</b></p><blockquote>pip install -r requirements.txt
python -m src.data &amp;&amp; python -m src.split &amp;&amp; python -m src.check
python -m src.jobs                                # write job lists
sbatch slurm/pool.sbatch slurm/jobs/all.txt       # grid, then extra.txt, tuned, negative.txt
sbatch slurm/timing.sbatch; sbatch slurm/timing_tuned.sbatch; sbatch slurm/part2.sbatch
python -m src.stats &amp;&amp; python -m src.understand &amp;&amp; python -m src.ai_rounds &amp;&amp; python -m src.build_report
STUDENT_ID=... python -m src.build_pdf</blockquote>"""
    return a


def main():
    assert KS_REQUIRED == (5, 20) and SHARED["d"] == 128
    body = cover() + q1() + q2() + q3() + q4() + q5() + q6() + q7() + q8() + part2() + refs() + appendices()
    page = f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>HW5 Mohite</title><style>{R.CSS}{CSS_EXTRA}</style></head><body>{body}</body></html>"
    (REPORT / "report.html").write_text(page)
    print(f"report.html {len(page) // 1024} KB, {TABLES[0]} tables, {FIGS[0]} figures")


if __name__ == "__main__":
    main()
