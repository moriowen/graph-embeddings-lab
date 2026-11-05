"""Every fixed choice in one place, so results can record exactly what ran."""
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
RESULTS = ROOT / "results"
NODES = RESULTS / "nodes"          # per-node arrays for paired tests; gitignored
EMB = ROOT / "embeddings"          # gitignored
REPORT = ROOT / "report"

# Wall-clock is only comparable within one machine; every record carries this label.
MACHINE = os.environ.get("BDS_MACHINE", "m2-air")

COMMITS = {"deepwalk": "6e6dff245e4692e9bea47e9017c1034e51afbf29",
           "node2vec": "872af473836786688707157d10ad2b7db3c6423d"}

SOURCES = {"wikispeedia": "https://snap.stanford.edu/data/wikispeedia/wikispeedia_paths-and-graph.tar.gz",
           "facebook": "https://snap.stanford.edu/data/facebook.tar.gz",
           "facebook_combined": "https://snap.stanford.edu/data/facebook_combined.txt.gz"}

# name -> size band the handout requires (None: Part 2 replication, no band applies to it as a pair)
GRAPHS = {"small": (100, 500), "large": (1000, 10000), "fb_small": (100, 500), "fb_large": (1000, 10000)}
MAIN_GRAPHS = ("small", "large")

# Rule fixed before any embedding existed (PLAN section 3): sort the small graph by (degree, title),
# split into 5 equal bins, draw one per bin with random.Random(QUERY_RULE_SEED). data.py re-derives this
# list and asserts it matches. No swaps after results.
QUERY_RULE_SEED = 6220
QUERIES = ["HD_28185", "Integrated_circuit", "90377_Sedna", "Andromeda_Galaxy", "Special_relativity"]

KS_REQUIRED = (5, 20)
KS_ALL = (1, 5, 10, 20, 50)
KMAX = max(KS_ALL)
PRIMARY = ("ndcg", 20)             # declared before results: test NDCG@20

SPLIT_TEST, SPLIT_VAL = 0.20, 0.10
SPLITS = (0, 1, 2)                 # split 0 is the main one; 1, 2 measure split variation
SEEDS = (0, 1, 2, 3, 4)            # reported runs
TUNE_SEEDS = (10, 11, 12)          # selection runs, disjoint from the reported seeds
PQ_GRID = (0.25, 0.5, 1, 2, 4)     # the node2vec paper's grid

# The paper's settings, shared by every comparison row so only the named variable differs.
SHARED = dict(d=128, l=80, r=10, w=10, epochs=5, sample=1e-3)
LOSS = {"ns": dict(hs=0, negative=5), "hs": dict(hs=1, negative=0), "hsns": dict(hs=1, negative=5)}

# Rows of the control matrix. Repo defaults are separate rows: DeepWalk's CLI (d=64, l=40, r=10, w=5, hs+ns)
# and eliorc's constructor (d=128, l=80, r=10) with gensim's w=5.
NAMED = {
    "dw_ns": dict(algo="dw", **LOSS["ns"]),
    "dw_hs": dict(algo="dw", **LOSS["hs"]),
    "dw_hsns": dict(algo="dw", **LOSS["hsns"]),      # DeepWalk as its code is written
    "n2v_ns": dict(algo="n2v", p=1, q=1, **LOSS["ns"]),  # node2vec with its default loss, uniform walk
    "n2v_hs": dict(algo="n2v", p=1, q=1, **LOSS["hs"]),
    "n2v_hsns": dict(algo="n2v", p=1, q=1, **LOSS["hsns"]),
    "dw_repo": dict(algo="dw", d=64, l=40, r=10, w=5, **LOSS["hsns"]),
    "n2v_repo": dict(algo="n2v", p=1, q=1, d=128, l=80, r=10, w=5, **LOSS["ns"]),
}
BASELINES = ("base_random", "base_popularity", "base_cn", "base_aa")
MATRIX = ("dw_ns", "dw_hs", "dw_hsns", "n2v_ns", "n2v_hs", "n2v_hsns")

LABELS = {"dw_hsns": "DeepWalk (as written: HS+NS)", "dw_ns": "DeepWalk walks, NS", "dw_hs": "DeepWalk walks, HS",
          "n2v_ns": "node2vec p=q=1, NS", "n2v_hs": "node2vec p=q=1, HS", "n2v_hsns": "node2vec p=q=1, HS+NS",
          "n2v_tuned": "node2vec tuned p*, q* (NS)", "dw_repo": "DeepWalk repo defaults",
          "n2v_repo": "node2vec repo defaults", "base_random": "Random", "base_popularity": "Popularity",
          "base_cn": "Common neighbours", "base_aa": "Adamic-Adar"}


def resolve(name):
    """Config name -> full parameter dict. Grammar: named row or n2v_p<p>_q<q>, then +key=value overrides."""
    head, *over = name.split("+")
    m = re.fullmatch(r"n2v_p([\d.]+)_q([\d.]+)", head)
    if m:
        c = dict(algo="n2v", p=float(m[1]), q=float(m[2]), **LOSS["ns"])
    elif head in NAMED:
        c = dict(NAMED[head])
    else:
        raise KeyError(name)
    c = {**SHARED, **c}
    for o in over:
        k, v = o.split("=")
        c[k] = float(v) if k in ("p", "q", "sample", "hubdrop") else int(v)
    return c
