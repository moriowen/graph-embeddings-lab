# HW5 plan: DeepWalk vs node2vec for top-K recommendation

**Status (Nov 5): awaiting approval.** The deadline lives in `~/dev/acads/calendar.md`.

## 1. Checklist and handout anomalies

| # | Step | Deliverable | Report |
|---|---|---|---|
| D1 | 1 | Read blog and node2vec paper; what differs, what it predicts for our graphs | Q1 |
| D2 | 2 | Download and evaluate `phanein/deepwalk` and `eliorc/node2vec` | Q2 |
| D3 | 3 | Two graphs, ideally same entities: 100–500 and 1,000–10,000 nodes | Q3 |
| D4 | 4a | Embedding-distance recommendation: 5 nodes × 2 algorithms × 2 graphs = 20 runs | Q4 |
| D5 | 4b | Top-K nearest nodes; report runtime and quality | Q5 |
| D6 | 4c | Top-K per query per algorithm: tables for all, figures for a selection | Q6 |
| D7 | 4d | Two K settings (K=5, K=20), so 40 lists | Q7 |
| D8 | 4e bonus | DeepWalk vs node2vec on both graphs | Q8 |
| — | | `HW5_P_Mohite_Atharva.pdf` (the course convention; the handout has no naming line) | |

Anomalies, recorded in the report as well:
1. Step 2 says "either", 4a and 4c say "each of the two", and 4e makes the comparison a bonus. Run both.
2. 4e says "the above two learning tasks", but step 4 describes one. Read as 4a queries plus 4b runtime and quality.
3. "Recommendation quality" is undefined. §3 is our definition.
4. No naming line or final-remarks section. The rubric template asks for nothing beyond its standard Steps 1–4.
5. The handout says posted Oct 26; Canvas (`calendar.md`) says opens Oct 16. Recorded under *Unresolved*.

## 2. Smoke-test results (these change the design)

**Not yet executed.** The session's permission classifier blocked running the cloned repos ("Code from
External"). The checks below come from reading the code at the pinned commits and querying the installed
gensim 4.3.3. A grep of both repos found no network, subprocess or exec calls.

| Finding | Evidence | Consequence |
|---|---|---|
| **DeepWalk is not pure hierarchical softmax.** It passes `hs=1` and leaves gensim's `negative=5`, and gensim then trains **both** losses | `deepwalk/__main__.py:75`; `word2vec_inner.pyx` runs `if c.hs` and `if c.negative`; a test model holds both `syn1` and `syn1neg` | Corrects the brief's "`hs=1` = hierarchical softmax". The controls need three losses (§3.3) |
| DeepWalk cannot import | `collections.Mapping` (`skipgram.py:1`) and `collections.Iterable` (`graph.py:14`) are gone; `size=` is not a gensim 4 argument (`__main__.py:75,95`, `skipgram.py:21`) | Patch as below. **`word2vec.Vocab` still exists in 4.3.3**, so that part of the brief does not cause a failure |
| **DeepWalk skips walks from node 0.** `if start:` (`graph.py:130`) treats `0` as "no start" | Read from source | Node ids start at 1. This matters for ego-Facebook, which has a node `0` |
| DeepWalk never passes a seed to Word2Vec; gensim's default `seed=1` means seeds only vary the walks | `__main__.py:75` | Patch adds `seed` |
| eliorc's `__init__.py` needs installed package metadata (`metadata.version('node2vec')`) | `node2vec/__init__.py` | A vendored copy needs a one-line patch or `pip install --no-deps` |
| eliorc's `fit()` inherits gensim's `min_count=5`, `window=5` | `node2vec.py:186` | Always pass `min_count=0` and `window` |
| eliorc's unweighted path raises and catches an exception per neighbour during precompute | `node2vec.py:115-123` | Inflates precompute time. Report it as an implementation cost, not an algorithm cost |

**Planned DeepWalk patch (smallest diff):** `collections.abc` imports, `vector_size=`, and a `seed`
argument. `if start:` stays unfixed and is worked around by ids starting at 1. Pins: deepwalk `6e6dff2`,
node2vec `872af47`, gensim 4.3.3, Python 3.11.

## 3. Design

**Datasets (re-verified from SNAP, Nov 5).**

| Role | Graph | Nodes | Edges | Labels |
|---|---|---:|---:|---|
| Small | Wikispeedia induced on `subject.Science.Physics` | 222 (connected) | 1,982 | 5 subcategories |
| Large | Wikispeedia, largest component | 4,589 (of 4,592 with links) | ~106.5k, exact at build | 15 top-level subjects |
| Replication (S10) | ego-Facebook ego `0` / combined | 333 / 4,039 | 2,519 / 88,234 | circles |

Rules: symmetrise, drop 110 self-links, URL-decode titles, keep the largest component. Articles without a
category stay in the graph and are left out of the label measure only. "Relevant" means sharing any label.
**Label trap:** small uses subcategories and large uses top-level subjects, so their lifts are not comparable
across graphs. S5 re-scores physics nodes on the large graph with subcategories.

**Query nodes (rule fixed before any embedding exists):** sort the small graph by degree, split into quintiles,
pick one per quintile with `random.Random(6220)`. No swaps afterwards. Result, as small/large degree:
`HD_28185` 7/11, `Integrated_circuit` 8/43, `90377_Sedna` 13/20, `Andromeda_Galaxy` 18/19,
`Special_relativity` 28/39. **Limitation:** the draw includes no true hub, since `Sun` has 113; S8 covers hubs.

**1. Quality, defined before results.**
- *Held-out links:* a seeded spanning tree is protected, then 20% test and 10% validation edges are removed
  **before any walk**. Metrics are P/R/NDCG/hit@K, over all nodes with ≥1 held-out edge (count reported).
  Candidates exclude the query and its training neighbours; test scoring also excludes validation edges.
- *Label agreement:* the fraction of top-K sharing a label, as **lift over each node's base rate**, over all
  labelled nodes, with only the query excluded.
- Primary metric: **test NDCG@20**, declared now.

**2. Baselines:** random, popularity (train degree), common neighbours, Adamic-Adar, on the same candidates.

**3. Control matrix** on one shared config, d=128, l=80, r=10, w=10 (the paper's), 5 epochs, `workers=1`:

| | NS (`neg=5`) | HS (`hs=1, neg=0`) | HS+NS (DeepWalk as written) |
|---|---|---|---|
| DeepWalk walks | ✓ | ✓ | ✓ **DeepWalk row** |
| node2vec p=q=1 | ✓ **node2vec default loss** | ✓ | ✓ |
| node2vec tuned p*, q* | ✓ **node2vec row** | | |

Reading it: columns give the loss effect; the first two rows should tie, which is an A/A check that the two
codebases agree; the third row gives the walk-bias effect. Repo defaults (DeepWalk d=64, l=40, w=5; node2vec
d=128, l=80, w=5) are a separate row. Gensim's `sample=1e-3` stays on for all and is examined in S8.

**4. Tuning:** {0.25, 0.5, 1, 2, 4}² with NS, 3 seeds per cell, selected per graph on validation NDCG@20
only, then reported on test with 5 fresh seeds.

**5. Variation:** 5 seeds on one fixed split, mean ± sd, paired per-node Wilcoxon and bootstrap CI, top-K
Jaccard between seeds. Split variation: 2 further splits for the main rows. `workers=1` and fixed `PYTHONHASHSEED`.

**6. Lists:** the 40 required lists use the literal handout meaning (nearest nodes, query excluded), marked
✓ held-out, ● training neighbour, and by label match. Seed-to-seed stability is shown next to them.

**7. Runtime:** dedicated job on 8 `gold6226` cores, runs one at a time, separate from the quality grid, 3 repeats per row (amended Nov 5: an exclusive node sat pending for 3 hours, so other users may share the node and `workers` scaling stops at 8):
probability precompute, walks, word2vec, single-query mean and p95, peak RSS, and baselines. Scaling is compared
with node ratio (20.7×), edge ratio (~54×) and Σdeg² ratio (~244×).

**8. K:** each K reports the precision cap (mean of min(K, |held-out|)/K) beside P@K, with NDCG as the
normalised metric.

**9. Distance:** cosine for all required results; Euclidean and dot product in S9.

**Runs:** 7 matrix × 5 seeds + 2 defaults × 5 + 24 tuning cells × 3 ≈ 117 per graph, ~234 in total, plus timing.
Estimated at 10–15 min each on the large graph (to be measured), which is about 25 core-hours on the array.
If the ICE queue is backed up, cut first to l=40, then to fewer tuning seeds.

## 4. Part 2 questions

S1 how much of the gap is walk bias, loss or codebase · S2 was tuning worth it, BFS-like or DFS-like · S3 do
embeddings beat common neighbours · S4 noise (seeds, splits, top-K stability) · S5 same node when its graph
grows 20× · S6 cost, `workers` scaling, where time goes · S7 dimension, walk length, window · S8 hubs (excluded
from candidates and walks; `sample`) · S9 distance metric · S10 ego-Facebook replication · S11 what's uncertain.
A section gets added for any bug that produced a wrong conclusion.

## 5. Tiers

| Tier | When | Scope | Ends with |
|---|---|---|---|
| 0 | Thu | Vendor + patch, graphs, split, embed, recommend, evaluate, baselines; both algorithms, both graphs, 5 seeds; 40 lists, runtime and quality tables | Submittable D1–D7 |
| 1 | Thu night | Tuning grid, controls, D8, figures, recorded AI rounds, 4b notes | S1–S4, S6 |
| 2 | Fri morning | S5, S7, S8, S9, S10 | Full Part 2 |
| Write | Fri afternoon | Report, humanize, prompts, checklist; **packaged by Fri 20:00** | `HW5_P_Mohite_Atharva.pdf` |

Cut order: site, S10, S7, S9, S8, S5. **Never cut:** held-out split, baselines, more than one seed, the p=q=1
control, runtime breakdown, recorded AI rounds.

Acceptance checks, written before the code: graph sizes within bands; held-out edges absent from every training
walk; query never in its own list; metrics match a hand count on a toy graph; same seed gives the same embedding
with `workers=1`.

## 6. Risks

| Risk | Plan |
|---|---|
| PACE needs the VPN (unreachable from this session on Nov 5) | User connects GlobalProtect; ICE CPU partition looked up then, not assumed |
| ICE queue slow Thursday night | Tell the user, then choose a smaller grid or the Mac for small only. Never mix machines in a table |
| Heuristics beat embeddings on these dense graphs (plausible) | A finding to explain (S3), not hide |
| Tuned node2vec fails to beat DeepWalk | Same. Report the validation-selected result as it stands |
| The AI writes 4b | User reads every 4b for voice, since the rubric says "your own words" |

## 7. References

Perozzi et al. KDD 2014; Grover & Leskovec KDD 2016; the handout's TDS blog (author taken from the page at
fetch time; note if paywalled); Mikolov et al. NeurIPS 2013; Liben-Nowell & Kleinberg 2007; Adamic & Adar 2003;
Qiu et al. WSDM 2018; West & Leskovec WWW 2012; West, Pineau, Precup IJCAI 2009; McAuley & Leskovec NeurIPS 2012;
Leskovec & Krevl SNAP 2014; the two repos at their pinned commits.
