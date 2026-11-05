# Prompt: CS 6220 HW5 (graph embedding: DeepWalk and node2vec), programming option

Paste everything below the line into a fresh Claude Code session started in
`~/dev/acads/bds/HW5/`.

---

You are doing CS 6220 (Big Data Systems) **HW5, programming option**, in `~/dev/acads/bds/HW5/`.
The handout is `2026-ProgrammingAssignment-5.pdf`. Read all of it before you do anything else.
`pdftotext` is not installed on this Mac; use `pypdf` (`/usr/local/bin/python3` or
`/opt/homebrew/bin/python3`). **Due Fri Nov 6 at midnight. The no-penalty extension runs to Sat Nov 7
at 09:00, and the deadline is hard.** Today is Thu Nov 5, so you have about a day and a half, half
of what HW4 had. Plan in tiers, so that each tier ends with something I can submit.

Follow the same setup as HW4 (`../HW4/`), which followed HW3 (`../HW3/`, the graded "Top submission"
and the gold standard). Before you plan anything, read `../HW4/AGENTS.md`, `../HW4/PLAN.md`,
`../HW4/IMPLEMENTATION.md`, `../HW4/README.md`, `../HW4/.gitignore`, and skim `../HW4/src/`
(especially `build_report.py`, `report_lib.py`, `build_pdf.py`, `package.py`, `prompts.py`,
`stats.py`, `part2.py`, `understand.py`) and `../HW4/slurm/`. Reuse and adapt those builders. Do not
rewrite them from scratch. The workspace rules in `~/dev/acads/AGENTS.md` apply, including **never
write a GTID into a tracked file**.

## 1. What the handout asks (turn this into the checklist in PLAN.md §1)

| # | Handout step | Deliverable |
|---|---|---|
| D1 | 1 | Read the blog and the node2vec paper (Grover & Leskovec, KDD 2016) comparing the two algorithms. State what differs and what the comparison predicts for our graphs. |
| D2 | 2 | Download and evaluate the handout's implementations: `phanein/deepwalk` and `eliorc/node2vec`. |
| D3 | 3 | Two graph datasets, "ideally from the same set of entities": one of 100–500 nodes, one of 1,000–10,000 nodes. |
| D4 | 4a | Graph-based recommendation by embedding distance: 5 queries on 5 different nodes, for each algorithm, on both graphs. That is 5 × 2 × 2 = 20 query runs. |
| D5 | 4b | Given a node, recommend its top-K nearest nodes. Report **runtime** and **recommendation quality**. |
| D6 | 4c | Visualise the top-K for each query node under each algorithm, or report and compare them in tables. Do both: tables for all, figures for a selection. |
| D7 | 4d | Elaborate the results under two K settings (use K=5 and K=20 unless the plan argues otherwise). That makes 40 top-K lists. |
| D8 | 4e, bonus | Compare DeepWalk and node2vec on both graphs. Elaborate. |
| — | | File `HW5_P_Mohite_Atharva.pdf` (or `.zip`). |

**Handout anomalies. Record each in PLAN.md and in the report rather than resolving it silently:**

1. Step 2 says "either DeepWalk or Node2Vec", but 4a and 4c say "each of the two algorithms", and 4e
   makes the comparison a bonus. Run both, which satisfies every reading and earns the bonus.
2. 4e says "the above two learning tasks", but step 4 only describes one task, top-K recommendation.
   Read it as the 4a query results and the 4b runtime and quality results.
3. The handout never defines "recommendation quality". Section 3 defines it. Say that this is our
   definition.
4. The handout has no naming line and no final-remarks section. The filename comes from the course
   convention (HW1–HW4). Check the rubric template for anything else it requires.
5. Its posting date is Oct 26, while Canvas (per `calendar.md`) opens HW5 on Oct 16. See section 7.

## 2. Grading reality (read the `bds-6220-hw-report` skill now, not at the end)

The rubric is applied **per question**: Step 1 final answer 15%, Step 2 AI interaction record 80%
(tool URL 5%, exact prompts and every round 15%, **4a strategy 30%**, **4b critique 30%**), and
references 5%. Appendices can earn bonus points. Match HW3 and HW4's shape, not the template's
8-page limit: Questions 1–N answer the handout first and plainly, about a page each, with "At a
glance" and the limitations under the cover. Extra experiments go in **Part 2, "Understanding the
results"**, as 8–11 sections titled as questions, each showing evidence and explaining it. Every
surprising result gets an explanation. My standing preference is in memory as
`bds-reports-exhaustive`.

**Who writes 4b:** unlike HW4's original prompt, you draft every 4b critique, in HW3's three parts
(requirements met; what the AI overlooked, oversimplified or miscalculated, with file and number
evidence; refined synthesis). Keep a running `report/critique-notes.md` as you work. Tell me to read
them for voice, since the rubric says "your own words". Draft the analysis in recorded AI rounds
(`report/ai_analysis_r*.md`, as in HW4) so that there are real claims to critique.

## 3. Scientific design (this is what moves the report from a "pass" to a "pass+")

The naive version is "I ran both libraries with defaults, printed five neighbour lists, and they
look reasonable." Build it to answer questions that version can't.

**Datasets. I checked these on Nov 5 by downloading them from SNAP; re-verify when you build.**

| Role | Graph | Nodes | Undirected edges | Labels for quality |
|---|---|---:|---:|---|
| Small | Wikispeedia, induced on `subject.Science.Physics` | 222 | 1,982 (connected) | 5 subcategories, such as `Space_Astronomy` and `General_Physics` |
| Large | Wikispeedia, full | 4,592 (largest component 4,589) | 106,537 | 15 top-level subjects |
| Replication (Part 2) | ego-Facebook: ego net `0` / combined | 333 / 4,039 | 2,519 / 88,234 | circles |

Wikispeedia is my recommended primary because nodes are named Wikipedia articles, so a reader can
judge a recommendation by eye ("Wonderful tables, graphs, and examples" was the HW3 feedback, and
"node 412 → 519, 63" can't earn that). The small graph is nested in the large one, which meets
"the same set of entities" literally and lets you run **the same five query nodes on both graphs**.
ego-Facebook is anonymised, so it's the replication check, not the main result. Present this choice
in PLAN.md for my approval.

Data traps I found:
- Wikispeedia links are directed. Symmetrise them, drop self-links, URL-decode titles (`%28`), and
  say so. 598 articles have several categories, so relevance means "shares any label".
- Six articles have no category and a few nodes fall outside the largest component. Decide, then
  state the rule.
- SNAP's `facebook/0.edges` lists each edge twice and **omits the ego node** (`0`). The combined
  graph has a hub of degree 1,045.
- The hubs in Wikispeedia are large (United_States has degree 1,621).

**Design:**

1. **Quality has two independent measures, both defined before you look at results.** (a) **Held-out
   links:** remove 20% of edges keeping the graph connected (spanning-tree protected), plus 10% for
   validation; train on what's left, then compute Precision@K, Recall@K, NDCG@K and hit rate against
   held-out neighbours. (b) **Label agreement:** the fraction of top-K sharing a category with the
   query, reported as **lift over the base rate**, since large categories make raw agreement look
   good. Exclude the query node and its training neighbours from candidates under (a), and say so.
   Remove held-out edges **before** the walks, or the result is leakage. Compute each measure over
   **all nodes**, because five queries alone are anecdotes. The five queries are the required
   examples.
2. **Baselines.** Random, popularity (highest degree), and the classical link predictors **common
   neighbours and Adamic-Adar**. An embedding has only earned its cost if it beats these. If a
   heuristic wins on a dense graph, that's a finding to explain, not to hide.
3. **Isolate the algorithm from the implementation.** This is the important control. DeepWalk
   hard-codes `sg=1, hs=1` (hierarchical softmax), while eliorc's node2vec uses gensim's default
   negative sampling. The repos also have different defaults: DeepWalk `d=64, l=40, r=10, w=5`, and
   node2vec `d=128, l=80, r=10`. A raw defaults comparison conflates all of this with the biased walk.
   So compare on one shared config, and include **node2vec with p=q=1**, which is a DeepWalk-style
   uniform walk. Then run it with `hs=1` too. The gaps tell you how much comes from walk bias and how
   much from the loss. Report repo defaults as a separate row.
4. **Tune p and q honestly.** Use the paper's grid ({0.25, 0.5, 1, 2, 4}), select **on the validation
   split only**, and report on test. Selecting on test hands node2vec an unearned win. It has two
   extra knobs, and its paper's advantage partly rests on them.
5. **Variation.** Use at least 5 seeds per config. Report mean ± sd, plus a paired comparison over
   nodes (bootstrap CI or Wilcoxon) rather than bare means. Also report the **top-K Jaccard overlap
   between seeds**: if a node's recommendations change with the seed, a single list is not evidence.
   gensim is only deterministic with `workers=1` and a fixed `PYTHONHASHSEED`, so quality runs use
   that.
6. **Query nodes by a rule fixed in advance**, not cherry-picked: stratified by degree within the
   small graph (low, median and high, seeded), with the list stored in `config.py`. No swapping a node
   afterwards because its list looks bad.
7. **Runtime broken down, on one machine.** Record walk generation (and node2vec's probability
   precompute separately), word2vec training, and single-query latency (mean and p95 over many
   queries), plus peak RSS. Report how each scales from small to large against the node and edge
   ratios. Every timed run uses the same node type and thread count, and every record carries `machine`.
8. **Two K settings explained, not just tabulated.** Precision falls and recall rises with K, but
   precision@K is capped for nodes with fewer than K held-out neighbours. Report that cap, or use a
   normalised metric beside it.
9. **Distance is a choice.** "Embedding distance" defaults to cosine. State it, and compare Euclidean
   and dot product in Part 2. Dot product favours high-norm nodes, which may be the hubs.

**Part 2 candidates.** Draft this list in PLAN.md, since the runs need scheduling:
S1 how much of the gap is walk bias and how much is the loss (item 3) · S2 was tuning p and q worth
it, and do BFS-like or DFS-like settings win · S3 do embeddings beat common neighbours (item 2) ·
S4 is the difference noise (seeds, top-K stability) · S5 what happens to the same node's
recommendations when its graph grows 20× · S6 what does it cost, including `workers` scaling and
where the time goes · S7 do dimension, walk length and window matter · S8 how much do hubs drive
this (exclude them from candidates and from walks) · S9 does the distance metric change the lists ·
S10 does it replicate on ego-Facebook · S11 what's still uncertain. If you hit a bug that produced a
wrong conclusion, add a section on it; HW3's was praised.

## 4. Handling rules (these are vetoes, not preferences)

- **`phanein/deepwalk` will not import on a modern stack.** I checked on Nov 5: commit `6e6dff2`
  (Apr 2020) has `from collections import Mapping` (gone in Python 3.10), `from
  gensim.models.word2vec import Vocab` and `size=` (gone in gensim 4). Smoke-test both repos **first**.
  My preference: vendor DeepWalk pinned at that commit with the smallest possible patch, kept as a
  reviewable diff and disclosed in the report, so both algorithms share one gensim. Pin eliorc at
  `872af47` the same way.
- **Use the handout's two repos.** Don't substitute PyG, `karateclub` or a reimplementation for the
  required results. A separate implementation is allowed only as a labelled check in Part 2.
- **No model output or AI claim becomes ground truth.** Numbers come from `results/*.json` at build
  time. Labels come from the dataset. A list "looking sensible" is never evidence of quality.
- **Compute.** No GPU is needed: this is CPU work, and word2vec won't benefit. The precompute in
  eliorc's node2vec scales with the sum of squared degrees (about 32M on full Wikispeedia after
  symmetrising) and can be memory-hungry, so don't run the large graph on the Mac (8 GB, swaps hard).
  Develop and smoke-test on the small graph locally. Run the timed grid on a **PACE ICE CPU node**
  (runbook: `../HW4/IMPLEMENTATION.md`, `../HW1/docs/PACE-ICE.md`), with a new small venv. Look up the
  CPU partition instead of assuming it, and check queue times early. If ICE is backed up on Thursday
  night, tell me, and then decide between a smaller grid and the Mac. Never mix machines in one timing
  table.
- **Git.** Set up the HW5 folder as its own repo as in HW4, with `.gitignore` covering the handout,
  downloaded datasets (re-fetchable), embeddings and build output. Ask me before creating a remote or
  deploying a site. Commits carry no AI attribution.
- **Cite the datasets** as SNAP asks: West & Leskovec for Wikispeedia, McAuley & Leskovec for
  ego-Facebook.

## 5. Build (HW4 layout, adapted)

```
HW5/
  AGENTS.md, PLAN.md, IMPLEMENTATION.md, README.md   # same roles as HW4
  config.py            # graphs, repo commits, shared hyperparams, p/q grid, K values, seeds, query rule, MACHINE label
  requirements.txt
  vendor/              # pinned deepwalk + eliorc node2vec, with the patch as a diff
  data/                # built graphs + provenance JSON; raw SNAP downloads gitignored
  src/data.py          # fetch, symmetrise, induce the small graph, connectivity + size checks
  src/split.py         # seeded train/val/test edge split (connectivity protected)
  src/embed.py         # one entry point for each algorithm config -> embeddings + phase timings + peak RSS
  src/recommend.py     # top-K by distance; the one place candidates are filtered, so run and report agree
  src/baselines.py     # random, popularity, common neighbours, Adamic-Adar
  src/evaluate.py      # P/R/NDCG/hit@K, label lift, top-K lists for the query nodes -> results/*.json
  src/stats.py         # mean ± sd over seeds, paired comparison, seed-to-seed Jaccard
  src/understand.py, src/part2.py   # Part 2 analyses and figures
  src/build_report.py, report_lib.py, build_pdf.py, package.py, prompts.py   # adapted from HW4
  slurm/               # one CPU array job; same node type and thread count for every row
  results/             # one JSON per run: the only source for every number
  report/              # report.html, ai_analysis_r*.md, critique-notes.md
```

Invariants carried over from HW3 and HW4: numbers come from `results/*.json` at build time; one
recommendation function shared by evaluation and report; every record carries `machine`, the repo
commit, hyperparameters and seed; `STUDENT_ID` only enters at PDF build time.

## 6. Tiers and cut lines

Times assume you start Thursday afternoon. I also have other coursework due Nov 6–8, so don't count on
long review sessions from me.

- **Tier 0 (Thu afternoon–evening), required.** Smoke-test both repos. Build the graphs and split.
  Implement embed, recommend, evaluate and the baselines. Run both algorithms on both graphs on the
  shared config, 5 seeds. Produce the 40 top-K lists, the runtime table and the quality table. This
  covers D1–D7 and alone is submittable.
- **Tier 1 (Thu night).** The p and q tuning grid, the p=q=1 and `hs=1` controls, the D8 comparison,
  and the figures. Run the AI analysis rounds, recorded, and start the 4b notes. This gets S1–S4 and S6.
- **Tier 2 (Fri morning, only if Tier 1 is done).** S5, S7, S8, S9 and the ego-Facebook replication
  (S10). A public site as in HW3 and HW4, only if I ask.
- **Fri afternoon: write. Be packaged by Fri 20:00, leaving Sat 09:00 for emergencies only.** Run the
  `bds-6220-hw-report` skill and use its skeleton. Map D1–D8 onto Questions. Collect prompts with
  `~/.claude/skills/bds-6220-hw-report/scripts/extract_prompts.sh "$PWD" 2026-11-05`. Humanize the
  prose. Report the page count. Run the pre-submission checklist, package
  `HW5_P_Mohite_Atharva.pdf`, and grep everything for a GTID.
- **Cut order:** the site first, then S10, S7, S9, S8, S5. **Never cut:** the held-out split, the
  baselines, more than one seed, the p=q=1 control, the runtime breakdown, or the recorded AI rounds
  for 4a and 4b.

## 7. Workspace bookkeeping

When the plan exists, update `../README.md` (the HW5 status line) and the "as of" date in
`~/dev/acads/README.md`. The due date lives only in `~/dev/acads/calendar.md`, so check that it says
Nov 6 and don't copy it anywhere else. **Calendar conflict:** `calendar.md` and `../README.md` list
HW5's opening as an unresolved Canvas Oct 16 question, and the handout states a posting date of Oct 26.
Record the handout's date as a second source in the calendar's Unresolved section; don't pick
silently. Then update the note in `../README.md` to match.

## 8. How to work with me

1. Read everything in sections 1–2 and the HW4 files listed above. Smoke-test both repos, because
   that result changes the plan.
2. Write `AGENTS.md` and `PLAN.md` (checklist, dataset choice, design, tiers, risks, references) and
   **stop for my approval** before writing the rest of the code. Keep it short: approval is one message
   from me, and time is tight.
3. After approval, build and run Tier 0 with acceptance checks written before the code: graphs fall
   in the handout's size bands, held-out edges are absent from training walks, the query node never
   appears in its own list, the metrics match a hand count on a toy graph, and the same seed reproduces
   the same embedding with `workers=1`.
4. Keep `IMPLEMENTATION.md` as a dated status and results log, the same way HW4 did.
5. Flag every inversion. For example, if DeepWalk beats tuned node2vec, if common neighbours beats
   both, or if the larger graph is *faster per edge*, that is a finding to explain, not something to
   hide.

## References to start from

Perozzi, Al-Rfou, Skiena, *DeepWalk: Online Learning of Social Representations* (KDD 2014). Grover &
Leskovec, *node2vec: Scalable Feature Learning for Networks* (KDD 2016). The handout's Towards Data
Science blog, *Exploring Graph Embeddings: DeepWalk and Node2Vec* (take the author from the page when
you fetch it, and note if it's paywalled). Mikolov et al., *Distributed Representations of
Words and Phrases and their Compositionality* (NeurIPS 2013). Liben-Nowell & Kleinberg, *The
Link-Prediction Problem for Social Networks* (2007). Adamic & Adar, *Friends and Neighbors on the Web*
(2003). Qiu et al., *Network Embedding as Matrix Factorization* (WSDM 2018). West & Leskovec, *Human
Wayfinding in Information Networks* (WWW 2012), and West, Pineau, Precup, *Wikispeedia* (IJCAI 2009).
McAuley & Leskovec, *Learning to Discover Social Circles in Ego Networks* (NeurIPS 2012). Leskovec &
Krevl, *SNAP Datasets* (2014). The two repos at their pinned commits.
