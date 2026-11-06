# AGENTS.md

CS 6220 HW5, programming option: DeepWalk and node2vec for top-K node recommendation on two graphs
from one entity set (Wikispeedia physics subgraph and full Wikispeedia). `PLAN.md` has the reasoning,
and `IMPLEMENTATION.md` has status, the runbook and the dated results log. The workspace rules in
`~/dev/acads/AGENTS.md` apply, including **never write a GTID into a tracked file**. The build follows
HW3 and HW4 (`../HW3/`, `../HW4/`): builders are copied and adapted, not rewritten.

## Invariants

- **No model output or AI claim becomes ground truth.** Relevance comes from held-out edges and dataset
  labels. A list looking sensible is not evidence of quality.
- **Numbers come from `results/*.json`** at build time. Every record carries `machine`, repo commit,
  hyperparameters and seed.
- **One recommendation function.** `src/recommend.py` is the only place candidates are filtered, so the
  metrics and the report's lists cannot drift.
- **Held-out edges leave before the walks.** Test edges are never seen by a walk. p and q are selected on
  validation only.
- **`STUDENT_ID` enters only at PDF build time.**

## Vetoes

- Required results use the handout's two repos at the pinned commits in `vendor/`. The patches stay a
  reviewable diff and are disclosed in the report. Other implementations appear only as labelled Part 2
  checks.
- Never run the large graph on the Mac (8 GB). Timed runs share one node type and thread count, and
  machines are never mixed in one timing table.
- No remote without asking. The site is `public/`, from `src/build_site.py`, at https://graph-embeddings-lab.vercel.app (user asked, Nov 5). `.vercelignore` allows only `public/` and `vercel.json`, so keep it that way: this folder holds the ID-bearing PDF and the handout. Commits carry no AI attribution.

## Decisions

See `PLAN.md` §2 onward. Amend the decision there when it changes, with the date.

## Commands

See `IMPLEMENTATION.md` → Runbook. On the Mac, run only the small graph, stats and report building.
