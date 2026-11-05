# CS 6220 HW5: DeepWalk and node2vec for top-K node recommendation

Programming option. The report is `HW5_P_Mohite_Atharva.pdf`, built by `src/build_report.py` from `results/*.json`.

| Graph | Nodes | Edges | Labels |
|---|---:|---:|---|
| Wikispeedia, physics subgraph | 222 | 1,982 | 5 subcategories |
| Wikispeedia, largest component | 4,589 | 106,534 | 15 subjects |
| ego-Facebook, ego net 0 (replication) | 324 | 2,514 | circles |
| ego-Facebook, combined (replication) | 4,039 | 88,234 | circles |

- `vendor/` holds the two handout repos at their pinned commits; `vendor/patch.diff` is the patch, as a diff.
- `src/recommend.py` is the only place candidates are filtered, so evaluation and the report agree.
- `results/` holds one JSON per run. Every number in the report is read from there at build time.

`PLAN.md` explains the design, `IMPLEMENTATION.md` has the runbook and dated log, and `AGENTS.md` has the working rules.

Run on PACE ICE for anything on the large graph. Python 3.11 only.

Datasets: West & Leskovec 2012 and West, Pineau & Precup 2009 (Wikispeedia); McAuley & Leskovec 2012 (ego-Facebook); via SNAP
(Leskovec & Krevl 2014).
