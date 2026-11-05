# HW5 implementation: status and runbook

`PLAN.md` says why; this file says what's built, what's verified, and the exact commands. Dated results log at the bottom.

## Runbook

```sh
# Mac -> ICE (code and built graphs), needs GlobalProtect
./sync.sh
# Venv + acceptance checks (python/3.11.9, gold6226 node)
ssh amohite8@login-ice.pace.gatech.edu 'cd ~/scratch/HW5 && sbatch slurm/venv.sbatch'
# Grid: shared list, several 8-core pool jobs claim lines. Arrays exceed the 50-job QOS.
ssh ... 'cd ~/scratch/HW5 && sbatch slurm/pool.sbatch slurm/jobs/all.txt'
# Timing: 8 gold6226 cores, sequential; the only source of timing tables
ssh ... 'cd ~/scratch/HW5 && sbatch slurm/timing.sbatch'
# ICE -> Mac, then analysis and report
./sync.sh pull
PYTHONPATH=$PWD .venv/bin/python -m src.stats
```

## Sharp edges

- `coc-ice` QOS: 50 submitted jobs per user, array tasks count.
- `-m src.*` needs `PYTHONPATH=$PWD`. Python 3.11 only: gensim 4.3.3 has no 3.14 wheels.
- Vendored DeepWalk's `if start:` skips node 0, so ids start at 1 (`src/data.py`, `src/embed.py`).
- Timing on gold6226 only; `machine` separates `-grid` (contended, quality only) from `-8core` (timing tables).
- The TDS blog returns HTTP 403 to automated fetches. The user has to read it in a browser, or Q1 discloses it wasn't read.

## Results log

### Nov 5: Tier 0 code built, grid submitted
- Unpatched failures recorded: `ImportError: Iterable` (`graph.py`), `ImportError: Mapping` (`skipgram.py`),
  `PackageNotFoundError` (`node2vec/__init__.py`). Patch: `vendor/patch.diff`.
- Graphs built: small 222/1,982; large 4,589/106,534 (15 articles dropped outside the largest component; `Pikachu`
  kept but unlabelled); fb_small 324/2,514 (largest component of 347 listed, ego node omitted); fb_large 4,039/88,234.
- `src.check` passes on the Mac (small) and on ICE (small and large).
- Dev look at one seed on the Mac (not reportable): Adamic-Adar NDCG@20 0.50 vs embeddings ~0.35, embeddings ahead on
  label lift (~2.0 vs 1.7). The inversion is flagged for S3.

### Nov 5 (afternoon): grid running; first findings on the small graph
- Queue: my fair-share is 0.016 and `gold6226` had ~4 free cores, so 8-core jobs sat pending for 45 min. Dropped the CPU
  constraint for quality runs (`pool.sbatch`; the record names the actual CPU) and used mixed job sizes. Timing stays on
  an exclusive `gold6226` node.
- Cost is far below the plan's estimate: large-graph runs take 200–400 s and 490–630 MB. The brief's warning that the
  precompute is memory-hungry does not hold at this size.
- Reported rows exclude the tuning seeds (a bug caught before any table was built, `src/stats.py`).
- Small graph, 5 seeds, to be explained in Part 2: Adamic-Adar beats every embedding on held-out links and every embedding
  beats Adamic-Adar on label lift. HS beats HS+NS beats NS. Selected p=2, q=0.5.
- S9 inversion: embedding norm falls with degree (Spearman -0.72), so dot product picks low-degree nodes and Euclidean picks
  hubs. Euclidean beats cosine on links (0.418 vs 0.338) and loses on lift. This contradicts the brief's expectation.
- S3: 94% of small-graph test edges are within two hops and 95% of embedding slots land within two hops, so embeddings are
  not reaching further than the heuristics.

### Nov 5 (13:45): timing job moved off an exclusive node
- A whole node stayed pending for 3 hours. The user accepted the 8-core `gold6226` job: same CPU model and thread count, runs
  one at a time, `workers` scaling to 8. Limitation for the report: other users' jobs may share the node.
- Selection: p=2, q=0.5 on both graphs. A/A check holds on the large graph (same loss, DeepWalk vs p=q=1, no significant gap).
