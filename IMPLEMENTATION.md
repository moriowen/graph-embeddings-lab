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
# Timing: one exclusive node, sequential; the only source of timing tables
ssh ... 'cd ~/scratch/HW5 && sbatch slurm/timing.sbatch'
# ICE -> Mac, then analysis and report
./sync.sh pull
PYTHONPATH=$PWD .venv/bin/python -m src.stats
```

## Sharp edges

- `coc-ice` QOS: 50 submitted jobs per user, array tasks count.
- `-m src.*` needs `PYTHONPATH=$PWD`. Python 3.11 only: gensim 4.3.3 has no 3.14 wheels.
- Vendored DeepWalk's `if start:` skips node 0, so ids start at 1 (`src/data.py`, `src/embed.py`).
- Only gold6226 nodes; `machine` separates `-grid` (contended, quality only) from `-exclusive` (timing tables).
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
