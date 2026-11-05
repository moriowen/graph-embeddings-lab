#!/bin/bash
# Timed runs: 8 cores on one CPU model, run one at a time, so these runs cannot inflate each other's clocks. Other users'
# jobs may share the node (a whole node sat pending for nearly 2 hours at fair-share 0.016), so the 3 repeats show that noise.
# Every timing table is drawn from this job only. sbatch slurm/timing.sbatch
#SBATCH -J hw5-timing
#SBATCH -p ice-cpu
#SBATCH --constraint=gold6226
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o slurm/logs/%x-%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv-ice/bin/activate
export PYTHONPATH="$PWD" PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BDS_MACHINE="pace-ice-gold6226-8core"
echo "node $(hostname) $(lscpu | grep 'Model name') start $(date)"
for g in small large; do
  for rep in 0 1 2; do
    for c in dw_hsns dw_ns dw_hs n2v_ns n2v_hsns dw_repo n2v_repo base_cn base_aa; do
      python -m src.run --graph $g --config $c --seed $rep --tag timing
    done
  done
done
# workers scaling (S6): large graph, walks stay as each library runs them with that many workers
for w in 2 4 8; do
  for c in dw_ns n2v_ns; do
    python -m src.run --graph large --config $c --seed 0 --tag timing --workers $w
  done
done
echo "done $(date)"
