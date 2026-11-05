#!/bin/bash
# The selected p and q were not known when the timing job was written. Same CPU model, cores and label as timing.sbatch.
#SBATCH -J hw5-timing2
#SBATCH -p ice-cpu
#SBATCH --constraint=gold6226
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o slurm/logs/%x-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv-ice/bin/activate
export PYTHONPATH="$PWD" PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BDS_MACHINE="pace-ice-gold6226-8core"
for g in small large; do for rep in 0 1 2; do python -m src.run --graph $g --config n2v_p2_q0.5 --seed $rep --tag timing; done; done
echo done
