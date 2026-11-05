#!/bin/bash
# One line of a job list per array task. From ~/scratch/HW5:
#   sbatch --array=0-N%60 slurm/array.sbatch slurm/jobs/<list>.txt
# One core per task on one CPU model, so every record carries the same machine label.
#SBATCH -J hw5
#SBATCH -p ice-cpu
#SBATCH --constraint=gold6226
#SBATCH -c 1
#SBATCH --mem=12G
#SBATCH -t 03:00:00
#SBATCH -o slurm/logs/%x-%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv-ice/bin/activate
export PYTHONPATH="$PWD" PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BDS_MACHINE="pace-ice-gold6226-grid"
first=$(( SLURM_ARRAY_TASK_ID + 1 ))
cmd=$(sed -n "${first}p" "$1")
echo "node $(hostname) start $(date) cmd: $cmd"
eval "$cmd" < /dev/null
echo "done $(date)"
