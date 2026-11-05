#!/bin/bash
# The coc-ice QOS allows 50 submitted jobs per user and array tasks count, so a 350-line grid cannot be an array.
# Each pool job runs one worker per core; workers claim lines of a shared list with an atomic mkdir, so several
# pool jobs can share one list. Relaunch after a timeout with slurm/requeue.sh, which frees unfinished claims.
#   sbatch slurm/pool.sbatch slurm/jobs/all.txt
#SBATCH -J hw5-pool
#SBATCH -p ice-cpu
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 06:00:00
#SBATCH -o slurm/logs/%x-%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv-ice/bin/activate
export PYTHONPATH="$PWD" PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# Any CPU model is fine for quality; the record says which one. Timing tables never use these records.
export BDS_MACHINE="pace-ice-$(lscpu | sed -nE 's/^Model name: +//p' | tr -s ' ' '-')-grid"
mkdir -p slurm/claims
list="$1"
echo "node $(hostname) cores ${SLURM_CPUS_PER_TASK} start $(date)"
worker() {
  while IFS= read -r cmd; do
    [ -z "$cmd" ] && continue
    id=$(printf '%s' "$cmd" | md5sum | cut -d' ' -f1)
    mkdir "slurm/claims/$id" 2>/dev/null || continue
    eval "$cmd" < /dev/null || echo "FAILED: $cmd"
  done < "$list"
}
for ((i = 0; i < SLURM_CPUS_PER_TASK; i++)); do worker & done
wait
echo "done $(date)"
