#!/bin/bash
# Part 2 analyses that read vectors; the large embeddings stay on ICE. sbatch slurm/part2.sbatch
#SBATCH -J hw5-part2
#SBATCH -p ice-cpu
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -o slurm/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv-ice/bin/activate
export PYTHONPATH="$PWD" PYTHONHASHSEED=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python -m src.part2 small > /dev/null
python -m src.part2 large > /dev/null
python -m src.part2 cross > /dev/null
echo done
