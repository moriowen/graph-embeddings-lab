#!/bin/bash
#SBATCH -J hw5-venv
#SBATCH -p ice-cpu
#SBATCH --constraint=gold6226
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 00:20:00
#SBATCH -o slurm/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load python/3.11.9
python3 -m venv .venv-ice
.venv-ice/bin/pip -q install -r requirements.txt
.venv-ice/bin/python -c "import gensim,numpy,scipy,networkx;print(gensim.__version__,numpy.__version__,scipy.__version__,networkx.__version__)"
export PYTHONPATH="$PWD" PYTHONHASHSEED=0
.venv-ice/bin/python -m src.check
echo CHECK-OK
