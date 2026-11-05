#!/bin/bash
# Mac -> ICE: code and built graphs. Results stay put on both sides; pull them with `./sync.sh pull`.
H=amohite8@login-ice.pace.gatech.edu
if [ "${1:-}" = pull ]; then
  rsync -az --exclude 'run_dev_*' "$H:~/scratch/HW5/results/" results/
  rsync -az "$H:~/scratch/HW5/slurm/logs/" slurm/logs/
else
  rsync -az --exclude 'data/raw' --exclude 'results/' --exclude 'embeddings/' --exclude 'slurm/logs/' \
    --exclude '__pycache__' --exclude '*.pdf' --exclude '*.zip' --exclude '.git' --exclude '.venv' --exclude '.venv-ice' \
    ./ "$H:~/scratch/HW5/"
fi
