#!/bin/bash
# Free claims whose result never appeared (timed-out or failed runs), when no pool job is running. From ~/scratch/HW5:
#   bash slurm/requeue.sh slurm/jobs/all.txt
source .venv-ice/bin/activate
while IFS= read -r cmd; do
  [ -z "$cmd" ] && continue
  id=$(printf '%s' "$cmd" | md5sum | cut -d' ' -f1)
  g=$(sed -E 's/.*--graph ([^ ]+).*/\1/' <<<"$cmd"); c=$(sed -E 's/.*--config ([^ ]+).*/\1/' <<<"$cmd")
  s=$(sed -E 's/.*--seed ([0-9]+).*/\1/' <<<"$cmd"); k=$(sed -E 's/.*--split ([0-9]+).*/\1/' <<<"$cmd")
  [ -e "results/run_${g}_${c}_split${k}_s${s}.json" ] || rmdir "slurm/claims/$id" 2>/dev/null
done < "$1"
