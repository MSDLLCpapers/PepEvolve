#!/bin/bash
#PBS -l select=1:ncpus=8:mem=32GB
#PBS -N fps_div
#PBS -j oe

cd "$PBS_O_WORKDIR" || exit $?

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pepevolve

echo "Start: $(date '+%Y-%m-%d_%H:%M:%S')"
echo "Label: $LABEL  Mode: $MODE"

python analysis_fps_diversity.py --label "$LABEL" --mode "$MODE" --n-jobs 8

echo "End: $(date '+%Y-%m-%d_%H:%M:%S')"
