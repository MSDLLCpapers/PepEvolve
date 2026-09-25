#!/bin/bash
cd "$PBS_O_WORKDIR" || exit $?
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pepevolve || exit 1
echo "Start time: $(date '+%Y-%m-%d_%H:%M:%S')"
START_TIME=$SECONDS
python traditional_ga.py run \
    --config "$CONFIG_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --model_base_dir "$MODEL_BASE_DIR" \
    --pool "$POOL" \
    --seed "$SEED" || exit 1
ELAPSED_TIME=$(($SECONDS - START_TIME))
echo "End time: $(date '+%Y-%m-%d_%H:%M:%S')"
echo "Elapsed time: $(($ELAPSED_TIME / 60))m $(($ELAPSED_TIME % 60))s"
