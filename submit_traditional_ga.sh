#!/bin/bash
set -e
cd "$(dirname "$0")"

CONFIG="data/manuscript/crbp/ga.json"
RUNS=(1 2 3)
SEEDS=(42 123 456)
OUTPUT_BASE="output/journal/result"
MODEL_BASE_DIR="data/models"
POOL="data/monomer_pool.pkl"

CONFIG_PATH="$(pwd)/${CONFIG}"

for j in "${!RUNS[@]}"; do
    run="${RUNS[$j]}"
    seed="${SEEDS[$j]}"
    RUN_NAME="traditional_ga${run}"
    OUTPUT_DIR="$(pwd)/${OUTPUT_BASE}/${RUN_NAME}"

    if [[ "$1" == "--local" ]]; then
        python traditional_ga.py run \
            --config "$CONFIG_PATH" \
            --output_dir "$OUTPUT_DIR" \
            --model_base_dir "$MODEL_BASE_DIR" \
            --pool "$POOL" \
            --seed "$seed"
    else
        qsub -l select=1:ncpus=4:mem=16gb \
             -N "tga${run}" -j oe \
             -v CONFIG_PATH="$CONFIG_PATH",OUTPUT_DIR="$OUTPUT_DIR",MODEL_BASE_DIR="$MODEL_BASE_DIR",POOL="$POOL",SEED="$seed" \
             job_template_ga.sh
    fi
done
