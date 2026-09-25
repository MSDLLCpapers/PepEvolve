#!/bin/bash
# Submit all methods as separate PBS jobs

MODES=(
    "crbp_neighbor_multi:nm_pepevolve"
    "crbp_neighbor_single:ns_pepevolve"
    "crbp_self_multi:sm_pepevolve"
    "crbp_self_single:ss_pepevolve"
    "crbp_pepinvent:pepinvent"
    "traditional_ga:traditional_ga"
)

for entry in "${MODES[@]}"; do
    LABEL="${entry%%:*}"
    MODE="${entry##*:}"
    echo "Submitting: $LABEL ($MODE)"
    qsub -v LABEL="$LABEL",MODE="$MODE" -N "fps_${LABEL}" submit.sh
done
