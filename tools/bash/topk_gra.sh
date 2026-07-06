#!/bin/bash
# Set BASE_DIR to the directory where this script is located
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

python input_to_reinforcement_learning.py $BASE_DIR/data/manuscript/ablation/topk_gra/neighbor_multi.json nm_topk_gra1
python input_to_reinforcement_learning.py $BASE_DIR/data/manuscript/ablation/topk_gra/neighbor_multi.json nm_topk_gra2
python input_to_reinforcement_learning.py $BASE_DIR/data/manuscript/ablation/topk_gra/neighbor_multi.json nm_topk_gra3

python input_to_reinforcement_learning.py $BASE_DIR/data/manuscript/ablation/topk_gra/neighbor_single.json ns_topk_gra1
python input_to_reinforcement_learning.py $BASE_DIR/data/manuscript/ablation/topk_gra/neighbor_single.json ns_topk_gra2
python input_to_reinforcement_learning.py $BASE_DIR/data/manuscript/ablation/topk_gra/neighbor_single.json ns_topk_gra3

python input_to_reinforcement_learning.py $BASE_DIR/data/manuscript/ablation/topk_gra/self_multi.json sm_topk_gra1
python input_to_reinforcement_learning.py $BASE_DIR/data/manuscript/ablation/topk_gra/self_multi.json sm_topk_gra2
python input_to_reinforcement_learning.py $BASE_DIR/data/manuscript/ablation/topk_gra/self_multi.json sm_topk_gra3

python input_to_reinforcement_learning.py $BASE_DIR/data/manuscript/ablation/topk_gra/self_single.json ss_topk_gra1
python input_to_reinforcement_learning.py $BASE_DIR/data/manuscript/ablation/topk_gra/self_single.json ss_topk_gra2
python input_to_reinforcement_learning.py $BASE_DIR/data/manuscript/ablation/topk_gra/self_single.json ss_topk_gra3