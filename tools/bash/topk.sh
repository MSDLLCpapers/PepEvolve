#!/bin/bash
# Set BASE_DIR to the directory where this script is located
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

python input_to_reinforcement_learning.py $BASE_DIR/data/manuscript/ablation/topk/neighbor_multi.json nm_topk1
python input_to_reinforcement_learning.py $BASE_DIR/data/manuscript/ablation/topk/neighbor_multi.json nm_topk2
python input_to_reinforcement_learning.py $BASE_DIR/data/manuscript/ablation/topk/neighbor_multi.json nm_topk3

python input_to_reinforcement_learning.py $BASE_DIR/data/manuscript/ablation/topk/neighbor_single.json ns_topk1
python input_to_reinforcement_learning.py $BASE_DIR/data/manuscript/ablation/topk/neighbor_single.json ns_topk2
python input_to_reinforcement_learning.py $BASE_DIR/data/manuscript/ablation/topk/neighbor_single.json ns_topk3

python input_to_reinforcement_learning.py $BASE_DIR/data/manuscript/ablation/topk/self_multi.json sm_topk1
python input_to_reinforcement_learning.py $BASE_DIR/data/manuscript/ablation/topk/self_multi.json sm_topk2
python input_to_reinforcement_learning.py $BASE_DIR/data/manuscript/ablation/topk/self_multi.json sm_topk3

python input_to_reinforcement_learning.py $BASE_DIR/data/manuscript/ablation/topk/self_single.json ss_topk1
python input_to_reinforcement_learning.py $BASE_DIR/data/manuscript/ablation/topk/self_single.json ss_topk2
python3 input_to_reinforcement_learning.py $BASE_DIR/data/manuscript/ablation/topk/self_single.json ss_topk3