#!/bin/bash
# Set BASE_DIR to the directory where this script is located
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

python input_to_reinforcement_learning.py $BASE_DIR/data/manuscript/ablation/cs_topk/neighbor_multi.json nm_cs_topk1
python input_to_reinforcement_learning.py $BASE_DIR/data/manuscript/ablation/cs_topk/neighbor_multi.json nm_cs_topk2
python input_to_reinforcement_learning.py $BASE_DIR/data/manuscript/ablation/cs_topk/neighbor_multi.json nm_cs_topk3

python input_to_reinforcement_learning.py $BASE_DIR/data/manuscript/ablation/cs_topk/neighbor_single.json ns_cs_topk1
python input_to_reinforcement_learning.py $BASE_DIR/data/manuscript/ablation/cs_topk/neighbor_single.json ns_cs_topk2
python input_to_reinforcement_learning.py $BASE_DIR/data/manuscript/ablation/cs_topk/neighbor_single.json ns_cs_topk3

python input_to_reinforcement_learning.py $BASE_DIR/data/manuscript/ablation/cs_topk/self_multi.json sm_cs_topk1
python input_to_reinforcement_learning.py $BASE_DIR/data/manuscript/ablation/cs_topk/self_multi.json sm_cs_topk2
python input_to_reinforcement_learning.py $BASE_DIR/data/manuscript/ablation/cs_topk/self_multi.json sm_cs_topk3

python input_to_reinforcement_learning.py $BASE_DIR/data/manuscript/ablation/cs_topk/self_single.json ss_cs_topk1
python input_to_reinforcement_learning.py $BASE_DIR/data/manuscript/ablation/cs_topk/self_single.json ss_cs_topk2
python input_to_reinforcement_learning.py $BASE_DIR/data/manuscript/ablation/cs_topk/self_single.json ss_cs_topk3