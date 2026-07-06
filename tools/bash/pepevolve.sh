#!/bin/bash
# Set BASE_DIR to the directory where this script is located
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

python input_to_reinforcement_learning.py $BASE_DIR/data/manuscript/crbp/neighbor_multi.json nm_pepevolve1
python input_to_reinforcement_learning.py $BASE_DIR/data/manuscript/crbp/neighbor_multi.json nm_pepevolve2
python input_to_reinforcement_learning.py $BASE_DIR/data/manuscript/crbp/neighbor_multi.json nm_pepevolve3

python input_to_reinforcement_learning.py $BASE_DIR/data/manuscript/crbp/neighbor_single.json ns_pepevolve1
python input_to_reinforcement_learning.py $BASE_DIR/data/manuscript/crbp/neighbor_single.json ns_pepevolve2
python input_to_reinforcement_learning.py $BASE_DIR/data/manuscript/crbp/neighbor_single.json ns_pepevolve3

python input_to_reinforcement_learning.py $BASE_DIR/data/manuscript/crbp/self_multi.json sm_pepevolve1
python input_to_reinforcement_learning.py $BASE_DIR/data/manuscript/crbp/self_multi.json sm_pepevolve2
python input_to_reinforcement_learning.py $BASE_DIR/data/manuscript/crbp/self_multi.json sm_pepevolve3

python input_to_reinforcement_learning.py $BASE_DIR/data/manuscript/crbp/self_single.json ss_pepevolve1
python3 input_to_reinforcement_learning.py $BASE_DIR/data/manuscript/crbp/self_single.json ss_pepevolve2
python3 input_to_reinforcement_learning.py $BASE_DIR/data/manuscript/crbp/self_single.json ss_pepevolve3