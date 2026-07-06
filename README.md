# PepEVOLVE

**PepEVOLVE** is a reinforcement learning framework for the multi-position optimization of peptide sequences. It combines a pretrained Mol2Mol generative model with a router-learner and a group-relative policy optimization (GRPO) strategy to iteratively evolve peptide candidates toward user-defined physicochemical or biological objectives.

---

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Running a Reinforcement Learning Experiment](#running-a-reinforcement-learning-experiment)
  - [Entry Point](#entry-point)
  - [Configuration File](#configuration-file)
    - [Top-Level Fields](#top-level-fields)
    - [`learning_configuration`](#learning_configuration)
    - [`scoring_function`](#scoring_function)
    - [`logging`](#logging)
    - [`diversity_filter`](#diversity_filter)
- [Experiment Configurations Used in the Paper](#experiment-configurations-used-in-the-paper)
  - [Router Experiment (`data/manuscript/router/`)](#router-experiment-datamanuscriptrouter)
  - [Main CRBP Experiment (`data/manuscript/crbp/`)](#main-crbp-experiment-datamanuscriptcrbp)
  - [Ablation Study (`data/manuscript/ablation/`)](#ablation-study-datamanuscriptablation)
- [Outputs](#outputs)
- [Citation](#citation)

---

## Overview

PepEVOLVE optimizes cyclic and linear peptides through a two-phase reinforcement learning loop. In the **routing phase**, a learnable router identifies which residue positions are most productive to modify. In the **evolving phase**, a Mol2Mol Transformer agent is fine-tuned to generate high-scoring variants at those positions, guided by a user-defined composite scoring function.

---

## Requirements

| Dependency | Version |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.2.0 |
| RDKit | 2023.9.2 |
| NumPy | 1.26.2 |
| Pandas | 2.2.2 |
| Pydantic | 2.10.5 |
| TensorBoard | 2.15.1 |
| reinvent-chemistry | 0.0.51 |
| scikit-learn | 1.8.0 |

Full dependency list is available in [`requirements.txt`](requirements.txt).

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/MSDLLCpapers/PepEVOLVE.git
cd PepEVOLVE

# 2. Create and activate a conda environment
conda create -n pepevolve python=3.12.3
conda activate pepevolve

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install the package
pip install -e .
```

> **GPU note:** CUDA 12.1 is recommended. The framework automatically detects GPU availability and moves tensors accordingly.

---

## Repository Structure

```
PepEVOLVE/
├── input_to_reinforcement_learning.py   # Main RL entry point
├── input_to_sampling.py                 # Sampling entry point
├── input_to_training.py                 # Supervised training entry point
├── manager.py                           # Orchestrates run types
├── requirements.txt
├── setup.py
├── data/
│   ├── manuscript/
│   │   ├── router/                      # Router experiment configs (LogP)
│   │   ├── crbp/                        # Main CRBP experiment configs
│   │   └── ablation/                    # Ablation study configs
│   ├── models/
│   │   ├── dynamic_mask_shift_36.ckpt   # Pretrained Mol2Mol checkpoint (used in paper)
│   │   ├── predictive_model.pckl        # CRBP classifier
│   │   └── feature_scalar.pckl          # Feature scaler for CRBP classifier
│   └── experiment_configurations/       # Additional example configs
├── pepinvent/
│   ├── reinforcement/                   # Core RL loop and router
│   ├── scoring_function/                # Scoring components
│   └── reinvent_logging/
└── reinvent_models/                     # Mol2Mol Transformer model factory
```

---

## Running a Reinforcement Learning Experiment

### Entry Point

All reinforcement learning experiments are launched through `input_to_reinforcement_learning.py`:

```bash
python input_to_reinforcement_learning.py <config_path> <run_name>
```

| Argument | Description |
|---|---|
| `<config_path>` | Path to a JSON configuration file describing the experiment. |
| `<run_name>` | A unique string identifier for this run. It is automatically appended to both `logging_path` and `result_path` defined in the config, so each run writes to its own isolated output directory. |

**Example:**

```bash
python input_to_reinforcement_learning.py \
    data/manuscript/crbp/self_single.json \
    crbp_self_single_run1
```

This writes logs to `<logging_path>/crbp_self_single_run1/` and results to `<result_path>/crbp_self_single_run1/`.

Total wall-clock runtime is saved to `<logging_path>/<run_name>/execution_time.txt` upon completion.

---

### Configuration File

The experiment is fully described by a single JSON file. The top-level structure is:

```json
{
    "name":                  "...",
    "model_type":            "mol2mol",
    "model_path":            "/path/to/checkpoint.ckpt",
    "input_sequence":        "SMILES_fragment_1|SMILES_fragment_2|...",
    "learning_configuration": { ... },
    "scoring_function":      { ... },
    "logging":               { ... },
    "diversity_filter":      { ... }
}
```

---

#### Top-Level Fields

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Experiment label used in logging output. |
| `model_type` | `str` | Generative model type. Use `"mol2mol"`. |
| `model_path` | `str` | Absolute path to the pretrained generative model checkpoint (`.ckpt`). The same checkpoint initializes both the trainable **agent** and the frozen **prior**. The checkpoint used in the paper is `data/models/dynamic_mask_shift_36.ckpt`. |
| `input_sequence` | `str` | The peptide to optimize, encoded as a `\|`-delimited sequence of per-residue SMILES fragments. Positions to be mutated are replaced with `?`. See the experiment configs in `data/manuscript/` for examples. |

---

#### `learning_configuration`

Controls the two-phase optimization loop. The two phases — **routing** and **evolving** — are governed independently and can each be disabled by setting their step count to `0`.

| Parameter | Type | Description |
|---|---|---|
| `routing_steps` | `int` | Number of routing phase steps. The router learns which positions to modify. Set to `0` to skip routing and use `positions` directly. |
| `evolving_steps` | `int` | Number of evolving phase steps. The agent is fine-tuned at the selected positions. Set to `0` to run routing only. |
| `num_positions` | `int` | Number of positions to select and optimize simultaneously. |
| `positions` | `list` | Zero-indexed list of positions to use when routing is skipped (`routing_steps: 0`). Ignored if routing is active. Example: `[2, 3, 7, 8]`. |
| `num_samples` | `int` | Number of distinct mask configurations sampled per routing step. Only used when `routing_steps > 0`. |
| `num_gen_per_sample` | `int` | Number of sequences generated per masked input during routing. Only used when `routing_steps > 0`. |
| `batch_size` | `int` | Number of sequences generated per input during the evolving phase. |
| `learning_rate` | `float` | Adam optimizer learning rate for the generative agent. |
| `score_multiplier` | `int` | Scales the score signal in the loss. Higher values push the agent more aggressively toward high-scoring regions. |
| `distance_threshold` | `float` | Log-likelihood threshold below which a prior-distance penalty is applied, preventing the agent from drifting too far from the pretrained model. Typical values: `-20` to `-30`. |
| `K` | `int` | Size of the top-K candidate pool maintained during the evolving phase. At each step, `K × G` sequences are generated from the pool. |
| `G` | `int` | Number of sequences generated per top-K candidate. Within each group of `G`, scores are normalized before the policy update (GRPO). |
| `evolve_mode` | `str` | `"single"`: one shared agent is used for all positions (memory efficient). `"multi"`: a separate agent is maintained per position (allows position-specific specialization). |
| `agent_context_visibility` | `str` | Controls what the agent sees as context when generating a residue. `"self_mask"`: the agent's own position is masked in the input (the agent generates without seeing itself). `"neighbor_mask"`: neighboring positions are masked instead. |
| `use_group_relative_advantage` | `bool` | Enables GRPO-style normalization: scores within each group of `G` are mean-centered before the policy update. Defaults to `true`. |
| `num_original` | `int` | Number of copies of the original (fully masked) input appended to the evolving batch each step. Useful for anchoring exploration. Defaults to `0`. |
| `router_learning_rate` | `float` | Adam optimizer learning rate for the router. Only used when `routing_steps > 0`. |
| `router_baseline_weight` | `float` | Exponential moving average weight for the REINFORCE baseline used in the router update. Only used when `routing_steps > 0`. |
| `router_entropy_frac` | `float` | Fraction of routing steps over which entropy regularization is linearly annealed from `router_entropy_start` to `router_entropy_end`. |
| `router_entropy_start` | `float` | Initial entropy regularization coefficient for the router (encourages exploration early in routing). |
| `router_entropy_end` | `float` | Final entropy regularization coefficient (typically `0.0`, i.e., no regularization after annealing). |

---

#### `scoring_function`

Defines the objective used to score generated peptides.

```json
"scoring_function": {
    "scoring_function": "geometric_mean",
    "scoring_components": [
        {
            "name": "predictive_model",
            "weight": 3,
            "specific_parameters": {
                "transformation": {"transformation_type": "no_transformation"},
                "model_path": "/path/to/predictive_model.pckl",
                "scalar_path": "/path/to/feature_scalar.pckl"
            }
        },
        {
            "name": "lipophilicity",
            "weight": 1,
            "specific_parameters": {
                "transformation": {"transformation_type": "reverse_sigmoid", "low": -5, "high": 0, "k": 0.5}
            }
        }
    ]
}
```

| Field | Description |
|---|---|
| `scoring_function` | Aggregation rule across components. Use `"geometric_mean"`. |
| `scoring_components` | List of individual scoring components. Each component contributes a score in `[0, 1]`, optionally transformed and weighted before aggregation. |

Each component has:
- `name`: the component identifier (see available components below).
- `weight`: relative weight in the geometric mean aggregation. A component with `weight: 3` has three times the influence of a component with `weight: 1`.
- `specific_parameters`: component-specific settings, always including a `transformation` block.

**Available scoring components:**

| `name` | Description |
|---|---|
| `predictive_model` | An external scikit-learn classifier. Requires `model_path` (`.pckl`) and `scalar_path` (`.pckl`) in `specific_parameters`. Returns predicted class probability. |
| `logp` | Lipophilicity (LogP) computed via RDKit. |
| `lipophilicity` | RDKit-based lipophilicity (alias of logp in physchem module). |
| `hbd` | Hydrogen bond donor count. |
| `synthetic_accessibility` | Synthetic accessibility (SA) score. |
| `mol_weight` | Molecular weight. |
| `maximum_ring_size` | Maximum ring size in the molecule. |
| `custom_alerts` | Penalizes structures matching any provided SMARTS patterns. Provide `smarts` as a list of strings in `specific_parameters`. |
| `matching_substructure` | Rewards presence of a target substructure. Provide the SMARTS in `specific_parameters`. |

**Transformation types** (specified under `specific_parameters.transformation`):

| `transformation_type` | Parameters | Effect |
|---|---|---|
| `no_transformation` | — | Raw score passed through unchanged. |
| `sigmoid` | `low`, `high`, `k` | Maps score to `[0, 1]` with a sigmoid centered between `low` and `high`. |
| `reverse_sigmoid` | `low`, `high`, `k` | Same as sigmoid but inverted (rewards lower values). |
| `double_sigmoid` | `low`, `high`, `coef_div`, `coef_si`, `coef_se` | Bell-shaped; rewards values in a target range `[low, high]`. |

---

#### `logging`

| Field | Type | Description |
|---|---|---|
| `logging_path` | `str` | Base directory for TensorBoard logs and router visualizations. The `<run_name>` CLI argument is appended automatically by the script. Use a path relative to the repository root (the script prepends the absolute repo path). Example: `"/output/journal/logging/"`. |
| `result_path` | `str` | Base directory for CSV result files. The `<run_name>` CLI argument is appended automatically. Example: `"/output/journal/result/"`. |

---

#### `diversity_filter`

Prevents mode collapse by penalizing over-represented scaffolds.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | `"NoFilter"` | Filter strategy. `"NoFilter"`: no filtering. `"NoFilterWithPenalty"`: apply penalty without scaffold bucketing. `"IdenticalMurckoScaffold"`: penalize molecules whose Murcko scaffold has been seen too many times. |
| `score_threshold` | `float` | `0.4` | Minimum score for a molecule to be recorded in the scaffold memory. |
| `bucket_size` | `int` | `25` | Maximum number of molecules allowed per scaffold bucket before the penalty activates. |
| `similarity_threshold` | `float` | `0.4` | Tanimoto similarity threshold for grouping molecules into the same scaffold bucket. |
| `penalty` | `float` | `0.5` | Multiplicative penalty factor applied to scores of molecules from over-represented buckets. |

---

## Experiment Configurations Used in the Paper

All configurations from the paper are in `data/manuscript/`. Absolute paths in these files are set to the training environment and should be updated to match your local paths before running.

### Router Experiment (`data/manuscript/router/`)

These experiments evaluate the router's ability to identify productive positions to modify, using LogP as the optimization objective on the cyclosporin A-like peptide (luna18).

| Config file | `routing_steps` | `evolving_steps` | `num_positions` | Description |
|---|---|---|---|---|
| `logp_luna18_1pos.json` | 500 | 0 | 1 | Router only, selects 1 position |
| `logp_luna18_2pos.json` | 500 | 0 | 2 | Router only, selects 2 positions |
| `logp_luna18_2pos_adversarial.json` | 500 | 0 | 2 | Adversarial variant |
| `hbd_1pos.json` | 500 | 0 | 1 | Router only, HBD objective, 1 position |
| `hbd_2pos.json` | 500 | 0 | 2 | Router only, HBD objective, 2 positions |

**How to run:**
```bash
python input_to_reinforcement_learning.py \
    data/manuscript/router/logp_luna18_2pos.json \
    router_logp_2pos_run1
```

---

### Main CRBP Experiment (`data/manuscript/crbp/`)

These are the primary experiments optimizing a peptide for CRBP (cellular retinol-binding protein) binding, permeability, and solubility using a composite scoring function (`predictive_model` + `maximum_ring_size` + `lipophilicity` + `custom_alerts`). Routing is disabled (`routing_steps: 0`) and positions `[2, 3, 7, 8]` are fixed. The four configs explore combinations of `evolve_mode` and `agent_context_visibility`:

| Config file | `evolve_mode` | `agent_context_visibility` | Description |
|---|---|---|---|
| `self_single.json` | `single` | `self_mask` | One shared agent; each position generates without seeing itself |
| `self_multi.json` | `multi` | `self_mask` | Per-position agents; each masks its own position |
| `neighbor_single.json` | `single` | `neighbor_mask` | One shared agent; each position generates without seeing neighbors |
| `neighbor_multi.json` | `multi` | `neighbor_mask` | Per-position agents; each masks neighboring positions |

**How to run:**
```bash
python input_to_reinforcement_learning.py \
    data/manuscript/crbp/self_single.json \
    crbp_self_single_run1
```

---

### Ablation Study (`data/manuscript/ablation/`)

These experiments isolate the contribution of individual components of the framework. Each subdirectory corresponds to one ablation variant and contains the same four `evolve_mode` × `agent_context_visibility` combinations as the CRBP experiment.

| Subdirectory | Description |
|---|---|
| `topk_gra/` | Full method: top-K candidate pool + GRPO normalization (`use_group_relative_advantage: true`) |
| `topk/` | Top-K pool only, without GRPO normalization |
| `cs_topk/` | Curriculum-scheduled variant of the top-K pool |

**How to run:**
```bash
python input_to_reinforcement_learning.py \
    data/manuscript/ablation/topk_gra/self_single.json \
    ablation_topk_gra_self_single_run1
```

---

## Outputs

All outputs are written under `logging_path/<run_name>/` and `result_path/<run_name>/`.

| File | Description |
|---|---|
| `results_routing_period.csv` | Scored peptides collected during the routing phase. |
| `results_evolving_step<N>.csv` | Scored peptides at the final evolving step. |
| `execution_time.txt` | Total wall-clock runtime in seconds. |
| `router_ps.npy` | Router probability vectors saved at every step (routing phase only). |
| `router_smiles.gif` | Animated visualization of the router distribution over the peptide sequence. |
| `router_argmax.png` | Plot of the highest-probability position over routing steps. |
| `router_animation.gif` | Animated bar chart of router probabilities. |
| `router_heatmap.png` | Heatmap of router probability evolution over routing steps. |
| `router_accumulated_probs.png` | Accumulated router probabilities across all routing steps. |
| `tensorboard/` | TensorBoard event files for live monitoring of scores and likelihoods. |

---