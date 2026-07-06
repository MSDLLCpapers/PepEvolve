import gc
import copy
import torch
from pathlib import Path
from copy import deepcopy
from collections import Counter
from typing import List, Set, Tuple
from pepinvent.scoring_function.scoring_components.scoring_component_parameters import ScoringComponentParameters


def find_extension(search_dir: str, extension: str = ".pt") -> List[str]:
    folder = Path(search_dir)
    return list(folder.rglob(f"*{extension}"))


def clean_memory(*arg):
    for a in arg:
        del a
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def reformat_param(parameters, excluded_keys: Set[str]) -> Tuple[List[ScoringComponentParameters], dict]:
    global_sp = deepcopy(parameters.specific_parameters)
    for k in excluded_keys:
        global_sp.pop(k, None)

    local_tgts, local_tgts_sp, local_weights = global_sp.keys(), list(global_sp.values()), parameters.weight

    local_sp = []
    for tgt, sp, w in zip(local_tgts, local_tgts_sp, local_weights):
        p = deepcopy(parameters)
        p.name, p.specific_parameters, p.weight = tgt, sp, w
        local_sp.append(p)

    return local_sp, local_tgts, local_tgts_sp


def remove_unclosed_ring(smiles: str) -> str:
    digit_counts = Counter(c for c in smiles if c.isdigit())
    return "".join(c for c in smiles if not c.isdigit() or digit_counts[c] > 1)


# Archived
def get_scoring_function(self, final_summary, step):
    GLOBAL_CONDITION = ["fraction_valid", "total_score"]
    components = copy.deepcopy(self._scoring_components)

    add_idx, remove_idx = set(), set()
    for i, component in enumerate(components):
        # In step 0, only add components with no condition (doesn't have output yet to compare with condition)
        if step == 0:
            if not component.condition:
                add_idx.add(i)
            continue

        # Raise error if component has condition but miss start_step (step to run even the condition is never met)
        if component.condition and component.start_step == 0:
            raise ValueError("You need to specify the minimum start step")

        # Add component that has reached start_step or condition has been resolved
        if step >= component.start_step or component.condition == "RESOLVED":
            add_idx.add(i)
            continue

        # Has not reached start_step and has no condition -> skip checking
        if not component.condition:
            continue

        meet_local, meet_global = True, True

        # Extract and check global conditions
        valid_req = component.condition.pop(GLOBAL_CONDITION[0], 0)
        score_req = component.condition.pop(GLOBAL_CONDITION[1], 0)
        if (valid_req and self._logger._fraction_valid < valid_req) or (
            score_req and self._logger._total_score < score_req
        ):
            meet_global = False

        # Check local condition
        for name, (requirement, keep) in component.condition.items():
            matched = next(
                (log_component for log_component in final_summary.profile if log_component.name == name), None
            )
            if not matched:
                raise ValueError(f"Component '{name}' not found in final_summary.profile")
            if np.mean(matched.score) <= requirement:
                meet_local = False
                break

        # If both global and local conditions are satisfied
        if meet_local and meet_global:
            add_idx.add(i)

            # If any related components should not be kept, mark them for removal
            if component.condition:
                for name, (_, keep) in component.condition.items():
                    if not keep:
                        remove_idx.add(next(j for j, c in enumerate(components) if c.name == name))
                components[i].condition = "RESOLVED"
                self._scoring_components = [components[i] for i in add_idx - remove_idx]

    # Update the scoring function with selected components
    scoring_func = copy.deepcopy(self._config.scoring_function)
    scoring_func.scoring_components = [components[i] for i in add_idx - remove_idx]

    return ScoringFunctionFactory(scoring_func).create_scoring_function()
