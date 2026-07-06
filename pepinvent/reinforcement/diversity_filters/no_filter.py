from copy import deepcopy
from typing import List

import numpy as np

from pepinvent.reinforcement.diversity_filters.base_diversity_filter import BaseDiversityFilter
from pepinvent.reinforcement.diversity_filters.diversity_filter_parameters import DiversityFilterParameters
from pepinvent.scoring_function.score_summary import FinalSummary
from reinvent_models.model_factory.dto.sampled_sequence_dto import SampledSequencesDTO


class NoFilter(BaseDiversityFilter):
    """Don't penalize compounds."""

    def __init__(self, parameters: DiversityFilterParameters):
        super().__init__(parameters)

    def update_score(self, score_summary: FinalSummary, sampled_sequences: List[SampledSequencesDTO], step, period, mask_pos) -> np.array:
        score_summary = deepcopy(score_summary)
        scores = score_summary.total_score
        smiles = score_summary.scored_smiles
        input_chuckles = [dto.input for dto in sampled_sequences]
        monomers = [dto.output for dto in sampled_sequences]
        for i in score_summary.valid_idxs:
            if scores[i] >= self.parameters.score_threshold:
                smile = self._chemistry.convert_to_rdkit_smiles(smiles[i])
                self._add_to_memory(i, scores[i], smile, smiles[i], score_summary.scaffold_log, step, input_chuckles=input_chuckles[i], monomers=monomers[i], period=period, mask_pos=mask_pos)
        return scores