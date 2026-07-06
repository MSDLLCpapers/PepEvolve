from copy import deepcopy
from typing import List

import numpy as np

from pepinvent.reinforcement.diversity_filters.base_diversity_filter import BaseDiversityFilter
from pepinvent.reinforcement.diversity_filters.diversity_filter_parameters import DiversityFilterParameters
from pepinvent.scoring_function.score_summary import FinalSummary
from reinvent_models.model_factory.dto.sampled_sequence_dto import SampledSequencesDTO


class NoFilterWithPenalty(BaseDiversityFilter):
    """Don't penalize compounds."""

    def __init__(self, parameters: DiversityFilterParameters):
        super().__init__(parameters)

    def update_score(self, score_summary: FinalSummary, sampled_sequences: List[SampledSequencesDTO], step, period, mask_pos) -> np.array:
        score_summary = deepcopy(score_summary)
        scores = score_summary.total_score
        input_chuckles = [dto.input for dto in sampled_sequences]
        monomers = [dto.output for dto in sampled_sequences]
        for i, chuckles in enumerate(score_summary.scored_smiles):
            if self._chemistry.smile_to_mol(chuckles): #Checks validity
                rdkit_smile = self._chemistry.convert_to_rdkit_smiles(chuckles)
                scores[i] = self.parameters.penalty*scores[i] if self._smiles_exists(rdkit_smile) else scores[i]
                self._add_to_memory_check(i, scores, rdkit_smile, chuckles, score_summary, step, input_chuckles=input_chuckles[i], monomers=monomers[i], period=period, mask_pos=mask_pos)


        return scores

    def _add_to_memory_check(self, index: int, scores: List[float], smile: str, chuckles: str, score_summary: FinalSummary, step: int, input_chuckles: str, monomers: str, period: str, mask_pos: int):
        if scores[index] >= self.parameters.score_threshold:
            self._add_to_memory(index, scores[index], smile, chuckles, score_summary.scaffold_log, step, input_chuckles=input_chuckles, monomers=monomers, period=period, mask_pos=mask_pos)
