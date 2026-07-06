import numpy as np
from rdkit import Chem
from typing import List
from pepinvent.reinforcement.dto.scoring_input_dto import ScoringInputDTO
from pepinvent.scoring_function.score_summary import ComponentSummary
from pepinvent.scoring_function.scoring_components.base_score_component import BaseScoreComponent
from pepinvent.scoring_function.scoring_components.scoring_component_parameters import ScoringComponentParameters


class LogP(BaseScoreComponent):
    def __init__(self, parameters: ScoringComponentParameters):
        super().__init__(parameters)

    def calculate_score(self, scoring_input: ScoringInputDTO, step=-1) -> ComponentSummary:
        peptides = [p for p in scoring_input.peptides]
        score, raw_score = self._calculate_score(peptides)
        score_summary = ComponentSummary(total_score=score, parameters=self.parameters, raw_score=raw_score)
        return score_summary

    def _calculate_score(self, peptides: List[str]) -> np.array:
        scores = []

        for p in peptides:
            scores.append(Chem.Crippen.MolLogP(Chem.MolFromSmiles(p)))

        transform_params = self.parameters.specific_parameters.get(
            self.component_specific_parameters.TRANSFORMATION, {}
        )
        transformed_scores = self._transformation_function(scores, transform_params)
        return np.array(transformed_scores, dtype=np.float32), np.array(scores, dtype=np.float32)
