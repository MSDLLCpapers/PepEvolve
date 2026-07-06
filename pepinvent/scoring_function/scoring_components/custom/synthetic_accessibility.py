from typing import List

import numpy as np
from rdkit import Chem
from rdkit.Contrib.SA_Score import sascorer

from pepinvent.reinforcement.dto.scoring_input_dto import ScoringInputDTO
from pepinvent.scoring_function.score_summary import ComponentSummary
from pepinvent.scoring_function.scoring_components.base_score_component import BaseScoreComponent
from pepinvent.scoring_function.scoring_components.scoring_component_parameters import ScoringComponentParameters


class SyntheticAccessibility(BaseScoreComponent):
    def __init__(self, parameters: ScoringComponentParameters):
        super().__init__(parameters)

    def calculate_score(self, scoring_input: ScoringInputDTO, step=-1) -> ComponentSummary:
        peptide_outputs = [o for o in scoring_input.peptide_outputs]
        molecules = [x.split("|") for x in peptide_outputs]
        score, raw_score = self._calculate_score(molecules)
        score_summary = ComponentSummary(total_score=score, parameters=self.parameters, raw_score=raw_score)
        return score_summary

    def _calculate_score(self, query_mols: List[List[str]]) -> np.array:
        scores = []
        for amino_acids in query_mols:
            score = 0.0
            for aa in amino_acids:
                try:
                    score += sascorer.calculateScore(Chem.MolFromSmiles(aa))
                except:
                    score += 10.0
            scores.append(score)
        transform_params = self.parameters.specific_parameters.get(
            self.component_specific_parameters.TRANSFORMATION, {}
        )
        transformed_scores = self._transformation_function(scores, transform_params)
        return np.array(transformed_scores, dtype=np.float32), np.array(scores, dtype=np.float32)
