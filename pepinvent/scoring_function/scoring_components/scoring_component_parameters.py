from dataclasses import dataclass
from typing import Dict, List, Union


@dataclass
class ScoringComponentParameters:
    name: str
    weight: Union[int, List[int]]
    specific_parameters: Dict
    start_step: int = 0
