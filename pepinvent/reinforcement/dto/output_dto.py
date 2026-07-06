from typing import Any, List

from pydantic.dataclasses import dataclass


@dataclass
class OutputDTO:
    peptide: Any
    amino_acids: List[str]
