from typing import List

import pandas as pd

from reinvent_models.mol2mol.models.vocabulary import SMILESTokenizer, Vocabulary


class DynamicVocabularyMaker:
    def __init__(self):
        self.tokenizer = SMILESTokenizer()
        self.vocabulary = Vocabulary()
        self.vocabulary.update(["*", "^", "$", "8", "?"])
    def create_vocabulary(self, path_training_set: str, path_validation_set: str) -> Vocabulary:
        self._extract_tokens(path_training_set)
        self._extract_tokens(path_validation_set)
        return self.vocabulary

    def _extract_tokens(self, path: str):
        df_iterator = pd.read_csv(path,  iterator=True, chunksize=100)
        for frame in df_iterator:
            smiles_list = self._get_batch_smiles(frame)
            self._update_vocabulary(smiles_list)

    def _get_batch_smiles(self, data: pd.DataFrame) -> List[str]:
        batch_source_smiles = list(data['Source_Mol'])
        return batch_source_smiles

    def _update_vocabulary(self, smiles_list: List[str]):
        """Creates a vocabulary for the SMILES syntax."""
        tokens = set()
        for smi in smiles_list:
            smi_tokens = set(self.tokenizer.tokenize(smi, with_begin_and_end=False))
            tokens.update(smi_tokens)

        new_tokens = [t for t in tokens if t not in self.vocabulary.tokens()]
        self.vocabulary.update(new_tokens)
