# coding=utf-8

"""
Implementation of a SMILES dataset.
"""
from typing import List, Tuple
import random 
import re 
from rdkit import Chem
from collections import Counter

import pandas as pd
import torch
from torch import Tensor
from torch.autograd import Variable
import torch.utils.data as tud

from reinvent_models.mol2mol.models.module.subsequent_mask import subsequent_mask


class DynamicMaskShiftDataset(tud.Dataset):
    """Custom PyTorch Dataset that takes a file containing"""

    def __init__(self, data, vocabulary, tokenizer):
        """
        :param data: dataframe read from training, validation or test file
        :param vocabulary: used to encode source/target tokens
        :param tokenizer: used to tokenize source/target smiles
        """
        self._vocabulary = vocabulary
        self._tokenizer = tokenizer
        self._data = data

    def _shift_chuckles(self, chuckles: str):
        def _extract_unpaired_num(sequence_part: str) -> str:
            # Find all ring numbers (single digits or %XX format)
            ring_numbers = re.findall(r'%\d+|\d', sequence_part)
            
            # Count occurrences of each ring number
            counts = Counter(ring_numbers)
            
            # Find the ring number that appears an odd number of times
            for ring_num, count in counts.items():
                if count % 2 != 0:  # Odd count indicates unpaired ring
                    return ring_num
            
            return ''
        
        # Split the sequence into individual units
        chuckles_list = chuckles.split('|')
        shift = random.randint(0, len(chuckles_list)-1)
        
        # Extract unpaired ring numbers from first and last units
        # These represent the cyclization points of the peptide
        first_number = _extract_unpaired_num(chuckles_list[0])
        last_number = _extract_unpaired_num(chuckles_list[-1])
        
        # If ring numbers don't match, this isn't a proper cyclic peptide
        if first_number != last_number:
            first_number, last_number = '', ''
        
        # Remove ring numbers from terminal units before shifting
        chuckles_list[0] = chuckles_list[0].replace(first_number, '')
        chuckles_list[-1] = chuckles_list[-1].replace(first_number, '')
        
        # Perform the cyclic shift (wraps around list length)
        shift = shift % len(chuckles_list)
        chuckles_shifted = chuckles_list[-shift:] + chuckles_list[:-shift]
        
        # Restore ring numbers to new terminal positions
        if first_number:
            # Add ring number after first character of new first unit
            chuckles_shifted[0] = (chuckles_shifted[0][:1] + 
                                first_number + 
                                chuckles_shifted[0][1:])
            
            # Add ring number to carbonyl group in new last unit
            chuckles_shifted[-1] = re.sub(r'C\(=O\)$', 
                                        f'C{last_number}(=O)', 
                                        chuckles_shifted[-1])
        
        # Validate the shifted sequence using RDKit
        combined_smiles = ''.join(chuckles_shifted)
        if Chem.MolFromSmiles(combined_smiles):
            return '|'.join(chuckles_shifted)
        else:
            return chuckles

    def _mask_chuckles(self, chuckles: str):
        masked_monomers = []
        monomers = chuckles.split('|')
        num_mask = round(random.triangular(1, len(monomers)*0.4, 0))
        mask_index = random.sample(range(0, len(monomers)), num_mask) 

        for i in mask_index: 
            masked_monomers.append(monomers[i])
            monomers[i] = '?'

        return '|'.join(monomers), '|'.join(masked_monomers)


    def __getitem__(self, i):
        """
        Tokenize and encode source smile and/or target smile
        :param i:
        :return:
        """
        row = self._data.iloc[i]

        # tokenize and encode source smiles
        full_chuckles = row['Source_Mol']
        full_chuckles = self._shift_chuckles(full_chuckles)
        source_smi, target_smi = self._mask_chuckles(full_chuckles)

        source_tokens = []
        source_tokens.extend(self._tokenizer.tokenize(source_smi))
        source_encoded = self._vocabulary.encode(source_tokens)

        # target_smi = row['Target_Mol']
        target_tokens = self._tokenizer.tokenize(target_smi)
        target_encoded = self._vocabulary.encode(target_tokens)
        return torch.tensor(source_encoded, dtype=torch.long), torch.tensor(target_encoded, dtype=torch.long), row

    def __len__(self):
        return len(self._data)

    @classmethod
    def collate_fn(cls, input_data):
        # sort based on source sequence's length
        input_data.sort(key=lambda x: len(x[0]), reverse=True)
        source_encoded, target_encoded, data = zip(*input_data)
        data = pd.DataFrame(data)

        # maximum length of source sequences
        max_length_source = max([seq.size(0) for seq in source_encoded])
        # padded source sequences with zeroes
        collated_arr_source = torch.zeros(len(source_encoded), max_length_source, dtype=torch.long)

        for i, seq in enumerate(source_encoded):
            collated_arr_source[i, :seq.size(0)] = seq
        # length of each source sequence
        source_length = [seq.size(0) for seq in source_encoded]
        source_length = torch.tensor(source_length)
        # mask of source seqs
        src_mask = (collated_arr_source !=0).unsqueeze(-2)

        # target seq
        max_length_target = max([seq.size(0) for seq in target_encoded])
        collated_arr_target = torch.zeros(len(target_encoded), max_length_target, dtype=torch.long)
        for i, seq in enumerate(target_encoded):
            collated_arr_target[i, :seq.size(0)] = seq

        trg_mask = (collated_arr_target != 0).unsqueeze(-2)
        trg_mask = trg_mask & Variable(subsequent_mask(collated_arr_target.size(-1)).type_as(trg_mask))
        trg_mask = trg_mask[:, :-1, :-1]  # save start token, skip end token

        return collated_arr_source, source_length, collated_arr_target, src_mask, trg_mask, max_length_target, data


def _mask_batch(encoded_seqs: List) -> Tuple[Tensor, Tensor]:
    """Pads a batch.

    :param encoded_seqs: A list of encoded sequences.
    :return: A tensor with the sequences correctly padded and masked
    """
    # maximum length of input sequences
    max_length_source = max([seq.size(0) for seq in encoded_seqs])
    # padded source sequences with zeroes
    collated_arr_seq = torch.zeros(len(encoded_seqs), max_length_source, dtype=torch.long, device='cpu')
    seq_mask = torch.zeros(len(encoded_seqs), 1, max_length_source, dtype=torch.bool, device='cpu')

    for i, seq in enumerate(encoded_seqs):
        collated_arr_seq[i, : len(seq)] = seq
        seq_mask[i, 0, : len(seq)] = True
    return collated_arr_seq, seq_mask

