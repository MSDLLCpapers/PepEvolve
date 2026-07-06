import argparse
import os
import re
from typing import List

from openbabel import openbabel
import pandas as pd
from rdkit import Chem


class Converter : 
    def __init__(self, path : str) : 
        self.monomer_df = pd.read_csv(path).set_index('symbol')
    
    def get_monomers(self, helm : str) -> List[str] : 
        # Extract the substring between 'PEPTIDE1' and '$PEPTIDE1'
        start_index = helm.index('PEPTIDE1{') + len('PEPTIDE1{')
        end_index = helm.index('}$PEPTIDE1')
        substring = helm[start_index:end_index]

        # Split the substring by '.' and remove brackets if present
        monomers = substring.split('.')
        monomer_list = []
        for mono in monomers:
            if mono.startswith('[') and mono.endswith(']'):
                monomer_list.append(mono[1:-1])
            else : 
                monomer_list.append(mono)
        return monomer_list
    
    def get_connections(self, helm: str) -> List[dict] : 
        connection_list = helm.split('$')[1].split('|')
        monomer_indices, r_indices = [], []
        for c in connection_list : 
            monomer_idx, r_idx = self._extract_from_connection_list(c)
            monomer_indices.append(monomer_idx)
            r_indices.append(r_idx)
        return monomer_indices, r_indices
    
    def _replace(self, x, dic) : 
        out = x 
        for k, v in dic.items() : 
            out = out.replace(k, v)
        return out 
    def _count_atoms(self, smiles: str) -> int : 
        try : return Chem.MolFromSmiles(smiles).GetNumAtoms() + smiles.count('[H]')
        except : return 0 
    def _count_ring(self, smiles: str) -> int : 
        try:
            mol = Chem.MolFromSmiles(smiles) 
            ring_info = mol.GetRingInfo()
            return len(ring_info.AtomRings())
        except:
            return 0
        
    def _rearrange(self, smiles, n_term_idx, c_term_idx) : 
        rearranger = openbabel.OBConversion()
        rearranger.SetInAndOutFormats('smi', 'smi')
        rearranger.AddOption('f', openbabel.OBConversion.OUTOPTIONS, str(n_term_idx))
        rearranger.AddOption('l', openbabel.OBConversion.OUTOPTIONS, str(c_term_idx))
        outmol = openbabel.OBMol()
        rearranger.ReadString(outmol, smiles)
        rearranged_smiles = rearranger.WriteString(outmol).strip()
        return rearranged_smiles 

    def _extract_from_connection_list(self, connection: str) -> dict :
        pattern = r"PEPTIDE1,PEPTIDE1,(\d+):R(\d)+-(\d+):R(\d+)"
        match = re.search(pattern, connection)
        if match:
            start_monomer_idx = int(match.group(1)) - 1
            start_R =  int(match.group(2))
            end_monomer_idx = int(match.group(3)) - 1
            end_R =  int(match.group(4))
            return [start_monomer_idx, end_monomer_idx], [start_R, end_R]
        else:
            print("Pattern not found in the HELM string.")

    def _insert_num(self, original_string: str, index: int, num: int) -> str : 
        if index < 0 or index > len(original_string):
            raise IndexError("Index out of bounds")
        return original_string[:index] + num + original_string[index:]

    def _fill(self, chuckles_list: List[str], r_list, monomer_indices, r_indices) -> List[str] : 
        replace_dic = {'R1': 'Ga',
                       'R2': 'Ge',
                       'R3': 'Te',
                       'R4': 'Se',
                       'R5': 'He'}
        out_chuckles_list = chuckles_list 

        for connecting_monomer_pair, connecting_r_pair in zip(monomer_indices, r_indices) : 
            for monomer_idx, r_idx in zip(connecting_monomer_pair, connecting_r_pair) : 
                out_chuckles_list[monomer_idx] = self._replace(chuckles_list[monomer_idx], {f'R{r_idx}': replace_dic[f'R{r_idx}']})

        for i, (monomer, rs) in enumerate(zip(chuckles_list, r_list)) : 
            for r in [3, 4, 5] : 
                monomer = monomer.replace(f'R{r}', str(rs[r-1]))

            n_term_idx = 1 
            c_term_idx = self._count_atoms(monomer) - 2
            monomer = self._rearrange(monomer, n_term_idx, c_term_idx)
            chuckles_list[i] = self._replace(monomer, {'Te': 'R3',
                                                 'Se': 'R4',
                                                 'He': 'R5',
                                                 'C(=O)(O)': 'C(=O)O'})
        return chuckles_list

    def _connect(self, chuckles_list, closure_num, monomer_indices, r_indices) : 
        closure_num_copy = closure_num 
        for connecting_monomer_pair, connecting_r_pair in zip(monomer_indices, r_indices) : 
            (start_monomer_idx, end_monomer_idx), (start_r, end_r) = connecting_monomer_pair, connecting_r_pair
            start_monomer, end_monomer = chuckles_list[start_monomer_idx], chuckles_list[end_monomer_idx]
            if start_monomer_idx == 0 and end_monomer_idx == len(chuckles_list) - 1: 
                start_monomer = self._insert_num(start_monomer, 1, str(closure_num_copy)) # Insert number right after N-terminus
                end_monomer = self._insert_num(end_monomer, len(end_monomer), str(closure_num_copy)) # Insert number right after C-terminus
            else :  
                start_monomer = start_monomer.replace(f'[R{start_r}]', str(closure_num_copy))
                end_monomer = end_monomer.replace(f'[R{end_r}]', str(closure_num_copy))
            
            closure_num_copy += 1
            chuckles_list[start_monomer_idx] = start_monomer
            chuckles_list[end_monomer_idx] = end_monomer
        return chuckles_list
            

    def _process_helm(self, helm: str) -> List[str] : 
        monomer_list = self.get_monomers(helm)

        monomer_data = self.monomer_df.loc[monomer_list]

        complete_smiles_list = monomer_data['Complete_SMILES'].tolist()
        r_list = monomer_data[['r1', 'r2', 'r3', 'r4', 'r5']].values.tolist()
        chuckles_list = monomer_data['CHUCKLES'].tolist()

        closure_num = max([self._count_ring(s) for s in complete_smiles_list]) + 1 # To not collide with the other ring number

        monomer_indices, r_indices = self.get_connections(helm) 

        chuckles_list = self._fill(chuckles_list, r_list, monomer_indices, r_indices)

        chuckles_list = [c[:-1] for c in chuckles_list] # Clean up the last oxygen 
        
        chuckles_list = self._connect(chuckles_list, closure_num, monomer_indices, r_indices)

        return chuckles_list
    
    def helm2smiles(self, helm: str, canonicalize : bool =False) -> str : 
        chuckles_list = self._process_helm(helm)
        smiles = ''.join(chuckles_list)
        if canonicalize : 
            return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
        else : 
            return smiles
    
    def helm2chuckles(self, helm: str, mask: List[int] = None) -> str: 
        chuckles_list = self._process_helm(helm) 
        if mask != None :
            for m in mask : 
                chuckles_list[m] = '?'
        return '|'.join(chuckles_list)
    

parser = argparse.ArgumentParser() 
parser.add_argument('--helm', '-i', type=str, required=True) 
parser.add_argument('--mask', '-m', type=int, nargs='+', help='List of index for masking', default=None)
parser.add_argument('--canonicalize', '-c', type=bool, default=True)
parser.add_argument(
    '--monomer-csv',
    '-p',
    type=str,
    default=os.environ.get('PEPEVOLVE_MONOMER_CSV'),
    help='Path to monomer CSV with CHUCKLES definitions (or set PEPEVOLVE_MONOMER_CSV).',
)
args = parser.parse_args() 

if not args.monomer_csv:
    parser.error('Missing --monomer-csv (or PEPEVOLVE_MONOMER_CSV).')

converter = Converter(path=args.monomer_csv)

if args.mask : 
    chuckles = converter.helm2chuckles(helm=args.helm, mask=args.mask) 
    print(f'\n\033[1;4mCHUCKLES\033[0m: {chuckles}\n')
else : 
    chuckles = converter.helm2chuckles(helm=args.helm) 
    smiles = converter.helm2smiles(helm=args.helm, canonicalize=args.canonicalize) 
    print(f'\n\033[1;4mCHUCKLES\033[0m: {chuckles}\n')
    print(f'\033[1;4mSMILES\033[0m: {smiles}\n')