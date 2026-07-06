import json
import pandas as pd 
from sys import argv

from manager import Manager
from pepinvent.sampling.sampling_config import SamplingConfig


def read_json_file(path):
    with open(path) as f:
        json_input = f.read().replace('\r', '').replace('\n', '')
    try:
        return json.loads(json_input)
    except (ValueError, KeyError, TypeError) as e:
        print(f"JSON format error in file ${path}: \n ${e}")

def fill_source_peptide(source: str, target: str) -> str:
    source = source.split('|')
    target_merge = target.split('|')
    indices = [idx for idx, s in enumerate(source) if s == '?']
    mask_count = source.count('?')
    new_target = []
    m = 0
    if len(target_merge) == mask_count:
        for idx, s in enumerate(source):
            if idx in indices:
                t = target_merge[m]
                new_target.append(t)
                m += 1
            else:
                new_target.append(s)
        new_target = [aa for aa in new_target[:-1]] + [new_target[-1]]
        new_target = '|'.join(new_target)
    else:
        new_target = 'none'
    return new_target


if __name__ == "__main__":
    path = argv[1]

    config = read_json_file(path)
    sampling_parameters = SamplingConfig.parse_obj(config)
    manager = Manager(sampling_parameters)
    manager.execute()

    results_path, num_samples = config['results_output'], config['num_samples']
    results_df = pd.read_csv(results_path)

    results_dic = {}

    for _, row in results_df.iterrows(): 
        input = row['Input']
        results_dic[input] = []
        for num in range(1, num_samples+1):
            results_dic[input].append(fill_source_peptide(input, row[f'Generated_smi_{num}']))

    validity_count = {}
    uniqueness_count = {}

    for input, peptides in results_dic.items():
        valid_peptides = [p for p in peptides if p != 'none']
        unique_peptides = set(valid_peptides)

        validity_percentage = (len(valid_peptides) / num_samples) * 100
        uniqueness_percentage = (len(unique_peptides) / num_samples) * 100

        validity_count[input] = validity_percentage
        uniqueness_count[input] = uniqueness_percentage

    print("Validity Percentage per Input:")
    for input, percentage in validity_count.items():
        print(f"{input}: {percentage:.2f}%")

    print("\nUniqueness Percentage per Input:")
    for input, percentage in uniqueness_count.items():
        print(f"{input}: {percentage:.2f}%")