"""
Traditional Genetic Algorithm for peptide optimization.

Uses the monomer pool from training data with frequency-weighted mutation,
single-point crossover, and tournament selection. Reuses PepEvolve's scoring
infrastructure for fair comparison.

Only the positions specified in the config (learning_configuration.positions)
are mutated. Ring-closure syntax is programmatically added/removed when
mutating terminal positions.

Usage:
    # First time: build and cache the monomer pool
    python traditional_ga.py --build_pool --train_data data/train_data/train_filled_chuckles.txt

    # Run GA (loads cached pool)
    python traditional_ga.py --config data/manuscript/crbp/neighbor_multi.json \
                             --output_dir output/journal/result/traditional_ga1 \
                             --model_base_dir data/models \
                             --seed 42
"""

import argparse
import json
import os
import pickle
import random
import re
import sys
from collections import Counter
from typing import List, Tuple

import numpy as np
import pandas as pd

from pepinvent.reinforcement.dto.scoring_input_dto import ScoringInputDTO
from pepinvent.scoring_function.scoring_function_factory import ScoringFunctionFactory
from pepinvent.scoring_function.scoring_config import ScoringConfig

POOL_CACHE_PATH = os.path.join(os.path.dirname(__file__), 'data', 'monomer_pool.pkl')


def build_monomer_pool(train_data_path: str) -> Tuple[List[str], np.ndarray]:
    """Build frequency-weighted interior monomer pool from training data."""
    counts = Counter()
    with open(train_data_path) as f:
        for line in f:
            parts = line.strip().split('|')
            for m in parts[1:-1]:
                counts[m] += 1

    monomers = list(counts.keys())
    freqs = np.array([counts[m] for m in monomers], dtype=np.float64)
    probs = freqs / freqs.sum()
    return monomers, probs


def save_monomer_pool(monomers: List[str], probs: np.ndarray, path: str = POOL_CACHE_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump({'monomers': monomers, 'probs': probs}, f)
    print(f"Monomer pool saved to {path} ({len(monomers)} monomers)")


def load_monomer_pool(path: str = POOL_CACHE_PATH) -> Tuple[List[str], np.ndarray]:
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['monomers'], data['probs']


def detect_ring_digit(query_peptide: str) -> str:
    """Detect the macrocycle ring-closure digit from the query peptide."""
    parts = query_peptide.split('|')
    m = re.match(r'^N(\d)', parts[0])
    if m:
        return m.group(1)
    m = re.match(r'^N\(C\)(\d)', parts[0])
    if m:
        return m.group(1)
    return None


def add_ring_closure_last(monomer: str, digit: str) -> str:
    """Add ring-closure digit to a monomer placed at the last position.
    Transforms terminal C(=O) -> C{digit}(=O)."""
    if monomer.endswith('C(=O)'):
        return monomer[:-5] + f'C{digit}(=O)'
    if monomer.endswith('C(=O)O'):
        return monomer[:-6] + f'C{digit}(=O)'
    return monomer


def add_ring_closure_first(monomer: str, digit: str) -> str:
    """Add ring-closure digit to a monomer placed at the first position.
    Inserts digit after leading N or N(C)."""
    if monomer.startswith('N(C)'):
        return f'N{digit}(C)' + monomer[4:]
    if monomer.startswith('N['):
        return f'N{digit}[' + monomer[2:]
    if monomer.startswith('N(') and not monomer.startswith('N(C)'):
        return f'N{digit}(' + monomer[2:]
    if monomer.startswith('N'):
        return f'N{digit}' + monomer[1:]
    return monomer


def remap_conflicting_ring_digit(monomer: str, ring_digit: str) -> str:
    """Renumber a monomer's own ring-closure digit if it collides with the
    peptide's macrocycle ring_digit. The macrocycle digit stays open across
    the entire peptide (from position 0 to the last position), so any
    unrelated side-chain/backbone ring inside a spliced-in monomer that
    happens to reuse the same digit would incorrectly pair with it,
    truncating the macrocycle and creating a bogus extra ring closure."""
    used_digits = set(re.findall(r'[A-Za-z\]\)](\d)', monomer))
    if ring_digit not in used_digits:
        return monomer
    for candidate in '987654321':
        if candidate not in used_digits:
            return re.sub(rf'([A-Za-z\]\)]){ring_digit}', rf'\g<1>{candidate}', monomer)
    return monomer


def adapt_monomer_for_position(monomer: str, pos: int, peptide_length: int,
                               ring_digit: str) -> str:
    """Adapt an interior monomer for placement at a specific position."""
    if ring_digit is None:
        return monomer
    monomer = remap_conflicting_ring_digit(monomer, ring_digit)
    if pos == 0:
        return add_ring_closure_first(monomer, ring_digit)
    if pos == peptide_length - 1:
        return add_ring_closure_last(monomer, ring_digit)
    return monomer


def chuckles_to_smiles(chuckles: str) -> str:
    """Convert pipe-delimited CHUCKLES to concatenated SMILES for scoring."""
    return ''.join(chuckles.split('|'))


def mutate(peptide: str, monomer_pool: List[str], monomer_probs: np.ndarray,
           mutation_rate: float, mutable_positions: List[int],
           ring_digit: str) -> str:
    """Point mutation at specified positions with frequency-weighted monomers."""
    parts = peptide.split('|')
    peptide_length = len(parts)
    for i in mutable_positions:
        if random.random() < mutation_rate:
            monomer = np.random.choice(monomer_pool, p=monomer_probs)
            parts[i] = adapt_monomer_for_position(monomer, i, peptide_length, ring_digit)
    return '|'.join(parts)


def crossover(parent1: str, parent2: str, mutable_positions: List[int]) -> Tuple[str, str]:
    """Crossover: swap monomers at a subset of mutable positions."""
    p1 = parent1.split('|')
    p2 = parent2.split('|')
    if len(p1) != len(p2) or len(mutable_positions) < 2:
        return parent1, parent2
    point = random.randint(1, len(mutable_positions) - 1)
    c1 = list(p1)
    c2 = list(p2)
    for pos in mutable_positions[point:]:
        c1[pos], c2[pos] = c2[pos], c1[pos]
    return '|'.join(c1), '|'.join(c2)


def tournament_select(population: List[str], fitness: np.ndarray,
                      k: int = 3) -> str:
    """Tournament selection: pick best of k random individuals."""
    indices = random.sample(range(len(population)), min(k, len(population)))
    best = max(indices, key=lambda i: fitness[i])
    return population[best]


def score_population(population: List[str], scoring_function) -> np.ndarray:
    """Score a batch of CHUCKLES peptides using PepEvolve's scoring function."""
    peptides_smiles = [chuckles_to_smiles(p) for p in population]
    scoring_input = ScoringInputDTO(
        peptides=peptides_smiles,
        peptide_input='placeholder',
        peptide_outputs=peptides_smiles,
        chuckles=list(population),
    )
    final_summary = scoring_function.calculate_score(scoring_input)
    return final_summary.total_score


def initialize_population(query_peptide: str, pop_size: int,
                          monomer_pool: List[str], monomer_probs: np.ndarray,
                          mutable_positions: List[int],
                          ring_digit: str) -> List[str]:
    """Create initial population by mutating only the specified positions."""
    parts = query_peptide.split('|')
    peptide_length = len(parts)
    population = [query_peptide]
    for _ in range(pop_size - 1):
        new_parts = list(parts)
        n_mut = max(1, random.randint(1, len(mutable_positions)))
        positions = random.sample(mutable_positions, n_mut)
        for pos in positions:
            monomer = np.random.choice(monomer_pool, p=monomer_probs)
            new_parts[pos] = adapt_monomer_for_position(monomer, pos, peptide_length, ring_digit)
        population.append('|'.join(new_parts))
    return population


def fix_model_paths(config: dict, model_base_dir: str) -> dict:
    """Replace hardcoded model paths in config with local paths."""
    for comp in config.get('scoring_function', {}).get('scoring_components', []):
        sp = comp.get('specific_parameters', {})
        for key in ['model_path', 'scalar_path']:
            if key in sp:
                sp[key] = os.path.join(model_base_dir, os.path.basename(sp[key]))
    return config


def run_ga(config_path: str, output_dir: str, seed: int,
           pool_path: str, pop_size: int, n_generations: int,
           mutation_rate: float, crossover_rate: float, elite_frac: float,
           tournament_k: int, model_base_dir: str = None):
    random.seed(seed)
    np.random.seed(seed)

    with open(config_path) as f:
        config = json.load(f)

    if model_base_dir:
        config = fix_model_paths(config, model_base_dir)

    query_peptide = config['input_sequence']
    mutable_positions = config['learning_configuration']['positions']
    ring_digit = detect_ring_digit(query_peptide)

    scoring_config = ScoringConfig(**config['scoring_function'])
    scoring_function = ScoringFunctionFactory(scoring_config).create_scoring_function()

    parts = query_peptide.split('|')
    print(f"Query: {query_peptide}")
    print(f"  {len(parts)} monomers, mutable positions: {mutable_positions}, ring_digit: {ring_digit}")

    print(f"Loading monomer pool from {pool_path}...")
    monomer_pool, monomer_probs = load_monomer_pool(pool_path)
    print(f"  {len(monomer_pool)} unique interior monomers")

    print(f"Initializing population of {pop_size}...")
    population = initialize_population(query_peptide, pop_size, monomer_pool,
                                       monomer_probs, mutable_positions, ring_digit)
    fitness = score_population(population, scoring_function)
    print(f"  Initial best={fitness.max():.4f}, mean={fitness.mean():.4f}")

    os.makedirs(output_dir, exist_ok=True)
    n_elite = max(1, int(pop_size * elite_frac))
    all_records = []

    for gen in range(n_generations):
        elite_idx = np.argsort(fitness)[-n_elite:]
        next_gen = [population[i] for i in elite_idx]

        while len(next_gen) < pop_size:
            p1 = tournament_select(population, fitness, k=tournament_k)
            p2 = tournament_select(population, fitness, k=tournament_k)
            if random.random() < crossover_rate:
                c1, c2 = crossover(p1, p2, mutable_positions)
            else:
                c1, c2 = p1, p2
            c1 = mutate(c1, monomer_pool, monomer_probs, mutation_rate, mutable_positions, ring_digit)
            c2 = mutate(c2, monomer_pool, monomer_probs, mutation_rate, mutable_positions, ring_digit)
            next_gen.append(c1)
            if len(next_gen) < pop_size:
                next_gen.append(c2)

        population = next_gen
        fitness = score_population(population, scoring_function)

        for pep, score in zip(population, fitness):
            all_records.append({
                'Step': gen,
                'SMILES': chuckles_to_smiles(pep),
                'CHUCKLES': pep,
                'total_score': score,
            })

        if (gen + 1) % 10 == 0 or gen == 0:
            print(f"  Gen {gen+1:>4d}/{n_generations} | "
                  f"best={fitness.max():.4f} mean={fitness.mean():.4f} "
                  f"unique={len(set(population))}/{pop_size}")

    results_df = pd.DataFrame(all_records)
    out_path = os.path.join(output_dir, f'results_evolving_step{n_generations}.csv')
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
    print(f"Final best={fitness.max():.4f}")
    print(f"Total unique peptides: {results_df['SMILES'].nunique()}")


def main():
    parser = argparse.ArgumentParser(description='Traditional GA for peptide optimization')
    subparsers = parser.add_subparsers(dest='command')

    # Subcommand: build pool
    build_parser = subparsers.add_parser('build_pool', help='Build and cache monomer pool')
    build_parser.add_argument('--train_data', default='data/train_data/train_filled_chuckles.txt')
    build_parser.add_argument('--output', default=POOL_CACHE_PATH)

    # Subcommand: run
    run_parser = subparsers.add_parser('run', help='Run the GA')
    run_parser.add_argument('--config', required=True, help='Path to experiment config JSON')
    run_parser.add_argument('--output_dir', required=True, help='Output directory for results')
    run_parser.add_argument('--seed', type=int, default=42)
    run_parser.add_argument('--pool', default=POOL_CACHE_PATH, help='Path to cached monomer pool')
    run_parser.add_argument('--model_base_dir', default=None,
                            help='Override model paths in config (e.g. data/models)')
    run_parser.add_argument('--pop_size', type=int, default=128)
    run_parser.add_argument('--n_generations', type=int, default=1000)
    run_parser.add_argument('--mutation_rate', type=float, default=0.15)
    run_parser.add_argument('--crossover_rate', type=float, default=0.7)
    run_parser.add_argument('--elite_frac', type=float, default=0.1)
    run_parser.add_argument('--tournament_k', type=int, default=3)

    args = parser.parse_args()

    if args.command == 'build_pool':
        monomers, probs = build_monomer_pool(args.train_data)
        save_monomer_pool(monomers, probs, args.output)
    elif args.command == 'run':
        if not os.path.exists(args.pool):
            parser.error(f"Monomer pool not found at {args.pool}. Run 'build_pool' first.")
        run_ga(
            config_path=args.config,
            output_dir=args.output_dir,
            seed=args.seed,
            pool_path=args.pool,
            pop_size=args.pop_size,
            n_generations=args.n_generations,
            mutation_rate=args.mutation_rate,
            crossover_rate=args.crossover_rate,
            elite_frac=args.elite_frac,
            tournament_k=args.tournament_k,
            model_base_dir=args.model_base_dir,
        )
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
