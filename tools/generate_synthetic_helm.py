#!/usr/bin/env python3
"""
Peptide HELM String Randomization Tool

This script processes HELM (Hierarchical Editing Language for Macromolecules) notation
strings to randomly replace peptide symbols with alternatives from a monomer database.
It supports parallel processing and generates multiple variants per input sequence.

The script filters out sequences containing specific patterns (R1-R5, PEPTIDE2, CHEM, PEG)
and focuses only on PEPTIDE polymer types for randomization.
"""

import os
import re
import json
import argparse
import random
from typing import List, Dict, Optional, Tuple

import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def randomize_peptide_symbols(
    reference: str,
    db: List[Dict[str, str]],
    num_replacements: Optional[int] = None
) -> str:
    """
    Randomly replace peptide symbols in a HELM notation string.
    
    Args:
        reference: Original HELM notation string
        db: Database of monomers with 'symbol' and 'polymertype' fields
        num_replacements: Number of symbols to replace (None = replace all)
        
    Returns:
        Modified HELM string with randomized peptide symbols
        
    Raises:
        ValueError: If no PEPTIDE entries found in database
    """
    # Extract all peptide symbols from the database
    peptide_symbols = [
        monomer["symbol"] 
        for monomer in db 
        if monomer["polymertype"].upper() == "PEPTIDE"
    ]
    
    if not peptide_symbols:
        raise ValueError("No PEPTIDE entries found in database.")

    # Pattern to match HELM blocks: TAG{content}
    pattern = re.compile(r"(\w+)\{([^}]+)\}")
    token_positions = []  # Store positions of replaceable tokens
    blocks = []  # Store all parsed blocks for reconstruction
    
    # Parse all blocks and identify replaceable peptide tokens
    for block_idx, match in enumerate(pattern.finditer(reference)):
        tag, content = match.group(1), match.group(2)
        tokens = content.split('.')
        
        # Only process PEPTIDE blocks
        if tag.upper().startswith("PEPTIDE"):
            for token_idx, token in enumerate(tokens):
                # Extract symbol (remove brackets if present)
                symbol = token[1:-1] if token.startswith("[") and token.endswith("]") else token
                
                # Track position if symbol is a valid peptide
                if symbol in peptide_symbols:
                    token_positions.append((block_idx, token_idx, symbol))
        
        # Store block information for reconstruction
        blocks.append({
            "tag": tag,
            "tokens": tokens,
            "start": match.start(),
            "end": match.end()
        })

    # Return original if no replaceable tokens found
    if not token_positions:
        return reference

    # Determine how many replacements to make
    total_replaceable = len(token_positions)
    num_replacements = min(num_replacements or total_replaceable, total_replaceable)
    
    # Randomly select positions to replace and assign new symbols
    selected_positions = random.sample(token_positions, k=num_replacements)
    replacement_map = {
        (block_idx, token_idx): random.choice(peptide_symbols)
        for block_idx, token_idx, _ in selected_positions
    }

    # Reconstruct the HELM string with replacements
    result_parts = []
    last_position = 0
    
    for block_idx, block in enumerate(blocks):
        # Add text between blocks
        result_parts.append(reference[last_position:block['start']])
        
        # Reconstruct block with potential replacements
        new_tokens = []
        for token_idx, token in enumerate(block['tokens']):
            has_brackets = token.startswith("[") and token.endswith("]")
            original_symbol = token[1:-1] if has_brackets else token
            
            # Use replacement if available, otherwise keep original
            new_symbol = replacement_map.get((block_idx, token_idx), original_symbol)
            new_tokens.append(f"[{new_symbol}]" if has_brackets else new_symbol)
        
        # Add reconstructed block
        result_parts.append(f"{block['tag']}{{{'.'.join(new_tokens)}}}")
        last_position = block['end']
    
    # Add remaining text after last block
    result_parts.append(reference[last_position:])
    return ''.join(result_parts)


def process_chunk(
    helm_sequences: List[str],
    monomer_db: List[Dict[str, str]],
    min_replacements: int,
    max_replacements: int,
    variants_per_sequence: int
) -> List[str]:
    """
    Process a chunk of HELM sequences in parallel, generating multiple variants.
    
    Uses normal distribution sampling for the number of replacements, with bounds
    defined as mean ± 3 standard deviations.
    
    Args:
        helm_sequences: List of HELM notation strings to process
        monomer_db: Database of monomer information
        min_replacements: Minimum number of replacements (mean - 3σ)
        max_replacements: Maximum number of replacements (mean + 3σ)  
        variants_per_sequence: Number of variants to generate per input sequence
        
    Returns:
        List of randomized HELM strings
    """
    output_sequences = []

    # Calculate normal distribution parameters
    # Using the constraint that min = mean - 3σ and max = mean + 3σ
    mean = (min_replacements + max_replacements) / 2
    std_dev = (max_replacements - min_replacements) / 6

    for helm_sequence in helm_sequences:
        # Generate multiple variants for each input sequence
        for _ in range(variants_per_sequence):
            # Sample number of replacements from normal distribution
            raw_sample = random.gauss(mu=mean, sigma=std_dev)
            num_replacements = max(
                min_replacements, 
                min(max_replacements, int(round(raw_sample)))
            )
            
            # Generate randomized variant
            randomized = randomize_peptide_symbols(
                helm_sequence, monomer_db, num_replacements
            )
            output_sequences.append(randomized)
    
    return output_sequences


def split_into_chunks(items: List, num_chunks: int) -> List[List]:
    """
    Split a list into approximately equal chunks for parallel processing.
    
    Args:
        items: List to split
        num_chunks: Number of chunks to create
        
    Returns:
        List of sublists (chunks)
    """
    chunk_size, remainder = divmod(len(items), num_chunks)
    return [
        items[i * chunk_size + min(i, remainder):(i + 1) * chunk_size + min(i + 1, remainder)]
        for i in range(num_chunks)
    ]


def load_and_filter_data(csv_path: str) -> List[str]:
    """
    Load HELM data from CSV and apply filtering rules.
    
    Filters out sequences containing:
    - R1-R5 patterns (reactive groups)
    - PEPTIDE2, CHEM, PEG polymer types
    
    Args:
        csv_path: Path to input CSV file
        
    Returns:
        List of filtered HELM strings
    """
    df = pd.read_csv(csv_path)
    
    # Define patterns to exclude
    exclusion_patterns = [
        r"\[R1\]", r"\[R2\]", r"\[R3\]", r"\[R4\]", r"\[R5\]",  # Reactive groups
        "PEPTIDE2",  # Secondary peptide chains
        "CHEM",      # Chemical modifications
        "PEG",       # Polyethylene glycol
        "Peg"        # Alternative PEG notation
    ]
    
    # Apply filters sequentially
    for pattern in exclusion_patterns:
        df = df[~df['HELM'].str.contains(pattern, na=False)]
    
    return df['HELM'].tolist()


def main():
    """Main execution function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Randomize peptide symbols in HELM notation strings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    default_db = os.environ.get('PEPEVOLVE_MONOMER_DB')
    default_input_csv = os.environ.get('PEPEVOLVE_PEPTIDE_CSV')
    
    # File paths
    parser.add_argument(
        '--db',
        default=default_db,
        help='Path to monomer database JSON file'
    )
    parser.add_argument(
        '--input_csv',
        default=default_input_csv,
        help='Path to input CSV file containing HELM sequences'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Path to output file for randomized sequences'
    )
    
    # Processing parameters
    parser.add_argument(
        '--workers',
        type=int,
        default=os.cpu_count(),
        help='Number of parallel worker processes'
    )
    parser.add_argument(
        '--min',
        type=int,
        default=1,
        help='Minimum number of symbol replacements per sequence'
    )
    parser.add_argument(
        '--max',
        type=int,
        default=5,
        help='Maximum number of symbol replacements per sequence'
    )
    parser.add_argument(
        '--var',
        type=int,
        default=10,
        help='Number of variants to generate per input sequence'
    )
    
    args = parser.parse_args()

    # Load monomer database
    print("Loading monomer database...")
    with open(args.db, 'r') as file:
        monomer_database = json.load(file)
    
    # Load and filter HELM sequences
    print("Loading and filtering HELM sequences...")
    helm_sequences = load_and_filter_data(args.input_csv)
    print(f"Processing {len(helm_sequences)} HELM sequences...")

    # Split work into chunks for parallel processing
    sequence_chunks = split_into_chunks(helm_sequences, args.workers)
    processing_tasks = [
        (chunk, monomer_database, args.min, args.max, args.var)
        for chunk in sequence_chunks
    ]

    # Process chunks in parallel and write results
    print("Processing sequences in parallel...")
    with ProcessPoolExecutor(max_workers=args.workers) as executor, \
         open(args.output, 'w') as output_file:
        
        # Submit all tasks
        futures = [
            executor.submit(process_chunk, *task_args) 
            for task_args in processing_tasks
        ]
        
        # Collect results as they complete
        for future in tqdm(
            as_completed(futures), 
            total=len(futures), 
            desc='Processing chunks'
        ):
            for randomized_sequence in future.result():
                output_file.write(randomized_sequence + '\n')

    print(f"Randomization complete. Results written to: {args.output}")


if __name__ == '__main__':
    main()