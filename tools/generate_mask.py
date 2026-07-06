#!/usr/bin/env python3
"""
Molecular Sequence Masking Tool

This script generates masked variants of molecular sequences (CHUCKLES format) for 
pretraining. It randomly masks fragments of sequences and creates 
source-target pairs suitable for pretraining PepINVENT.

Usage:
    python mask_sequences.py --input_txt sequences.txt --output_csv masked_data.csv
"""

import re
import random
import csv
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Tuple


def mask_sequence(sequence: str, num_masks: int) -> Tuple[str, str]:
    """
    Mask exactly num_masks fragments in a molecular sequence.
    
    Fragments are defined as parts separated by '|' or '.' delimiters.
    Masked fragments are replaced with '?' in the source sequence.
    
    Args:
        sequence: Input molecular sequence string (CHUCKLES format)
        num_masks: Number of fragments to mask
        
    Returns:
        Tuple of (masked_sequence, masked_fragments_joined)
        - masked_sequence: Original sequence with selected fragments replaced by '?'
        - masked_fragments_joined: The masked fragments joined with '|'
    """
    # Split sequence on delimiters while preserving them
    # Results in alternating pattern: [fragment, delimiter, fragment, delimiter, ...]
    parts = re.split(r'(\||\.)', sequence)

    # Fragment indices are at even positions (0, 2, 4, ...)
    fragment_indices = [i for i in range(0, len(parts), 2)]
    total_fragments = len(fragment_indices)

    # Ensure we don't try to mask more fragments than exist
    num_masks = min(max(1, num_masks), total_fragments)
    
    # Randomly select fragments to mask
    indices_to_mask = set(random.sample(fragment_indices, num_masks))

    # Build masked sequence and collect masked fragments
    masked_parts = []
    masked_fragments = []
    
    for idx, part in enumerate(parts):
        if idx in indices_to_mask:
            masked_parts.append('?')
            masked_fragments.append(part)
        else:
            masked_parts.append(part)

    # Reconstruct the sequences
    source_masked = ''.join(masked_parts)
    target_joined = '|'.join(masked_fragments)
    
    return source_masked, target_joined


def process_chunk(
    sequences: List[str],
    min_masks: int,
    max_masks: int,
    variants_per: int
) -> List[Tuple[str, str]]:
    """
    Process a chunk of sequences and generate masked variants.
    
    For each sequence, creates multiple masked variants with the number of masks
    sampled from a triangular distribution (skewed toward the minimum).
    
    Args:
        sequences: List of molecular sequences to process
        min_masks: Minimum number of fragments to mask
        max_masks: Maximum number of fragments to mask
        variants_per: Number of masked variants to generate per sequence
        
    Returns:
        List of (source, target) tuples for training data
    """
    results = []
    
    for sequence in sequences:
        # Generate multiple variants for each sequence
        for _ in range(variants_per):
            # Count available fragments
            parts = re.split(r'(\||\.)', sequence)
            fragment_indices = [i for i in range(0, len(parts), 2)]
            num_fragments = len(fragment_indices)
            
            # Sample number of masks using triangular distribution (skewed toward min)
            raw_masks = random.triangular(min_masks, max_masks, min_masks)
            num_masks = int(round(raw_masks))
            num_masks = max(min_masks, min(max_masks, num_masks))
            
            # Apply special rules based on sequence length
            if num_fragments <= 8:
                # Short sequences: only mask 1 fragment
                num_masks = 1
            elif num_fragments >= 15:
                # Long sequences: 5% chance to mask 5 fragments
                if random.random() >= 0.95:
                    num_masks = 5

            # Generate masked variant
            source, target = mask_sequence(sequence, num_masks)
            results.append((source, target))
    
    return results


def chunkify(lst: List, n_chunks: int) -> List[List]:
    """
    Split a list into approximately equal chunks for parallel processing.
    
    Args:
        lst: List to split
        n_chunks: Number of chunks to create
        
    Returns:
        List of chunks (sublists)
    """
    chunk_size, remainder = divmod(len(lst), n_chunks)
    return [
        lst[i * chunk_size + min(i, remainder):(i + 1) * chunk_size + min(i + 1, remainder)] 
        for i in range(n_chunks)
    ]


def validate_inputs(args) -> None:
    """
    Validate command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Raises:
        SystemExit: If validation fails
    """
    if not os.path.exists(args.input_txt):
        print(f"Error: Input file '{args.input_txt}' does not exist.")
        raise SystemExit(1)
        
    if args.min_masks < 1:
        print("Error: min_masks must be at least 1.")
        raise SystemExit(1)
        
    if args.max_masks < args.min_masks:
        print("Error: max_masks must be greater than or equal to min_masks.")
        raise SystemExit(1)
        
    if args.variants_per < 1:
        print("Error: variants_per must be at least 1.")
        raise SystemExit(1)


def main():
    """Main execution function."""
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Generate masked variants of molecular sequences for ML training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mask_sequences.py --input_txt molecules.txt --output_csv training_data.csv
  python mask_sequences.py --input_txt molecules.txt --output_csv data.csv --workers 8 --variants_per 3
        """
    )
    
    parser.add_argument(
        '--input_txt', 
        required=True,
        help='Path to input TXT file with one molecular sequence per line'
    )
    parser.add_argument(
        '--output_csv', 
        required=True,
        help='Path to output CSV file for masked sequence pairs'
    )
    parser.add_argument(
        '--workers', 
        type=int, 
        default=os.cpu_count(),
        help=f'Number of parallel workers (default: {os.cpu_count()})'
    )
    parser.add_argument(
        '--min_masks', 
        type=float, 
        default=1,
        help='Minimum number of fragments to mask (default: 1)'
    )
    parser.add_argument(
        '--max_masks', 
        type=float, 
        default=4.5,
        help='Maximum number of fragments to mask (default: 4.5)'
    )
    parser.add_argument(
        '--variants_per', 
        type=int, 
        default=1,
        help='Number of masked variants per input sequence (default: 1)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    validate_inputs(args)
    
    # Load input sequences
    print(f"Loading sequences from {args.input_txt}...")
    try:
        with open(args.input_txt, 'r', encoding='utf-8') as f:
            sequences = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading input file: {e}")
        raise SystemExit(1)
    
    if not sequences:
        print("Error: No sequences found in input file.")
        raise SystemExit(1)
        
    print(f"Loaded {len(sequences)} sequences")
    print(f"Generating {args.variants_per} variant(s) per sequence...")
    
    # Prepare data for parallel processing
    chunks = chunkify(sequences, args.workers)
    tasks = [
        (chunk, args.min_masks, args.max_masks, args.variants_per) 
        for chunk in chunks if chunk  # Skip empty chunks
    ]
    
    # Process sequences and write results
    total_written = 0
    try:
        with open(args.output_csv, 'w', newline='', encoding='utf-8') as fout:
            writer = csv.writer(fout)
            writer.writerow(['Source_Mol', 'Target_Mol'])

            # Process chunks in parallel with progress bar
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(process_chunk, *task) for task in tasks]
                
                for future in tqdm(
                    as_completed(futures), 
                    total=len(futures), 
                    desc='Processing chunks'
                ):
                    try:
                        results = future.result()
                        for source, target in results:
                            writer.writerow([source, target])
                            total_written += 1
                    except Exception as e:
                        print(f"Error processing chunk: {e}")
                        continue
                        
    except Exception as e:
        print(f"Error writing output file: {e}")
        raise SystemExit(1)
    
    print(f"Successfully generated {total_written} masked sequence pairs")
    print(f"Output written to {args.output_csv}")


if __name__ == '__main__':
    main()