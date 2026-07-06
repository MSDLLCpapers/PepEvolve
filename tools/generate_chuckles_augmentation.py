"""
Peptide Sequence Processing and Augmentation Tool

This module processes peptide sequences in SMILES format, performing cyclic shifts
to generate augmented datasets for machine learning applications. It uses parallel
processing to handle large datasets efficiently.

Author: Your Name
Date: 2025
"""

import re
import os
import argparse
import tempfile
import shutil
import multiprocessing as mp
from collections import Counter
from functools import partial
from typing import List, Optional

from rdkit import Chem
from rdkit import RDLogger

# Disable RDKit warnings to reduce console clutter
RDLogger.DisableLog('rdApp.*')


def shift_chuckles(chuckles: str, shift: int) -> Optional[str]:
    """
    Shift the peptide sequence by rotating the order of amino acid units.
    
    This function takes a peptide sequence in a pipe-separated format and rotates
    the order of the units while maintaining chemical validity through proper
    ring number handling.
    
    Args:
        chuckles (str): Pipe-separated peptide sequence (e.g., "unit1|unit2|unit3")
        shift (int): Number of positions to shift (wraps around sequence length)
    
    Returns:
        Optional[str]: Shifted sequence if chemically valid, None otherwise
    
    Example:
        >>> shift_chuckles("A|B|C", 1)
        "C|A|B"
    """
    def _extract_unpaired_num(sequence_part: str) -> str:
        """
        Extract ring numbers that appear an odd number of times (unpaired).
        
        In SMILES notation, ring numbers appear in pairs to indicate ring closures.
        An unpaired ring number indicates the start or end of a ring system.
        
        Args:
            sequence_part (str): Part of the SMILES sequence to analyze
            
        Returns:
            str: The unpaired ring number, or empty string if none found
        """
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
        return None


def process_chunk(peptides_chunk: List[str], chunk_id: int, output_dir: str) -> str:
    """
    Process a chunk of peptides and generate all possible cyclic shifts.
    
    This function takes a subset of peptides, generates all possible cyclic
    shifts for each, validates them, and writes valid shifts to a temporary file.
    
    Args:
        peptides_chunk (List[str]): List of peptide sequences to process
        chunk_id (int): Unique identifier for this chunk (used in filename)
        output_dir (str): Directory to write temporary output files
    
    Returns:
        str: Path to the temporary file containing processed results
    """
    # Create unique temporary file for this chunk
    temp_filename = os.path.join(output_dir, f'temp_chunk_{chunk_id}.txt')
    
    # Process each peptide in the chunk
    with open(temp_filename, 'w') as temp_file:
        for peptide_sequence in peptides_chunk:
            # Calculate maximum possible shifts (equal to number of units)
            num_units = len(peptide_sequence.split('|'))
            
            # Generate all possible cyclic shifts
            for shift_amount in range(num_units):
                shifted_sequence = shift_chuckles(peptide_sequence, shift_amount)
                
                # Only write valid (non-None) shifted sequences
                if shifted_sequence:
                    temp_file.write(shifted_sequence + '\n')
    
    return temp_filename


def merge_temp_files(temp_files: List[str], output_file: str) -> None:
    """
    Merge all temporary files into a single output file and clean up.
    
    Args:
        temp_files (List[str]): List of temporary file paths to merge
        output_file (str): Path to final output file
    """
    with open(output_file, 'a') as outfile:
        for temp_file_path in temp_files:
            # Copy contents of each temporary file
            with open(temp_file_path, 'r') as infile:
                outfile.write(infile.read())
            
            # Clean up temporary file immediately after processing
            os.remove(temp_file_path)


def parallel_process_peptides(input_file: str, 
                            output_file: str, 
                            num_processes: Optional[int] = None) -> None:
    """
    Main function to process peptides in parallel using multiprocessing.
    
    This function orchestrates the entire peptide processing pipeline:
    1. Load all peptides from input file
    2. Split into chunks for parallel processing
    3. Process chunks in parallel to generate augmented sequences
    4. Merge results into final output file
    
    Args:
        input_file (str): Path to input file containing peptide sequences
        output_file (str): Path to output file for augmented sequences
        num_processes (Optional[int]): Number of CPU cores to use (default: all available)
    """
    # Use all available CPU cores if not specified
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    print(f"Using {num_processes} CPU cores for processing...")
    
    # Load all peptide sequences from input file
    print("Loading peptides...")
    try:
        with open(input_file, 'r') as file:
            all_peptides = [line.strip() for line in file.readlines() if line.strip()]
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
    
    print(f"Loaded {len(all_peptides)} peptides")
    
    if not all_peptides:
        print("No peptides found in input file.")
        return
    
    # Calculate optimal chunk size for even distribution across processes
    chunk_size = len(all_peptides) // num_processes
    if len(all_peptides) % num_processes != 0:
        chunk_size += 1  # Add 1 to handle remainder
    
    # Split peptides into chunks for parallel processing
    peptide_chunks = []
    for i in range(0, len(all_peptides), chunk_size):
        chunk = all_peptides[i:i + chunk_size]
        peptide_chunks.append(chunk)
    
    print(f"Split into {len(peptide_chunks)} chunks of ~{chunk_size} peptides each")
    
    # Create temporary directory for intermediate files
    temp_dir = tempfile.mkdtemp(prefix='peptide_processing_')
    
    try:
        # Create partial function with pre-filled output directory argument
        process_func = partial(process_chunk, output_dir=temp_dir)
        
        # Process all chunks in parallel
        print("Processing chunks in parallel...")
        with mp.Pool(processes=num_processes) as pool:
            # Prepare arguments: (chunk, chunk_id) pairs
            chunk_args = [(chunk, i) for i, chunk in enumerate(peptide_chunks)]
            
            # Execute parallel processing
            temp_files = pool.starmap(process_func, chunk_args)
        
        # Combine all results into final output file
        print("Merging results...")
        merge_temp_files(temp_files, output_file)
        
        print(f"Processing complete! Output written to {output_file}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
    finally:
        # Clean up temporary directory and any remaining files
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def main():
    """
    Main execution function with configuration.
    """
    parser = argparse.ArgumentParser(description="Generate CHUCKLES augmentations in parallel.")
    parser.add_argument(
        '--input-file',
        help='Input file with one peptide sequence per line.',
    )
    parser.add_argument(
        '--output-file',
        help='Output file path for augmented sequences.',
    )
    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Process peptides using all available CPU cores
    parallel_process_peptides(input_file, output_file)
    
    # Alternative: specify number of processes manually
    # parallel_process_peptides(input_file, output_file, num_processes=8)


if __name__ == "__main__":
    main()