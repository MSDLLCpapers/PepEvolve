"""
HELM to CHUCKLES Converter

This module converts HELM (Hierarchical Editing Language for Macromolecules) notation
to CHUCKLES (Chemical Unicode Representation) SMILES format for peptide molecules.

HELM is a notation system for describing complex biological molecules like peptides,
while CHUCKLES is a standardized SMILES representation that can be processed by
chemical informatics tools.
"""

import re
import copy
import json
import time
import sys
import os
import argparse
import multiprocessing
from multiprocessing import Pool, cpu_count
from pathlib import Path
from functools import partial
from typing import Dict, List, Tuple, Union
from collections import Counter, defaultdict

import networkx as nx
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from openbabel import openbabel
from tqdm import tqdm


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def draw_smi(smi: str, size: Tuple[int, int] = (600, 600)):
    """
    Draw a molecule from SMILES string using RDKit.
    
    Args:
        smi: SMILES string
        size: Image size as (width, height) tuple
    
    Returns:
        PIL Image of the molecule
    """
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(Chem.MolFromSmiles(smi)))
    return Draw.MolToImage(mol, size=size)


def draw_graph(graphs: Union[nx.Graph, List[nx.Graph]], 
               layout: str = "circular", 
               node_size: int = 300) -> None:
    """
    Visualize NetworkX graphs using matplotlib.
    
    Args:
        graphs: Single graph or list of graphs to visualize
        layout: Graph layout algorithm ('spring', 'circular', 'kamada_kawai', etc.)
        node_size: Size of nodes in the visualization
    """
    layout_funcs = {
        "spring": nx.spring_layout,
        "circular": nx.circular_layout,
        "kamada_kawai": nx.kamada_kawai_layout,
        "random": nx.random_layout,
        "shell": nx.shell_layout
    }

    if not isinstance(graphs, list):
        graphs = [graphs]

    num_graphs = len(graphs)
    fig, axes = plt.subplots(1, num_graphs, figsize=(6 * num_graphs, 4))

    # Ensure axes is always iterable
    if num_graphs == 1:
        axes = [axes]

    for i, G in enumerate(graphs):
        pos = layout_funcs.get(layout, nx.spring_layout)(G)
        ax = axes[i]
        nx.draw(
            G, pos,
            ax=ax,
            with_labels=True,
            node_color='skyblue',
            node_size=node_size,
            font_size=14,
            edge_color='gray'
        )
        ax.set_title(f"Graph {i + 1}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def rm_num(x: str) -> str:
    """Remove all digits from a string."""
    return ''.join(char for char in x if not char.isdigit())


def lookup(records: List[Dict], keys: Union[str, List[str]], 
           values: Union[str, List[str]]) -> List[Dict]:
    """
    Find records in a list of dictionaries that match given key-value pairs.
    
    Args:
        records: List of dictionaries to search
        keys: Single key or list of keys to match
        values: Single value or list of values to match
    
    Returns:
        List of matching records
    """
    if not isinstance(keys, list):
        keys = [keys]
    if not isinstance(values, list):
        values = [values]

    if len(keys) != len(values):
        raise ValueError("Length of keys and values must be the same")

    return [
        record for record in records
        if all(record.get(k) == v for k, v in zip(keys, values))
    ]


# ============================================================================
# SMILES MANIPULATION FUNCTIONS
# ============================================================================

def wild2r(s: str, i: int = None) -> str:
    """
    Convert wildcard atoms (*:) to R groups in SMILES.
    
    Args:
        s: SMILES string with wildcards
        i: Optional index for specific wildcard replacement
    
    Returns:
        SMILES string with R groups
    """
    if i:
        return s.replace(f'*:{i}', 'R')
    return s.replace('*:', 'R')


def r2wild(smiles: str, idx: int = None) -> str:
    """
    Convert R groups back to wildcard atoms in SMILES.
    
    Args:
        smiles: SMILES string with R groups
        idx: Optional index for wildcard numbering
    
    Returns:
        SMILES string with wildcards
    """
    replacement = f'*:{idx}' if idx is not None else '*:'
    return re.sub(r'R(?=\d)', replacement, smiles)


def get_largest_ring_closure(G: nx.Graph) -> int:
    """
    Find the largest ring closure number used in any SMILES in the graph.
    
    This is important for avoiding conflicts when adding new ring closures.
    
    Args:
        G: NetworkX graph containing SMILES data
    
    Returns:
        Largest ring closure number found
    """
    output = 0
    
    def max_ring_closure(smiles: str) -> int:
        # Match single-digit closures (1–9) not preceded by '%'
        single_digits = re.findall(r'(?<!%)\d', smiles)
        # Match two-digit closures with % prefix (e.g., %10, %23)
        percent_closures = re.findall(r'%\d{2}', smiles)
        # Combine and convert to integers
        numbers = [int(n) for n in single_digits] + [int(n[1:]) for n in percent_closures]
        return max(numbers) if numbers else 0
    
    for n in G.nodes:
        smiles = G.nodes[n]['Complete_SMILES']
        output = max(output, max_ring_closure(smiles))

    return output


def get_chuckles_with_r(smiles_with_r: str) -> str:
    """
    Convert SMILES with R groups to CHUCKLES format by rearranging atom order.
    
    This function handles the complex process of:
    1. Identifying terminal attachment points
    2. Temporarily replacing R groups with dummy atoms
    3. Rearranging the molecule using OpenBabel
    4. Restoring the original R groups
    5. Handling unclosed ring digits
    
    Args:
        smiles_with_r: SMILES string containing R groups
    
    Returns:
        Rearranged SMILES in CHUCKLES format
    """
    # Mapping for temporary dummy atom replacements
    DUMMY_MAP = {'R1': '1000H', 'R2': '2000H', 'R3': '3000H', 'R4': '4000H', 'R5': '5000H'}
    UNDUMMY_MAP = {v: k for k, v in DUMMY_MAP.items()}

    def replace_all(s: str, replacements: dict) -> str:
        """Apply multiple string replacements."""
        for old, new in replacements.items():
            s = s.replace(old, new)
        return s
    
    def replace_ring_digit_outside_brackets(smiles: str, ring_digit: str, placeholder: str) -> str:
        """Replace ring digits only when they appear outside of atom brackets."""
        result = []
        i = 0
        inside_brackets = False
        ring_len = len(ring_digit)

        while i < len(smiles):
            char = smiles[i]

            if char == '[':
                inside_brackets = True
            elif char == ']':
                inside_brackets = False

            # Match ring digit exactly (e.g. "1" or "%10"), only outside brackets
            if not inside_brackets and smiles[i:i+ring_len] == ring_digit:
                result.append(placeholder)
                i += ring_len
                continue

            result.append(char)
            i += 1

        return ''.join(result)

    def count_hydrogens_before(substring: str, full_smiles: str) -> int:
        """Count [H] atoms before a given substring in SMILES."""
        return full_smiles.split(substring)[0].count('[H]')

    def get_terminal_indices(smiles: str) -> Union[Tuple[int], Tuple[int, int]]:
        """
        Find the indices of atoms connected to terminal wildcards.
        
        Returns:
            Tuple of (first_idx, last_idx) or (idx, None) for terminal atoms
        """
        smiles_with_wildcards = r2wild(smiles)
        mol = Chem.MolFromSmiles(smiles_with_wildcards)
        first_idx, last_idx = None, None

        for atom in mol.GetAtoms():
            atom_map = atom.GetAtomMapNum()
            if atom_map == 1:
                neighbor = atom.GetNeighbors()[0]
                first_idx = neighbor.GetIdx()
            elif atom_map == 2:
                neighbor = atom.GetNeighbors()[0]
                last_idx = neighbor.GetIdx()

        # Account for hydrogen atoms in indexing
        n_offset = count_hydrogens_before('[*:1]', smiles_with_wildcards) + 1
        c_offset = count_hydrogens_before('[*:2]', smiles_with_wildcards) + 1

        if first_idx is not None and last_idx is not None:
            return first_idx + n_offset, last_idx + c_offset
        elif first_idx is not None:
            return (first_idx + n_offset, None)
        elif last_idx is not None:
            return (None, last_idx + c_offset)
        else:
            return None, None

    def rearrange_smiles(smiles: str, first_idx: int, last_idx: int) -> str:
        """Use OpenBabel to rearrange SMILES starting from specified indices."""
        obconv = openbabel.OBConversion()
        obconv.SetInAndOutFormats('smi', 'smi')
        if first_idx:
            obconv.AddOption('f', openbabel.OBConversion.OUTOPTIONS, str(first_idx))
        if last_idx:
            obconv.AddOption('l', openbabel.OBConversion.OUTOPTIONS, str(last_idx))

        mol = openbabel.OBMol()
        obconv.ReadString(mol, smiles)
        return obconv.WriteString(mol).strip()

    def unclosed_ring_digits(smiles: str) -> List[int]:
        """Find ring digits that appear an odd number of times (unclosed rings)."""
        smiles_cleaned = re.sub(r'\[.*?\]', '', smiles)
        ring_tokens = re.findall(r'%\d{2}|\d', smiles_cleaned)
        ring_ids = [token[1:] if token.startswith('%') else token for token in ring_tokens]
        counts = Counter(ring_ids)
        unpaired = [int(ring) for ring, count in counts.items() if count % 2 != 0]

        if not unpaired:
            return []
        elif len(unpaired) == 1:
            return [unpaired[0]]
        else:
            return unpaired

    # === MAIN PROCESSING LOGIC ===
    
    # Handle unclosed ring digits by temporarily replacing them
    unpaired = unclosed_ring_digits(smiles_with_r)
    unclosed_map = {}
    
    for i, unclosed_num in enumerate(unpaired):
        ring_digit = f'%{unclosed_num}' if unclosed_num >= 10 else str(unclosed_num)
        placeholder = f'[{unclosed_num}H]'
        smiles_with_r = replace_ring_digit_outside_brackets(smiles_with_r, ring_digit, placeholder)
        # Save mappings for restoration
        unclosed_map[f'({placeholder})'] = ring_digit
        unclosed_map[placeholder] = ring_digit

    # Get terminal indices and rearrange molecule
    first_idx, last_idx = get_terminal_indices(smiles_with_r)
    smiles_with_dummy = replace_all(smiles_with_r, DUMMY_MAP)
    rearranged = rearrange_smiles(smiles_with_dummy, first_idx, last_idx)
    reverted = replace_all(rearranged, UNDUMMY_MAP)

    # Restore unclosed ring digits
    final_smiles = replace_all(reverted, unclosed_map)
    return final_smiles


# ============================================================================
# DATABASE LOADING AND PREPROCESSING
# ============================================================================

DEFAULT_MONOMER_DB = Path(__file__).resolve().parents[1] / "data" / "prep" / "filtered_monomers.json"


def load_monomer_database(monomer_db_path: str = None) -> List[Dict]:
    """Load monomer database from CLI path, env var, or default repo-relative location."""
    db_path = Path(monomer_db_path) if monomer_db_path else Path(os.environ.get("PEPEVOLVE_MONOMER_DB", DEFAULT_MONOMER_DB))
    if not db_path.is_file():
        raise FileNotFoundError(
            f"Monomer database not found at '{db_path}'. Provide --monomer-db or set PEPEVOLVE_MONOMER_DB."
        )

    with open(db_path, "r", encoding="utf-8") as f:
        monomers_database = json.load(f)

    for m in monomers_database:
        m['smiles'] = get_chuckles_with_r(m['smiles'])
    return monomers_database


# ============================================================================
# GRAPH CONSTRUCTION AND PROCESSING
# ============================================================================

def get_graph(helm: str, database: Dict = monomers_database) -> nx.Graph:
    """
    Convert HELM notation to a NetworkX graph representation.
    
    HELM format: POLYMER{monomer1.monomer2.monomer3}$connection1|connection2
    
    Args:
        helm: HELM notation string
        database: Monomer database for lookup
    
    Returns:
        NetworkX graph representing the molecule structure
    """
    def sort_key(key: str) -> Tuple[int, int]:
        """Sort polymer keys by type (PEPTIDE first, then CHEM) and number."""
        if key.startswith("PEPTIDE"):
            prefix_order = 0
            num = int(key[len("PEPTIDE"):])
        elif key.startswith("CHEM"):
            prefix_order = 1
            num = int(key[len("CHEM"):])
        else:
            prefix_order = 2
            num = 0
        return (prefix_order, num)

    def extract_connection(connection: str) -> Tuple[Tuple[str, int, str], Tuple[str, int, str]]:
        """Parse connection string to extract polymer, index, and attachment point info."""
        start_poly, end_poly, detail = connection.split(',')
        start_att, end_att = detail.split('-')
        (start_idx, start_r), (end_idx, end_r) = start_att.split(':'), end_att.split(':')
        return (start_poly, int(start_idx)-1, start_r), (end_poly, int(end_idx)-1, end_r)
    
    # Extract polymer definitions from HELM
    matches = re.findall(r'(\w+)\{(.*?)\}', helm)
    G = nx.Graph()
    global_index = 1  # Global counter for unique node names
    
    # Process each polymer chain
    for polymer, monomers_str in sorted(matches, key=lambda x: sort_key(x[0])):
        monomers_raw = [m.strip('[]') for m in monomers_str.split('.')]
        monomers = []
        
        for i, mon in enumerate(monomers_raw):
            unique_name = f"{mon}_{global_index}"
            # Look up monomer data in database
            data = lookup(database, ['polymertype', 'symbol'], [rm_num(polymer), mon])[0]
            
            # Mark the last monomer in each chain
            if i == len(monomers_raw) - 1:
                G.add_node(unique_name, **data, polymer=polymer, global_idx=global_index, is_last=True)
            else:
                G.add_node(unique_name, **data, polymer=polymer, global_idx=global_index)
            
            monomers.append(unique_name)
            global_index += 1

        # Add edges between consecutive monomers in the same polymer
        edges = [(monomers[i], monomers[i + 1]) for i in range(len(monomers) - 1)]
        G.add_edges_from(edges)

    # Process inter-polymer connections
    if '$' in helm:
        connections = helm.split('$')[1].split('|')
        
        for connection in connections:
            (start_poly, start_idx, start_r), (end_poly, end_idx, end_r) = extract_connection(connection)
            
            # Find nodes by polymer and index
            u_global_idx, u = [(i, n) for i, (n, attr) in enumerate(G.nodes(data=True), 1) 
                              if attr.get("polymer") == start_poly][start_idx]
            v_global_idx, v = [(i, n) for i, (n, attr) in enumerate(G.nodes(data=True), 1) 
                              if attr.get("polymer") == end_poly][end_idx]
            
            # Add edge with connection information
            G.add_edge(u, v, type=f'{u_global_idx}{start_r}-{v_global_idx}{end_r}')

    return G


def extract_special_subgraphs(G: nx.Graph) -> List[nx.Graph]:
    """
    Extract subgraphs containing special inter-polymer connections.
    
    Args:
        G: Full molecular graph
    
    Returns:
        List of subgraphs containing special connections
    """
    # Find edges with connection type information (inter-polymer bonds)
    special_edges = [(u, v) for u, v, d in G.edges(data=True) if d]
    H = G.edge_subgraph(special_edges).copy()
    # Split into connected components
    subgraphs = [H.subgraph(c).copy() for c in nx.connected_components(H)]
    return subgraphs


def process_subgraph(G: nx.Graph, ring_closure: int) -> Tuple[nx.Graph, int]:
    """
    Process a subgraph by converting inter-polymer connections to ring closures.
    
    This is the core of converting HELM connections to SMILES ring notation.
    
    Args:
        G: Subgraph to process
        ring_closure: Starting ring closure number
    
    Returns:
        Tuple of (processed_graph, next_ring_closure_number)
    """
    def replace_wildcards(smiles1: str, smiles2: str, labels: Tuple[int, int], number: int) -> Tuple[str, str]:
        """Replace wildcard atoms with ring closure numbers."""
        def replace_labels(smiles: str, labels: Tuple[int, int]) -> str:
            replacement = f"%{number}" if number >= 10 else str(number)
            for label in labels:
                # Handle different wildcard formats
                smiles = re.sub(rf'\(\[\*\:{label}\]\)', replacement, smiles)  # ([*:1]) → %10
                smiles = re.sub(rf'\[\*\:{label}\](\w)', lambda m: f"{m.group(1)}{replacement}", smiles)  # [*:1]C → C%10
                smiles = re.sub(rf'(\w)\[\*\:{label}\]', lambda m: f"{m.group(1)}{replacement}", smiles)  # C[*:1] → C%10
                smiles = re.sub(rf'\[\*\:{label}\]', replacement, smiles)  # [*:1] → %10
            return smiles
        
        return replace_labels(smiles1, labels), replace_labels(smiles2, labels)
    
    def modify_graph(graph: nx.Graph, u: str, v: str, labels: Tuple[int, int], ring_closure: int) -> Tuple[nx.Graph, int]:
        """Modify two connected nodes by adding ring closure and removing the edge."""
        u_data, v_data = graph.nodes[u], graph.nodes[v]
        
        # Convert to wildcard format temporarily
        u_wild = r2wild(u_data['smiles'], u_data['global_idx'])
        v_wild = r2wild(v_data['smiles'], v_data['global_idx'])
        
        # Replace wildcards with ring closures
        u_modified, v_modified = replace_wildcards(u_wild, v_wild, labels, ring_closure)
        
        # Update node data
        graph.nodes[u]['smiles'] = u_modified
        graph.nodes[v]['smiles'] = v_modified
        graph.remove_edge(u, v)
        
        return graph, ring_closure + 1
    
    def extract_labels(edge_attr: dict) -> Tuple[int, int]:
        """Extract attachment point labels from edge attributes."""
        start_attr, end_attr = edge_attr['type'].split('-')
        start_idx, start_ring = start_attr.split('R')
        end_idx, end_ring = end_attr.split('R')
        return int(f"{start_idx}{start_ring}"), int(f"{end_idx}{end_ring}")

    def has_special_edges(graph: nx.Graph) -> bool:
        """Check if graph has any special connection edges."""
        return any('type' in attr for _, _, attr in graph.edges(data=True))

    # === MAIN PROCESSING LOOP ===
    # Process one edge at a time until all special edges are handled
    while has_special_edges(G):
        for u, v, edge_attr in copy.deepcopy(G).edges(data=True):
            if 'type' in edge_attr:
                labels = extract_labels(edge_attr)
                G, ring_closure = modify_graph(G, u, v, labels, ring_closure)
                break  # Only process one edge per iteration

    return G, ring_closure


def process_special_subgraphs(G: nx.Graph, subgraphs: List[nx.Graph]) -> nx.Graph:
    """
    Process all special subgraphs and update the main graph.
    
    Args:
        G: Main molecular graph
        subgraphs: List of special subgraphs to process
    
    Returns:
        Updated main graph with processed connections
    """
    ring_closure = get_largest_ring_closure(G) + 1
    
    for sg in subgraphs:
        processed_sg, ring_closure = process_subgraph(sg, ring_closure)
        # Update main graph with processed data
        for node, attr in processed_sg.nodes(data=True):
            if node in G:
                G.nodes[node].update(attr)
    
    return G


def postprocess(G: nx.Graph) -> nx.Graph:
    """
    Post-process the graph to finalize SMILES strings.
    
    This includes:
    - Replacing R3, R4, R5 groups with actual chemical groups
    - Cleaning up remaining wildcards
    - Adding terminal groups (like -OH for peptides)
    
    Args:
        G: Graph to post-process
    
    Returns:
        Post-processed graph
    """
    # Common R-group replacements
    R_REPLACEMENTS = {
        'Vinyl': 'C=C',
        'Azide': 'N=[N+]=[N-]',
        'Alkyne': 'C#C'
    }

    # Process each node
    for n in G.nodes:
        node_data = G.nodes[n]
        smiles = wild2r(node_data['smiles'], node_data['global_idx'])
        
        # Replace R3, R4, R5 groups
        for r in [3, 4, 5]:
            r_val = node_data.get(f"r{r}")
            if r_val:
                if r_val in R_REPLACEMENTS:
                    smiles = smiles.replace(f'[R{r}]', R_REPLACEMENTS[r_val])
                else:
                    smiles = smiles.replace(f'R{r}', r_val)

        # Try to rearrange with CHUCKLES format
        try:
            smiles = get_chuckles_with_r(smiles)
        except Exception:
            pass  # Keep original if rearrangement fails

        # Clean up remaining R1 and R2 groups
        smiles = smiles.replace('([R1])', '').replace('([R2])', '').replace('[R1]', '').replace('[R2]', '')
        G.nodes[n]['smiles'] = smiles

    # Add terminal -OH groups to peptide chains
    for n, attr in G.nodes(data=True):
        if (attr.get('is_last') and 
            attr.get('r2') == 'OH' and 
            attr['smiles'].endswith('C(=O)')):
            G.nodes[n]['smiles'] = re.sub(r'C\(=O\)$', 'C(=O)O', attr['smiles'])

    return G


def join_peptide_smiles(graph: nx.Graph) -> str:
    """
    Join SMILES from all polymers into a single CHUCKLES string.
    
    Args:
        graph: Processed molecular graph
    
    Returns:
        CHUCKLES SMILES string with polymers separated by '|' and molecules by '.'
    """
    # Group SMILES by polymer
    polymer_smiles = defaultdict(list)

    for node, data in graph.nodes(data=True):
        polymer = data.get("polymer")
        smiles = data.get("smiles")
        if polymer and smiles:
            polymer_smiles[polymer].append(smiles)

    # Join: polymers with '|', molecules within polymers with '.'
    joined = ".".join("|".join(polymer_smiles[poly]) for poly in polymer_smiles)
    return joined


# ============================================================================
# MAIN CONVERSION FUNCTION
# ============================================================================

def helm2chuckles(helm: str, monomers_database: List[Dict] = None) -> str:
    """
    Convert HELM notation to CHUCKLES SMILES format.
    
    This is the main function that orchestrates the entire conversion process:
    1. Parse HELM and build molecular graph
    2. Process special inter-polymer connections
    3. Post-process and clean up SMILES
    4. Join into final CHUCKLES format
    5. Validate the result
    
    Args:
        helm: HELM notation string
        monomers_database: Database of monomer definitions
    
    Returns:
        CHUCKLES SMILES string, or empty string if conversion fails
    """
    try:
        if monomers_database is None:
            monomers_database = load_monomer_database()

        # Step 1: Build and process graph
        graph = get_graph(helm, monomers_database)
        special_subgraphs = extract_special_subgraphs(graph)
        graph = process_special_subgraphs(graph, special_subgraphs)
        graph = postprocess(graph)

        # Step 2: Join into CHUCKLES SMILES
        chuckles = join_peptide_smiles(graph)

        # Step 3: Validate result
        smiles = chuckles.replace('|', '')  # Remove polymer separators for validation
        if Chem.MolFromSmiles(smiles) is None:
            return ""  # Invalid SMILES
        
        return chuckles

    except Exception as e:
        # Return empty string for any conversion failure
        return ""


# ============================================================================
# PARALLEL PROCESSING UTILITIES
# ============================================================================

def read_in_chunks(file_path: str, chunk_size: int = 10000):
    """
    Generator to read large files in chunks for memory-efficient processing.
    
    Args:
        file_path: Path to input file
        chunk_size: Number of lines per chunk
    
    Yields:
        Lists of lines (chunks)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        chunk = []
        for line in f:
            chunk.append(line.strip())
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
        if chunk:  # Don't forget the last chunk
            yield chunk


def parallel_process_file(input_path: str, output_path: str,
                         workers: int = None, chunk_size: int = 10000,
                         monomer_db_path: str = None):
    """
    Process a large file of HELM strings in parallel.
    
    Args:
        input_path: Input file path (one HELM string per line)
        output_path: Output file path for CHUCKLES results
        workers: Number of worker processes (default: CPU count)
        chunk_size: Lines to process per chunk
    """
    if workers is None:
        workers = cpu_count()
        print(f"Using {workers} worker processes")
    
    monomer_database = load_monomer_database(monomer_db_path)
    convert_fn = partial(helm2chuckles, monomers_database=monomer_database)

    total_lines = 0
    start_time = time.time()

    with open(output_path, 'w', encoding='utf-8') as out_file:
        with Pool(processes=workers) as pool:
            for chunk in tqdm(read_in_chunks(input_path, chunk_size), desc="Processing"):
                # Process chunk in parallel
                results = pool.map(convert_fn, chunk)
                # Filter out empty results (failed conversions)
                valid = [r for r in results if r]
                # Write valid results
                if valid:
                    out_file.write('\n'.join(valid) + '\n')
                total_lines += len(results)

    # Print performance statistics
    end_time = time.time()
    duration = end_time - start_time
    rate = total_lines / duration if duration > 0 else 0

    print(f"\n✅ Done. Processed {total_lines:,} lines in {duration:.2f} seconds.")
    print(f"⚡ Speed: {rate:,.2f} lines/second")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    """
    Command line usage:
    python helm2chuckles.py input_file.txt
    
    Output will be saved as input_file_valid_CHUCKLES.txt
    """
    parser = argparse.ArgumentParser(description="Convert HELM sequences to CHUCKLES format.")
    parser.add_argument("input_file", help="Input text file containing one HELM sequence per line.")
    parser.add_argument(
        "--monomer-db",
        default=os.environ.get("PEPEVOLVE_MONOMER_DB", str(DEFAULT_MONOMER_DB)),
        help="Path to monomer database JSON file.",
    )
    args = parser.parse_args()

    input_file = args.input_file
    output_file = f"{input_file.split('.')[0]}_valid_CHUCKLES.txt"
    
    print(f"Converting HELM file: {input_file}")
    print(f"Output will be saved to: {output_file}")
    
    parallel_process_file(input_file, output_file, monomer_db_path=args.monomer_db)