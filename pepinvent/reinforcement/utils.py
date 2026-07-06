import io
import torch
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image, ImageDraw, ImageFont
from itertools import product
from typing import List, Tuple, Dict
from pepinvent.scoring_function.score_summary import FinalSummary
from reinvent_models.model_factory.dto.sampled_sequence_dto import SampledSequencesDTO
from collections import defaultdict

def mean_score_per_input(scores, groups, num_input=4, num_per_input=32):
    """
    Compute mean score for each group.

    Parameters:
        scores (list or array): List of scores (length N)
        groups (list or array): List of group indices (length N)
        num_input (int): Total number of groups (default=16)

    Returns:
        list: Mean score for each group, index corresponds to group number
    """
    group_sums = defaultdict(float)
    group_counts = defaultdict(int)
    
    # Aggregate sums and counts
    for score, group in zip(scores, groups):
        group_sums[group] += score
        group_counts[group] += 1
    
    # Compute mean for each group (handle empty groups gracefully)
    means = []
    for g in range(num_input):
        if group_counts[g] > 0:
            means.append(group_sums[g] / num_per_input)
        else:
            means.append(0)  # or 0, depending on preference
    
    return torch.tensor(means)

def grpo_normalize(scores, group_ids):
    """
    Normalize scores within each group (Group Relative Policy Optimization).
    
    For each group:
    - normalized_score = (score - group_mean) / group_std

    Args:
        scores: array-like of scores
        group_ids: array-like of group identifiers (same length as scores)
    
    Returns:
        normalized_scores: numpy array of normalized scores
    """
    scores = np.array(scores)
    group_ids = np.array(group_ids)
    normalized_scores = np.zeros_like(scores, dtype=float)
    
    unique_groups = np.unique(group_ids)
    
    for group in unique_groups:
        # Get mask for current group
        mask = group_ids == group
        group_scores = scores[mask]
        
        # Compute group statistics
        group_mean = np.mean(group_scores)
        group_std = np.std(group_scores)
        
        # Avoid division by zero
        if group_std == 0:
            normalized_scores[mask] = 0
        else:
            normalized_scores[mask] = (group_scores - group_mean) / group_std + 1e-8
    
    return normalized_scores

def combos(list_of_lists: List[List[Tuple[float, str]]]) -> List[str]:
    results: List[str] = []

    def backtrack(depth: int, current: List[str]):
        # base case: we reached the end
        if depth == len(list_of_lists):
            results.append('|'.join(current))
            return

        # iterate over the current list at this depth
        for _, s in list_of_lists[depth]:
            backtrack(depth + 1, current + [s])

    backtrack(0, [])
    return results


def softmax(x):
    # Subtract max for numerical stability
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def mask_chuckles(chuckles, positions):
    monomers = chuckles.split('|')
    
    # Convert single integer to list
    if isinstance(positions, int):
        positions = [positions]
    
    # Create mask array with 1s at specified positions
    mask = [1 if i in positions else 0 for i in range(len(monomers))]
    
    masked_parts = [('?' if m == 1 else part) for part, m in zip(monomers, mask)]
    return '|'.join(masked_parts)

def fill_chuckles(source: str, target: str) -> str:
    source_parts = source.split('|')
    target_parts = target.split('|')
    target_index = 0

    for i in range(len(source_parts)):
        if source_parts[i] == '?' and target_index < len(target_parts):
            source_parts[i] = target_parts[target_index]
            target_index += 1

    return '|'.join(source_parts)

def mask_batch_chuckles(chuckles, positions_2d):
    """
    Create multiple masked variants of a single chuckles string, one per row of positions.

    Parameters
    ----------
    chuckles : str
        A '|'-separated string, e.g. "A|B|C|D".
    positions_2d : list[list[int]] or 2D tensor
        Each row is a list of indices to mask for one output. Supports:
        - list of lists of ints (indices)
        - list of bools of length == number of monomers (boolean mask)
        - PyTorch/NumPy 2D tensors (will .tolist())

    Returns
    -------
    list[str]
        Masked chuckles strings (one per row in positions_2d).
    """
    # Normalize positions_2d to a Python list of lists
    if hasattr(positions_2d, "tolist"):
        positions_2d = positions_2d.tolist()

    monomers = chuckles.split('|')
    n = len(monomers)
    out = []

    for row in positions_2d:
        # If a tensor row sneaks through
        if hasattr(row, "tolist"):
            row = row.tolist()

        # Support boolean mask rows (length == n)
        if isinstance(row, (list, tuple)) and len(row) == n and any(isinstance(x, bool) for x in row):
            idxs = {i for i, b in enumerate(row) if bool(b)}
        else:
            # Treat as list of indices; coerce to int, drop out-of-range (and -1 etc.)
            idxs = set()
            for x in (row if isinstance(row, (list, tuple)) else [row]):
                try:
                    i = int(x)
                except (TypeError, ValueError):
                    continue
                if 0 <= i < n:
                    idxs.add(i)

        masked = [("?" if i in idxs else part) for i, part in enumerate(monomers)]
        out.append("|".join(masked))

    return out



##################################
######### Visualization ##########
##################################

def print_(text, text_color="red", width=100):
    # ANSI color map
    COLORS = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m"
    }

    # Fixed color for '=' signs (cyan)
    EQUAL_COLOR = "\033[96m"
    RESET = "\033[0m"

    # Get the ANSI code for the given text_color string, default to white if not found
    color_code = COLORS.get(text_color.lower(), COLORS["white"])

    # Build top and bottom border
    border = f"{color_code}{'=' * width}{RESET}"

    # Center text with padding but without losing ANSI colors
    centered_text = text.center(width)
    colored_text = f"{color_code}{centered_text}{RESET}"

    # Print banner
    print(border)
    print(colored_text)
    print(border)

def plot_argmax_indices(argmax_indices: List[int], save_path: str):
    # Number of possible positions
    L = len(argmax_indices) and max(argmax_indices) + 1

    # Prepare data
    steps = list(range(len(argmax_indices)))
    values = argmax_indices

    max_idx = max(values) if values else 0
    L = max_idx + 1  # number of possible positions

    plt.figure(figsize=(10, 4), dpi=150)
    plt.plot(steps, values, linestyle='-', marker='o', markersize=3)
    plt.xlabel('Step')
    plt.ylabel('Index of Highest Mask‐Probability')
    plt.title('Argmax Mask Position at Each Step (PPO)')

    # Show every integer position on y
    plt.yticks(range(L))
    plt.ylim(-0.5, max_idx + 1.0 + 0.1)

    # Find and annotate runs ≥ 3
    run_start = 0
    for i in range(1, len(values) + 1):
        # end of run if value changes or end of list
        if i == len(values) or values[i] != values[run_start]:
            run_length = i - run_start
            if run_length >= 3:
                y = values[run_start]
                x0, x1 = run_start, i - 1
                # draw thicker horizontal segment
                plt.hlines(y, x0, x1, linewidth=3)
                # label it
                plt.text((x0 + x1) / 2, y + 0.1, str(y),
                        ha='center', va='bottom', fontweight='bold')
            run_start = i

    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_categorical_evolution(distributions, save_path):
    """
    Plot and save a heatmap showing the evolution of categorical distributions over time.

    Parameters
    ----------
    distributions : list of np.ndarray
        Each element is a 1D array representing a categorical distribution at a timestep.
        All arrays must have the same length (number of categories).
    save_path : str
        File path to save the resulting image (e.g., 'output.png' or 'plots/evolution.pdf').
    """

    # Convert list of arrays into 2D NumPy array (T × C)
    data = np.vstack(distributions)
    timesteps, categories = data.shape

    plt.figure(figsize=(max(8, categories/2), max(5, timesteps/100)))
    # im = plt.imshow(data, aspect='auto')
    # Force smooth interpolation
    im = plt.imshow(
        data,
        aspect='auto',
        interpolation='bilinear',  # or 'bicubic' for even smoother
        origin='upper',
        vmin=0.0,
        vmax=1.0
    )
    plt.colorbar(im, label='Probability')

    plt.title(f'Router Optimal Position Distribution')
    plt.xlabel('Positions')
    plt.ylabel('Steps')

    # Reduce tick clutter for large numbers
    plt.xticks(range(categories))
    yticks = np.arange(0, timesteps, 20)
    plt.yticks(ticks=yticks, labels=[str(i) for i in yticks])

    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close()


def animate_router_distribution(router_distribution: List[np.ndarray], save_path="meta_distribution.gif"):
    fig, ax = plt.subplots(figsize=(12, 4), dpi=300)
    bar_container = ax.bar(range(len(router_distribution[0])), router_distribution[0])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Monomer Index")
    ax.set_ylabel("Mask Probability")
    ax.set_title("PPO Meta Learner Distribution Over Time")

    def update(frame):
        for bar, height in zip(bar_container, router_distribution[frame]):
            bar.set_height(height)
        ax.set_title(f"PPO Meta Learner - Step {frame}")
        return bar_container

    ani = animation.FuncAnimation(fig, update, frames=len(router_distribution), blit=False, interval=300)
    ani.save(save_path, writer='pillow', dpi=300)
    plt.close()

def get_monomer_atom_idx(chuckles: str) -> Dict[int, List[int]]:
    dummy_mol = Chem.MolFromSmiles(chuckles.replace('|', '[Ge]'))
    atom_dic, monomer_index, atom_index = {}, 0, 0

    for atom in dummy_mol.GetAtoms():
        if atom.GetSymbol() == 'Ge':
            monomer_index += 1
            continue
        atom_dic.setdefault(monomer_index, []).append(atom_index)
        atom_index += 1
    return atom_dic

def draw_highlighted_smiles(smiles, highlight_atoms=None, highlight_colors=None, highlight_radii=None, size=(800, 800), step: int = None):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise ValueError("Invalid SMILES string.")

    Chem.rdDepictor.Compute2DCoords(mol)

    highlight_atoms = highlight_atoms or []
    highlight_colors = highlight_colors or {}
    highlight_radii = highlight_radii or {idx: 0.4 for idx in highlight_atoms}

    highlight_bonds = []
    highlight_bond_colors = {}
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        if a1 in highlight_atoms and a2 in highlight_atoms:
            bond_idx = bond.GetIdx()
            c1 = highlight_colors.get(a1, (1, 0, 0))
            c2 = highlight_colors.get(a2, (1, 0, 0))
            avg_color = tuple((x + y) / 2 for x, y in zip(c1, c2))
            highlight_bond_colors[bond_idx] = avg_color
            highlight_bonds.append(bond_idx)

    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_colors,
        highlightBonds=highlight_bonds,
        highlightBondColors=highlight_bond_colors,
        highlightAtomRadii=highlight_radii,
    )
    drawer.FinishDrawing()
    img_data = drawer.GetDrawingText()

    mol_img = Image.open(io.BytesIO(img_data)).convert("RGB")

    if step is not None:
        # Create new image with extra space for the step text
        padding = 60
        new_img = Image.new("RGB", (mol_img.width, mol_img.height + padding), (255, 255, 255))
        new_img.paste(mol_img, (0, 0))

        draw = ImageDraw.Draw(new_img)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 30)
        except:
            font = ImageFont.load_default()

        text = f"Step {step}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = (mol_img.width - text_width) // 2
        y = mol_img.height + (padding - text_height) // 2

        draw.text((x, y), text, fill=(0, 0, 0), font=font)

        return new_img

    return mol_img


def get_mask_smiles_visualization(chuckles: str, probability_list, step: int = None):
    idx_dic = get_monomer_atom_idx(chuckles)

    smiles = chuckles.replace('|', '')
    mol = Chem.MolFromSmiles(smiles)
    color_dic = {i: (0.0, 0.0, 0.0, 0.0) for i, _ in enumerate(mol.GetAtoms())}
    radii_dic = {}

    top3_indices = sorted(range(len(probability_list)), key=lambda i: probability_list[i], reverse=True)[:3]
    top3_probs = [probability_list[i] for i in top3_indices]

    num1_idx, num2_idx, num3_idx = top3_indices
    num1_prob, num2_prob, num3_prob = top3_probs

    top1_atoms = idx_dic[num1_idx]
    top2_atoms = idx_dic[num2_idx]
    top3_atoms = idx_dic[num3_idx]

    def compute_radius(prob):
        return min(0.8, 0.3 + 0.6 * prob)  # radius grows with prob, capped at 0.8

    for i in top1_atoms:
        color_dic[i] = (1, 0.0, 0.0, min(1.0, 0.5 + num1_prob))
        radii_dic[i] = compute_radius(num1_prob)
    for i in top2_atoms:
        color_dic[i] = (1.0, 0.647, 0.0, min(1.0, 0.3 + num2_prob))
        radii_dic[i] = compute_radius(num2_prob)
    for i in top3_atoms:
        color_dic[i] = (1.0, 1.0, 0.0, min(1.0, 0.15 + num3_prob))
        radii_dic[i] = compute_radius(num3_prob)

    return draw_highlighted_smiles(
        smiles,
        highlight_atoms=list(color_dic.keys()),
        highlight_colors=color_dic,
        highlight_radii=radii_dic,
        step=step
    )


def save_probability_histogram(probabilities, save_path='histogram.png'):
    # Plot the histogram
    plt.figure(figsize=(12, 4))
    plt.bar(range(len(probabilities)), probabilities, color='blue', alpha=0.7)
    plt.xlabel('Index')
    plt.ylabel('Probability')
    plt.title('Histogram of Probabilities')
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Save the figure
    plt.savefig(save_path)
    plt.close()