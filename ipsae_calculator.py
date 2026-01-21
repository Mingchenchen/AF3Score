import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union, Set

# --- Constants Definition ---
# Standard amino acid set (3-letter codes)
PROTEIN_RESIDUES: Set[str] = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}

# Nucleic acid residue set (Includes DNA and RNA)
NUCLEIC_ACIDS: Set[str] = {"DA", "DC", "DT", "DG", "A", "C", "U", "G"}

# Combined set for token identification
VALID_RESIDUES: Set[str] = PROTEIN_RESIDUES | NUCLEIC_ACIDS


def parse_pdb_atom_line(line: str) -> Optional[Dict[str, Union[int, str]]]:
    """
    Parses a single ATOM or HETATM line from a PDB file.
    Returns a dictionary of parsed values or None if the line format is invalid.
    """
    if len(line) < 54:
        return None

    try:
        # PDB format uses fixed-column widths
        return {
            "atom_num": int(line[6:11]),
            "atom_name": line[12:16].strip(),
            "residue_name": line[17:20].strip(),
            "chain_id": line[21].strip(),
            "residue_seq_num": int(line[22:26]),
        }
    except ValueError:
        return None


def load_af3_pae_and_chains(
    json_path: Union[str, Path], pdb_path: Union[str, Path]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts PAE matrix, chain IDs, and residue types from AF3 output files.

    Logic:
    1. Parse PDB to identify residues corresponding to AF3 Tokens (typically CA or C1').
    2. Read JSON to retrieve the raw PAE matrix.
    3. Use the mask generated from PDB to slice the PAE matrix to valid residues.
    """
    json_path = Path(json_path)
    pdb_path = Path(pdb_path)

    # 1. Parse PDB to construct the token mask
    token_mask = []
    chains = []
    residue_types = []

    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    with open(pdb_path, "r") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue

            atom = parse_pdb_atom_line(line)
            if atom is None:
                continue

            atom_name = atom["atom_name"]
            res_name = atom["residue_name"]

            # Token Logic:
            # 1. Protein Alpha Carbons (CA) or Nucleic Acid C1' atoms -> Token=1 (Keep)
            if atom_name == "CA" or (
                res_name in NUCLEIC_ACIDS and "C1" in atom_name
            ):
                token_mask.append(1)
                chains.append(atom["chain_id"])
                residue_types.append(res_name)

            # 2. Non-backbone atoms and non-standard residues (e.g., ligands/modifications)
            # would be marked as Token=0 if we needed to track them in the full PAE.
            # Standard residue side-chain atoms are ignored as they don't represent a Token.

    token_array = np.array(
        token_mask, dtype=bool
    )  # Convert to boolean for indexing
    chain_ids = np.array(chains)
    res_types = np.array(residue_types)

    # 2. Read JSON configuration
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    # Compatibility check for different AF3 JSON key naming conventions
    if "pae" in data:
        raw_pae = np.array(data["pae"])
    elif "predicted_aligned_error" in data:
        raw_pae = np.array(data["predicted_aligned_error"])
    else:
        # Handle cases where AF3 output might be a list containing the data
        if isinstance(data, list) and len(data) > 0 and "pae" in data[0]:
            raw_pae = np.array(data[0]["pae"])
        else:
            raise ValueError(
                f"Could not find 'pae' data in JSON file: {json_path}"
            )

    # 3. Validation and Slicing
    n_tokens = len(token_mask)
    n_pae = raw_pae.shape[0]

    if n_tokens != n_pae:
        print(
            f"[Warning] Token count from PDB ({n_tokens}) does not match PAE dimensions ({n_pae})."
        )
        print(
            "This may cause slicing errors. Ensure the PDB contains all atoms and matches the model."
        )

        # Fallback: crop to the smallest common dimension to prevent hard crashes
        min_dim = min(n_tokens, n_pae)
        token_array = token_array[:min_dim]
        raw_pae = raw_pae[:min_dim, :min_dim]

    # Perform dual-axis slicing using boolean indexing to keep only identified residues
    filtered_pae = raw_pae[np.ix_(token_array, token_array)]

    return filtered_pae, chain_ids, res_types


def _calc_d0_array(
    L_array: np.ndarray, pair_type: str = "protein"
) -> np.ndarray:
    """Calculates the d0 normalization factor (vectorized)."""
    # L is clamped at a minimum of 27.0
    L = np.maximum(27.0, L_array.astype(float))

    min_value = 2.0 if pair_type == "nucleic_acid" else 1.0

    # Formula: d0 = 1.24 * (L-15)^(1/3) - 1.8
    d0 = 1.24 * np.cbrt(L - 15.0) - 1.8
    return np.maximum(min_value, d0)


def _classify_chain_type(residue_types_subset: np.ndarray) -> str:
    """Classifies chain type: if it contains any nucleic acid residue, it's a nucleic_acid chain."""
    if np.isin(residue_types_subset, list(NUCLEIC_ACIDS)).any():
        return "nucleic_acid"
    return "protein"


def calculate_ipsae(
    pae_matrix: np.ndarray,
    chain_ids: np.ndarray,
    residue_types: Optional[np.ndarray] = None,
    pae_cutoff: float = 10.0,
) -> Dict[str, float]:
    """
    Calculates the ipSAE score.

    Optimization: Fully vectorized Mean PTM calculation, removing Python loops for better performance.
    """
    unique_chains = np.unique(chain_ids)
    scores = {}

    # Pre-determine chain types
    chain_type_map = {}
    if residue_types is not None:
        for chain in unique_chains:
            mask = chain_ids == chain
            chain_type_map[chain] = _classify_chain_type(
                residue_types[mask]
            )
    else:
        for chain in unique_chains:
            chain_type_map[chain] = "protein"

    # Iterate through all chain pairs
    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1 == chain2:
                continue

            # Determine interaction type for d0 calculation
            c1_type = chain_type_map[chain1]
            c2_type = chain_type_map[chain2]
            pair_type = (
                "nucleic_acid"
                if "nucleic_acid" in (c1_type, c2_type)
                else "protein"
            )

            # Extract sub-matrix for the pair
            mask_c1 = chain_ids == chain1
            mask_c2 = chain_ids == chain2

            # sub_pae shape: (N_residues_c1, N_residues_c2)
            sub_pae = pae_matrix[np.ix_(mask_c1, mask_c2)]

            if sub_pae.size == 0:
                scores[f"{chain1}_{chain2}"] = 0.0
                continue

            # 1. Identify valid interactions (contacts within cutoff)
            valid_mask = sub_pae < pae_cutoff  # Boolean matrix

            # 2. Calculate n0res (effective contact count per residue in Chain1)
            n0res_per_residue = np.sum(valid_mask, axis=1)

            # 3. Calculate d0 per residue
            d0_per_residue = _calc_d0_array(n0res_per_residue, pair_type)

            # 4. Calculate PTM matrix
            # Use broadcasting: (N, 1) against (N, M)
            ptm_matrix = 1.0 / (
                1.0 + (sub_pae / d0_per_residue[:, np.newaxis]) ** 2.0
            )

            # 5. Calculate ipSAE (Vectorized average)
            # We only average PTM values where valid_mask is True

            # Set invalid positions to 0 for the summation
            masked_ptm_sum = np.sum(ptm_matrix * valid_mask, axis=1)

            # Prevent division by zero for residues with no valid contacts
            with np.errstate(divide="ignore", invalid="ignore"):
                ipsae_per_residue = masked_ptm_sum / n0res_per_residue

            # Replace NaN (0/0 cases) with 0.0
            ipsae_per_residue = np.nan_to_num(ipsae_per_residue, nan=0.0)

            # 6. Take the maximum value as the final directional score
            final_score = (
                np.max(ipsae_per_residue)
                if ipsae_per_residue.size > 0
                else 0.0
            )

            scores[f"{chain1}_{chain2}"] = float(final_score)

    return scores


if __name__ == "__main__":
    # Example paths
    pdb_file = Path(
        "/Users/wanghongzhun/Documents/Code/AF3score/ipsae_test/7a0w_ef_b.pdb"
    )
    json_file = Path(
        "/Users/wanghongzhun/Documents/Code/AF3score/ipsae_test/7a0w_ef_b/seed-10_sample-0/confidences.json"
    )

    if pdb_file.exists() and json_file.exists():
        try:
            print(f"Processing: {pdb_file} ...")
            pae, chains, res_types = load_af3_pae_and_chains(
                json_file, pdb_file
            )

            # Debugging output
            print(f"PAE shape: {pae.shape}, Chains shape: {chains.shape}")

            results = calculate_ipsae(
                pae, chains, res_types, pae_cutoff=10
            )

            print("\nipSAE Scores (Directional Chain A -> Chain B):")
            for pair_id, score in results.items():
                # Display results in "Chain1 -> Chain2: Score" format
                c1, c2 = pair_id.split("_")
                print(f"  {c1} -> {c2}: {score:.4f}")

        except Exception as e:
            print(f"Error during processing: {e}")
    else:
        print("Example files not found. Please check your file paths.")
