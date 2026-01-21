import os
import time
from typing import Dict, Tuple, List
import numpy as np
import jax
import jax.numpy as jnp
from Bio import PDB
from Bio.PDB.Structure import Structure
import h5py
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import argparse

# Suppress warnings from Bio.PDB during parsing
warnings.filterwarnings("ignore", category=PDBConstructionWarning)

# Define ATOM14: Mapping of amino acids to their heavy atom names (up to 14 atoms + OXT)
# This mapping is used to standardize the atom order in the coordinate arrays.
ATOM14: Dict[str, Tuple[str, ...]] = {
    "ALA": ("N", "CA", "C", "O", "CB", "OXT"),
    "ARG": (
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "NE",
        "CZ",
        "NH1",
        "NH2",
        "OXT",
    ),
    "ASN": ("N", "CA", "C", "O", "CB", "CG", "OD1", "ND2", "OXT"),
    "ASP": ("N", "CA", "C", "O", "CB", "CG", "OD1", "OD2", "OXT"),
    "CYS": ("N", "CA", "C", "O", "CB", "SG", "OXT"),
    "GLN": ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2", "OXT"),
    "GLU": ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2", "OXT"),
    "GLY": ("N", "CA", "C", "O", "OXT"),
    "HIS": (
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "ND1",
        "CD2",
        "CE1",
        "NE2",
        "OXT",
    ),
    "ILE": ("N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", "OXT"),
    "LEU": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "OXT"),
    "LYS": ("N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "OXT"),
    "MET": ("N", "CA", "C", "O", "CB", "CG", "SD", "CE", "OXT"),
    "PHE": (
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE1",
        "CE2",
        "CZ",
        "OXT",
    ),
    "PRO": ("N", "CA", "C", "O", "CB", "CG", "CD", "OXT"),
    "SER": ("N", "CA", "C", "O", "CB", "OG", "OXT"),
    "THR": ("N", "CA", "C", "O", "CB", "OG1", "CG2", "OXT"),
    "TRP": (
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "NE1",
        "CE2",
        "CE3",
        "CZ2",
        "CZ3",
        "CH2",
        "OXT",
    ),
    "TYR": (
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE1",
        "CE2",
        "CZ",
        "OH",
        "OXT",
    ),
    "VAL": ("N", "CA", "C", "O", "CB", "CG1", "CG2", "OXT"),
    "UNK": (),
}


def load_structure(file_path: str) -> Structure:
    """Loads a PDB or CIF file and returns a Biopython Structure object."""
    structure_id = os.path.basename(file_path).split(".")[0]
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(structure_id, file_path)
    return structure


def get_atom_coords(residue: PDB.Residue, atom_name: str) -> np.ndarray:
    """Gets coordinates for a specific atom; returns a zero vector if not found."""
    if atom_name in residue:
        return residue[atom_name].get_coord()
    return np.zeros(3)


def structure_to_array(
    structure: Structure, chain_ids: List[str] = None
) -> np.ndarray:
    """
    Converts structure to a coordinate array.
    Each residue is mapped to a 24-atom slot (padded with zeros) based on ATOM14 order.
    """
    model = structure[0]
    all_coords_list = []

    # If no specific chains are provided, process all chains in the model
    if chain_ids is None:
        chain_ids = [chain.id for chain in model]

    for chain_id in chain_ids:
        if chain_id not in model:
            print(f"Warning: Chain {chain_id} not found in structure")
            continue

        chain = model[chain_id]
        chain_coords_list = []

        for res in chain:
            if not PDB.is_aa(res):
                continue

            resname = res.get_resname()
            if resname not in ATOM14:
                print(
                    f"Warning: Residue {resname} not recognized. Skipping."
                )
                continue

            # Initialize fixed-size array (24 atoms max per residue)
            res_coords_24 = np.zeros((24, 3))
            atom_order = ATOM14[resname]

            atom_index = 0
            for atom_name in atom_order:
                if atom_name in res:
                    coord = res[atom_name].get_coord()
                    if atom_index < 24:
                        res_coords_24[atom_index] = coord
                        atom_index += 1
                    else:
                        break

            chain_coords_list.append(res_coords_24)

        if chain_coords_list:
            chain_coords = np.stack(chain_coords_list, axis=0)
            all_coords_list.append(chain_coords)

    if not all_coords_list:
        raise ValueError("No valid coordinates found in any chain")

    # Concatenate all chains into a single array: [Total_Residues, 24, 3]
    coords = np.concatenate(all_coords_list, axis=0)
    return coords


def get_sequence_length(
    structure: Structure, chain_ids: List[str] = None
) -> int:
    """Calculates the total amino acid sequence length across specified chains."""
    model = structure[0]
    total_length = 0
    if chain_ids is None:
        chain_ids = [chain.id for chain in model]
    for chain_id in chain_ids:
        if chain_id not in model:
            print(f"Warning: Chain {chain_id} not found in structure")
            continue
        chain = model[chain_id]
        total_length += len([res for res in chain if PDB.is_aa(res)])
    return total_length


def find_bucket_size(seq_length: int, buckets: List[int]) -> int:
    """Finds the smallest bucket size that is greater than or equal to the sequence length."""
    for bucket in buckets:
        if bucket >= seq_length:
            return bucket
    return buckets[
        -1
    ]  # Return max bucket if sequence exceeds all predefined sizes


def save_traced_array(
    traced_array: jax.Array,
    seq_length: int,
    save_path: str,
    metadata: Dict = None,
) -> None:
    """
    Saves a JAX-traced array to an HDF5 file including metadata.

    Args:
        traced_array: JAX array containing coordinates.
        seq_length: Original length of the sequence before padding.
        save_path: Destination path (.h5).
        metadata: Dictionary containing provenance data.
    """
    if not save_path.endswith(".h5"):
        save_path = save_path + ".h5"

    # Transfer from device to host
    numpy_array = np.array(jax.device_get(traced_array))

    with h5py.File(save_path, "w") as f:
        f.create_dataset("coordinates", data=numpy_array)
        f.create_dataset("seq_length", data=seq_length)
        f.create_dataset("shape", data=numpy_array.shape)

        if metadata:
            metadata_grp = f.create_group("metadata")
            for key, value in metadata.items():
                metadata_grp.attrs[key] = value


def load_traced_array(load_path: str) -> Tuple[jax.Array, int, Dict]:
    """Loads an HDF5 file and converts the stored array back into a JAX-traced array."""
    if not load_path.endswith(".h5"):
        load_path = load_path + ".h5"

    with h5py.File(load_path, "r") as f:
        numpy_array = f["coordinates"][:]
        seq_length = int(f["seq_length"][()])
        metadata = dict(f["metadata"].attrs) if "metadata" in f else {}

        # Re-trace using JIT to simulate JAX behavior
        @jax.jit
        def get_traced_array(x):
            return x

        jax_array = get_traced_array(jnp.array(numpy_array))
        return jax_array, seq_length, metadata


def pdb_to_traced_array(
    pdb_path: str,
    chain_ids: List[str] = None,
    num_copies: int = 5,
    save_path: str = None,
    max_length: int = 3072,
) -> Tuple[jnp.ndarray, int]:
    """Processes a PDB file: parses, pads to bucket size, repeats, and converts to JAX."""
    structure = load_structure(pdb_path)

    if chain_ids is None:
        chain_ids = [chain.id for chain in structure[0]]

    seq_length = get_sequence_length(structure, chain_ids)

    # Validate against max bucket size
    if seq_length > BUCKETS[-1]:
        print(
            f"Warning: {os.path.basename(pdb_path)} - Sequence length {seq_length} exceeds maximum allowed length {BUCKETS[-1]}"
        )
        return None, seq_length

    target_length = find_bucket_size(seq_length, BUCKETS)
    coords = structure_to_array(structure, chain_ids)

    # Pad with zeros to meet bucket size
    if seq_length < target_length:
        padding = np.zeros((target_length - seq_length, 24, 3))
        coords = np.concatenate([coords, padding], axis=0)

    # Tile the array to create multiple copies
    coords_repeated = np.stack([coords] * num_copies)
    jax_array = jnp.array(coords_repeated)

    @jax.jit
    def get_traced_array(x):
        return x

    traced_array = get_traced_array(jax_array)

    if save_path:
        metadata = {
            "pdb_file": os.path.basename(pdb_path),
            "chain_ids": ",".join(chain_ids),
            "num_copies": num_copies,
            "original_length": seq_length,
            "padded_length": target_length,
        }
        save_traced_array(traced_array, seq_length, save_path, metadata)

    return traced_array, seq_length


def process_single_file(args):
    """Wrapper function for processing a single file, used by multiprocessing."""
    input_path, output_path, chain_ids, num_copies = args
    try:
        result = pdb_to_traced_array(
            pdb_path=input_path,
            chain_ids=chain_ids,
            num_copies=num_copies,
            save_path=output_path,
            max_length=3072,
        )
        if result[0] is None:
            return (False, input_path)
        return (True, input_path)
    except Exception as e:
        print(f"Error processing {os.path.basename(input_path)}: {str(e)}")
        return (False, input_path)


def process_pdb_folder(
    pdb_folder: str,
    output_folder: str,
    chain_ids: List[str] = None,
    num_copies: int = 5,
    num_workers: int = None,
) -> None:
    """Parallel processing of a directory containing PDB files."""
    os.makedirs(output_folder, exist_ok=True)

    # Prepare list of arguments for the pool
    processing_args = []
    for filename in os.listdir(pdb_folder):
        if filename.endswith(".pdb"):
            input_path = os.path.join(pdb_folder, filename)
            output_path = os.path.join(
                output_folder, f"{os.path.splitext(filename)[0]}.h5"
            )
            processing_args.append(
                (input_path, output_path, chain_ids, num_copies)
            )

    if not processing_args:
        print("No valid files to process.")
        return

    print(f"🔵 Preparing to process {len(processing_args)} files...")
    print(f"🔵 Using {num_workers} processes for parallel execution.")

    # Initial processing pass
    with mp.Pool(processes=num_workers) as pool:
        results = list(
            tqdm(
                pool.imap(process_single_file, processing_args),
                total=len(processing_args),
                desc="Processing files (1st run)",
            )
        )

    # Separate success and failure
    success_paths = [path for success, path in results if success]
    failed_paths = [path for success, path in results if not success]

    print(f"✅ Successfully processed: {len(success_paths)} files")
    print(f"❌ Failed: {len(failed_paths)} files")

    # Retry logic for failed files (often handles transient I/O issues)
    if failed_paths:
        print(f"🟡 Starting retry for {len(failed_paths)} failed files...")

        retry_args = []
        for failed_file in failed_paths:
            filename = os.path.basename(failed_file)
            output_path = os.path.join(
                output_folder, f"{os.path.splitext(filename)[0]}.h5"
            )
            retry_args.append(
                (failed_file, output_path, chain_ids, num_copies)
            )

        with mp.Pool(processes=num_workers) as pool:
            retry_results = list(
                tqdm(
                    pool.imap(process_single_file, retry_args),
                    total=len(retry_args),
                    desc="Reprocessing failed files (2nd run)",
                )
            )

        retry_success = [
            path for success, path in retry_results if success
        ]
        retry_fail = [
            path for success, path in retry_results if not success
        ]

        print(f"🟢 Retry success: {len(retry_success)} files")
        print(f"🔴 Still failed: {len(retry_fail)} files")


def main():
    """Main execution entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdb_folder",
        type=str,
        required=True,
        help="Input directory containing PDB files",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Output directory for H5 files",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel processes (default: 4)",
    )
    args = parser.parse_args()

    pdb_folder = args.pdb_folder
    # Extract bucket size from the folder name suffix (e.g., folder_name_512)
    try:
        bucket = int(os.path.basename(pdb_folder).split("_")[-1])
    except (ValueError, IndexError):
        bucket = 3072
        print(
            f"Warning: Could not detect bucket from folder name, using default: {bucket}"
        )

    global BUCKETS
    BUCKETS = [bucket]
    print(f"🎯 Detected bucket value from folder name: {bucket}")

    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)
    num_workers = args.num_workers

    process_pdb_folder(
        pdb_folder=pdb_folder,
        output_folder=output_folder,
        chain_ids=None,
        num_copies=1,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    main()
