import os
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

# Disable Bio.PDB warnings
warnings.filterwarnings('ignore', category=PDBConstructionWarning)

# Define ATOM14
ATOM14: Dict[str, Tuple[str, ...]] = {
    'ALA': ('N', 'CA', 'C', 'O', 'CB','OXT'),
    'ARG': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2','OXT'),
    'ASN': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2','OXT'),
    'ASP': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2','OXT'),
    'CYS': ('N', 'CA', 'C', 'O', 'CB', 'SG','OXT'),
    'GLN': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2','OXT'),
    'GLU': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2','OXT'),
    'GLY': ('N', 'CA', 'C', 'O','OXT'),
    'HIS': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2','OXT'),
    'ILE': ('N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1','OXT'),
    'LEU': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2','OXT'),
    'LYS': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ','OXT'),
    'MET': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE','OXT'),
    'PHE': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ','OXT'),
    'PRO': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'CD','OXT'),
    'SER': ('N', 'CA', 'C', 'O', 'CB', 'OG','OXT'),
    'THR': ('N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2','OXT'),
    'TRP': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2','OXT'),
    'TYR': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH','OXT'),
    'VAL': ('N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2','OXT'),
    'UNK': (),
}  # pyformat: disable

# Define bucket list
#BUCKETS = [256, 512, 768, 1024, 1280, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120]
BUCKETS = [256, 512, 768, 1024, 1280, 1536, 2048, 2560, 3072]

def load_structure(file_path: str) -> Structure:
    """Load PDB or CIF file."""
    structure_id = os.path.basename(file_path).split('.')[0]
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(structure_id, file_path)
    return structure

def get_atom_coords(residue: PDB.Residue, atom_name: str) -> np.ndarray:
    """Get coordinates of specified atom, return zero vector if atom doesn't exist."""
    if atom_name in residue:
        return residue[atom_name].get_coord()
    return np.zeros(3)

def structure_to_array(structure: Structure, chain_ids: List[str] = None) -> np.ndarray:
    """Convert structure to atom coordinate array, process all chains, max 24 atom types per residue, following ATOM14 order."""
    model = structure[0]
    all_coords_list = []
    
    # If chain_ids not specified, process all chains
    if chain_ids is None:
        chain_ids = [chain.id for chain in model]
    
    for chain_id in chain_ids:
        if chain_id not in model:
            print(f"Warning: Chain {chain_id} not found in structure")
            continue
            
        chain = model[chain_id]
        chain_coords_list = []
        
        for res_idx, res in enumerate(chain):
            if not PDB.is_aa(res):
                continue
                
            # Get residue name
            resname = res.get_resname()
            if resname not in ATOM14:
                print(f"Warning: Residue {resname} not recognized. Skipping.")
                continue
            
            # Create 24-dimensional zero array
            res_coords_24 = np.zeros((24, 3))
            
            # Get atom order for current residue
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
            # Add filled residue coordinates to chain list
            chain_coords_list.append(res_coords_24)
        
        if chain_coords_list:
            chain_coords = np.stack(chain_coords_list, axis=0)
            all_coords_list.append(chain_coords)
    
    if not all_coords_list:
        raise ValueError("No valid coordinates found in any chain")
    
    coords = np.concatenate(all_coords_list, axis=0)
    return coords

def get_sequence_length(structure: Structure, chain_ids: List[str] = None) -> int:
    """Get total amino acid sequence length of all chains in structure."""
    model = structure[0]
    total_length = 0
    # If chain_ids not specified, process all chains
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
    """Find smallest bucket size greater than or equal to sequence length."""
    for bucket in buckets:
        if bucket >= seq_length:
            return bucket
    return buckets[-1]  # If all buckets are smaller than sequence length, return largest bucket

def save_traced_array(
    traced_array: jax.Array,
    seq_length: int,
    save_path: str,
    metadata: Dict = None
) -> None:
    """Save JAX traced array to file.
    
    Args:
        traced_array: JAX traced array
        seq_length: Sequence length
        save_path: Save path (.h5 format)
        metadata: Additional metadata dictionary
    """
    # Ensure file extension is .h5
    if not save_path.endswith('.h5'):
        save_path = save_path + '.h5'
        
    # Convert to numpy array
    numpy_array = np.array(jax.device_get(traced_array))
    
    # Save array and metadata
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('coordinates', data=numpy_array)
        f.create_dataset('seq_length', data=seq_length)
        f.create_dataset('shape', data=numpy_array.shape)
        
        # Save additional metadata
        if metadata:
            metadata_grp = f.create_group('metadata')
            for key, value in metadata.items():
                metadata_grp.attrs[key] = value

def load_traced_array(load_path: str) -> Tuple[jax.Array, int, Dict]:
    """Load JAX traced array from file.
    
    Args:
        load_path: Load file path (.h5 format)
        
    Returns:
        tuple: (JAX traced array, sequence length, metadata dictionary)
    """
    if not load_path.endswith('.h5'):
        load_path = load_path + '.h5'
        
    with h5py.File(load_path, 'r') as f:
        # Load main array
        numpy_array = f['coordinates'][:]
        
        # Load sequence length
        seq_length = int(f['seq_length'][()])
        
        # Load metadata
        metadata = {}
        if 'metadata' in f:
            metadata = dict(f['metadata'].attrs)
            
        # Convert to JAX array and add tracing
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
    max_length: int = 3072  # Add maximum length parameter
) -> Tuple[jnp.ndarray, int]:
    """Convert PDB file to JAX traced array."""
    structure = load_structure(pdb_path)
    
    if chain_ids is None:
        chain_ids = [chain.id for chain in structure[0]]
    
    seq_length = get_sequence_length(structure, chain_ids)
    
    
    # Print warning only if sequence is too long
    if seq_length > BUCKETS[-1]:
        print(f"Warning: {os.path.basename(pdb_path)} - Sequence length {seq_length} exceeds maximum allowed length {BUCKETS[-1]}")
        return None, seq_length
    
    target_length = find_bucket_size(seq_length, BUCKETS)
    coords = structure_to_array(structure, chain_ids)
    
    if seq_length < target_length:
        padding = np.zeros((target_length - seq_length, 24, 3))
        coords = np.concatenate([coords, padding], axis=0)
    
    coords_repeated = np.stack([coords] * num_copies)
    jax_array = jnp.array(coords_repeated)
    
    @jax.jit
    def get_traced_array(x):
        return x
    
    traced_array = get_traced_array(jax_array)
    
    if save_path:
        metadata = {
            'pdb_file': os.path.basename(pdb_path),
            'chain_ids': ','.join(chain_ids),
            'num_copies': num_copies,
            'original_length': seq_length,
            'padded_length': target_length
        }
        save_traced_array(traced_array, seq_length, save_path, metadata)
    
    return traced_array, seq_length

def process_single_file(args):
    """Function to process single file"""
    input_path, output_path, chain_ids, num_copies = args
    try:
        result = pdb_to_traced_array(
            pdb_path=input_path,
            chain_ids=chain_ids,
            num_copies=num_copies,
            save_path=output_path,
            max_length=3072  # Add maximum length restriction
        )
        if result[0] is None:
            return False
        return True
    except Exception as e:
        print(f"Error processing {os.path.basename(input_path)}: {str(e)}")
        return False

def process_pdb_folder(
    pdb_folder: str,
    output_folder: str,
    chain_ids: List[str] = None,
    num_copies: int = 5
) -> None:
    """Process all PDB format files in folder and convert to h5 format."""
    os.makedirs(output_folder, exist_ok=True)
    
    # Collect all files to be processed
    processing_args = []
    for filename in os.listdir(pdb_folder):
        if filename.endswith('.pdb'):
            input_path = os.path.join(pdb_folder, filename)
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.h5")
            processing_args.append((input_path, output_path, chain_ids, num_copies))
    
    if not processing_args:
        print("No valid files to process")
        return
    
    print(f"Processing {len(processing_args)} files")
    
    # Process files sequentially
    for args in tqdm(processing_args, desc="Processing files"):
        input_path, output_path, chain_ids, num_copies = args
        try:
            result = pdb_to_traced_array(
                pdb_path=input_path,
                chain_ids=chain_ids,
                num_copies=num_copies,
                save_path=output_path,
                max_length=3072
            )
            if result[0] is None:
                print(f"Skipped {os.path.basename(input_path)} due to length")
        except Exception as e:
            print(f"Error processing {os.path.basename(input_path)}: {str(e)}")

def main():
    """Main function"""
    pdb_folder = "./pdb"
    output_folder = "./complex_h5"
    
    process_pdb_folder(
        pdb_folder=pdb_folder,
        output_folder=output_folder,
        chain_ids=None,
        num_copies=1
    )

if __name__ == '__main__':
    main()