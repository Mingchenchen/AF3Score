from Bio import PDB
from Bio.PDB import Structure, Model, Chain
from Bio.PDB import PDBParser, MMCIFIO
import os
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from pathlib import Path

# Define dictionary for three-letter to one-letter amino acid conversion
protein_letters_3to1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
    'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    'MSE': 'M'  # Selenomethionine is typically treated as methionine
}

def get_sequence(chain):
    """Get amino acid sequence of the chain"""
    sequence = ""
    for residue in chain:
        if residue.id[0] == ' ':
            try:
                resname = residue.get_resname().upper()
                sequence += protein_letters_3to1.get(resname, 'X')
            except:
                sequence += 'X'
    return sequence

def process_single_pdb(args):
    input_pdb, output_dir_cif = args
    try:
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("structure", input_pdb)
        base_name = os.path.splitext(os.path.basename(input_pdb))[0]
        
        chain_sequences = {}
        
        for chain in structure[0]:
            chain_id = chain.id
            sequence = get_sequence(chain)
            chain_sequences[chain_id] = sequence            
            new_structure = Structure.Structure("new_structure")
            new_model = Model.Model(0)
            new_structure.add(new_model)
            new_model.add(chain.copy())
            
            cif_io = MMCIFIO()
            cif_io.set_structure(new_structure)
            cif_output = os.path.join(output_dir_cif, f"{base_name}_chain_{chain_id}.cif")
            cif_io.save(cif_output)
        
        return base_name, chain_sequences
    except Exception as e:
        print(f"\nError processing {input_pdb}: {str(e)}")
        return None, None

def main():
    input_dir = "./pdb"  # Input directory
    output_dir_cif = "./complex_chain_cifs"  # CIF output directory
    
    # Create output directory
    os.makedirs(output_dir_cif, exist_ok=True)
    
    # Get all PDB files
    pdb_files = list(Path(input_dir).glob("*.pdb"))
    
    # Prepare parameters for process pool
    args = [(str(f), output_dir_cif) for f in pdb_files]
    
    # Process files using process pool
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = list(tqdm(
            pool.imap(process_single_pdb, args),
            total=len(pdb_files),
            desc="Processing PDB files"
        ))
    
    # Collect results
    sequences_dict = {}
    for base_name, chain_sequences in results:
        if base_name is not None:
            sequences_dict[base_name] = chain_sequences
    
    # Find all possible chain IDs and sort them by custom order
    all_chain_ids = set()
    for complex_data in sequences_dict.values():
        all_chain_ids.update(complex_data.keys())
    
    def chain_sort_key(chain_id):
        if chain_id.startswith('B'):
            return ('0', chain_id)
        elif chain_id.startswith('A'):
            return ('2', chain_id)
        else:
            return ('1', chain_id)
    
    all_chain_ids = sorted(list(all_chain_ids), key=chain_sort_key)
    
    # Create DataFrame
    rows = []
    for complex_name, chain_data in sequences_dict.items():
        row = {'complex': complex_name}
        for chain_id in all_chain_ids:
            row[f'chain_{chain_id}_seq'] = chain_data.get(chain_id, '')
        rows.append(row)
    
    df = pd.DataFrame(rows)
    cols = ['complex'] + [col for col in df.columns if col != 'complex']
    df = df[cols]
    
    # Save CSV file
    df.to_csv('complex_chain_sequences.csv', index=False)
    print("\nSequence information has been saved to complex_chain_sequences.csv")

if __name__ == "__main__":
    main()