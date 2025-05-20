import json
import pandas as pd
import os

def format_msa_sequence(sequence):
    """Format MSA sequence"""
    return f">query\n{sequence}\n"

def get_chain_sequences(row):
    """Get all non-empty chain sequences from row data"""
    chain_sequences = []
    # Get all chain-related columns in their order of appearance in CSV
    chain_columns = [col for col in row.index if col.startswith('chain_') and col.endswith('_seq')]
    for col in chain_columns:
        if pd.notna(row[col]) and row[col] != '':
            # Extract chain ID from column name (e.g., 'A' from 'chain_A_seq')
            chain_id = col.split('_')[1]
            chain_sequences.append((chain_id, row[col]))
    return chain_sequences

def generate_json_files(csv_path, output_dir, cif_dir):
    """Generate JSON files from CSV file"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    json_count = 0
    
    # Process each row
    for _, row in df.iterrows():
        complex_name = row['complex']  # Get name from complex column
        
        # Get sequences for all chains
        chain_sequences = get_chain_sequences(row)
        
        if not chain_sequences:  # Skip if no valid chain sequences
            print(f"Warning: {complex_name} has no valid chain sequences")
            continue
        
        # Create a list of all chain sequences
        sequences = []
        for chain_id, sequence in chain_sequences:
            # Build cif file path
            cif_filename = f"{complex_name}_chain_{chain_id}.cif"
            cif_path = os.path.join(cif_dir, cif_filename)
            
            # Check if cif file exists
            if not os.path.exists(cif_path):
                print(f"Warning: {cif_filename} does not exist")
                continue
                
            sequences.append({
                "protein": {
                    "id": chain_id,
                    "sequence": sequence,
                    "modifications": [],
                    "unpairedMsa": format_msa_sequence(sequence),
                    "pairedMsa": format_msa_sequence(sequence),
                    "templates": [{
                        "mmcifPath": cif_path,
                        "queryIndices": list(range(len(sequence))),
                        "templateIndices": list(range(len(sequence)))
                    }]
                }
            })
        
        if not sequences:  # Skip if no valid sequence data
            print(f"Warning: {complex_name} has no valid sequence data")
            continue
        
        # Create complete JSON data
        json_data = {
            "dialect": "alphafold3",
            "version": 1,
            "name": complex_name,
            "sequences": sequences,
            "modelSeeds": [10],
            "bondedAtomPairs": None,
            "userCCD": None
        }
        
        # Generate output file name - REMOVED _data suffix
        output_filename = f"{complex_name}.json"  # Removed _data suffix to match H5 filename
        output_path = os.path.join(output_dir, output_filename)
        
        # Write JSON file
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        chain_ids = [chain[0] for chain in chain_sequences]
        print(f"Generated JSON file: {output_filename} (chains: {', '.join(chain_ids)})")
        json_count += 1
    
    print(f"\nComplete, generated {json_count} JSON files")

if __name__ == "__main__":
    csv_path = "./complex_chain_sequences.csv"  # Path to the CSV file just generated
    output_dir = "./complex_json_files"         # Output directory for JSON files
    cif_dir = "/lustre/grp/cmclab/liuyu/design/AF3Score/example/complex_chain_cifs"          # Directory where CIF files are located
    
    generate_json_files(csv_path, output_dir, cif_dir)