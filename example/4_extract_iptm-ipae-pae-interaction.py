import pandas as pd
import json
import os
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
from collections import defaultdict

def get_interface_res_from_cif(cif_file, dist_cutoff=10):
    """Get interface residues from CIF file"""
    chain_coords = defaultdict(lambda: defaultdict(dict))
    
    with open(cif_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                parts = line.split()
                atom_name = parts[3].strip()
                chain_id = parts[6]
                residue_id = int(parts[8])
                x = float(parts[10])
                y = float(parts[11])
                z = float(parts[12])
                
                if atom_name == "CA":
                    chain_coords[chain_id][residue_id] = np.array([x, y, z])
    
    chain_A_res = sorted(chain_coords["A"].keys())
    chain_B_res = sorted(list(set(res for chain, res_dict in chain_coords.items() 
                               if chain != "A" for res in res_dict.keys())))
    
    chain_A_coords = np.array([chain_coords["A"][res] for res in chain_A_res])
    chain_B_coords = np.array([coords for chain, res_dict in chain_coords.items() 
                              if chain != "A" for coords in res_dict.values()])
    
    dist = np.sqrt(np.sum((chain_A_coords[:, None, :] - chain_B_coords[None, :, :])**2, axis=2))
    interface_residues = np.where(dist < dist_cutoff)
    
    interface_A = sorted(set(chain_A_res[i] for i in interface_residues[0]))
    interface_B = sorted(set(chain_B_res[i] for i in interface_residues[1]))
    # print(f"interface_A: {interface_A}")
    # print(f"interface_B: {interface_B}")
    return interface_A, interface_B

def get_metrics_from_json(base_path, target, description):
    """Get all metric values for a specific description"""
    base_json_path = Path(base_path) / target / description / "seed-10_sample-0"
    try:
        with open(base_json_path / "summary_confidences.json") as f:
            summary_data = json.load(f)
            complex_iptm = summary_data["iptm"]
            complex_ptm = summary_data["ptm"]
            monomer_ptm = summary_data["chain_ptm"][0] 

        residue_atoms = defaultdict(list)
        chain_ca = defaultdict(dict)
        
        with open(base_json_path / "model.cif", 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    parts = line.split()
                    residue_id = int(parts[8])
                    atom_name = parts[3].strip()
                    plddt = float(parts[14])
                    chain_id = parts[6]
                    
                    if atom_name == "CA":
                        chain_ca[chain_id][residue_id] = plddt
        
        a_chain_ca_plddts = list(chain_ca["A"].values())
        monomer_ca_plddt = np.mean(a_chain_ca_plddts)
        
        all_ca_plddts = []
        for chain_plddts in chain_ca.values():
            all_ca_plddts.extend(chain_plddts.values())
        complex_ca_plddt = np.mean(all_ca_plddts)
        
        cif_file = base_json_path / "model.cif"
        interface_A, interface_B = get_interface_res_from_cif(cif_file)
        
        with open(base_json_path / "confidences.json") as f:
            conf_data = json.load(f)
            pae_matrix = np.array(conf_data["pae"])
            token_chain_ids = conf_data["token_chain_ids"]
            token_res_ids = conf_data["token_res_ids"]
            
            interface_A, interface_B = get_interface_res_from_cif(base_json_path / "model.cif")
            interface_A_indices = [i for i, (res_id, chain) in enumerate(zip(token_res_ids, token_chain_ids))
                                 if chain == "A" and res_id in interface_A]
            interface_B_indices = [i for i, (res_id, chain) in enumerate(zip(token_res_ids, token_chain_ids))
                                 if chain != "A" and res_id in interface_B]
            chain_A_indices = [i for i, chain in enumerate(token_chain_ids) if chain == "A"]
            chain_B_indices = [i for i, chain in enumerate(token_chain_ids) if chain != "A"]
            ipae = np.mean(pae_matrix[np.ix_(interface_A_indices, interface_B_indices)])
            monomer_pae = np.mean(pae_matrix[np.ix_(chain_A_indices, chain_A_indices)])
            pae_interaction = np.mean(pae_matrix[np.ix_(chain_A_indices, chain_B_indices)])
            complex_pae = np.mean(pae_matrix)
            
        return {
            "AF3Score_monomer_ca_plddt": monomer_ca_plddt,
            "AF3Score_monomer_pae": monomer_pae,
            "AF3Score_monomer_ptm": monomer_ptm,
            "AF3Score_complex_ca_plddt": complex_ca_plddt,
            "AF3Score_complex_pae": complex_pae,
            "AF3Score_complex_ptm": complex_ptm,
            "AF3Score_complex_iptm": complex_iptm,
            "AF3Score_pae_interaction": pae_interaction,
            "AF3Score_ipae": ipae
        }, None
    except Exception as e:
        return None, f"{target}/{description} - Error: {e}"

def process_row(base_path, row):
    target = str(row['target']).strip()
    description = str(row['description']).strip()
    metrics, error = get_metrics_from_json(base_path, target, description)
    return {'idx': row.name, 'metrics': metrics, 'error': error}

def update_sc_file():
    df = pd.read_csv("subset_data.csv", sep=',', low_memory=False)
    print("Columns in file:", df.columns.tolist())

    if 'target' not in df.columns or 'description' not in df.columns:
        raise ValueError(f"Missing required 'target' or 'description' columns in CSV file\nAvailable columns: {df.columns.tolist()}")
    
    df['target'] = df['target'].astype(str)
    df['description'] = df['description'].astype(str)

    metrics_list = [
        'AF3Score_monomer_ca_plddt', 'AF3Score_monomer_pae', 'AF3Score_monomer_ptm',
        'AF3Score_complex_ca_plddt', 'AF3Score_complex_pae', 'AF3Score_complex_ptm', 
        'AF3Score_complex_iptm', 'AF3Score_pae_interaction', 'AF3Score_ipae'
    ]
    for metric in metrics_list:
        df[metric] = np.nan
    
    print(f"\nTotal {len(df)} records to process")
    
    # Base path
    base_path = "./score_results"
    
    # Set number of processes
    num_processes = max(1, int(cpu_count() * 0.8))
    print(f"Using {num_processes} processes")
    
    # Create process pool
    with Pool(num_processes) as pool:
        process_func = partial(process_row, base_path)
        results = list(tqdm(
            pool.imap(process_func, [row for _, row in df.iterrows()]),
            total=len(df),
            desc="Processing"
        ))
    
    # Update DataFrame and collect errors
    failed_records = []
    success_count = {metric: 0 for metric in metrics_list}
    
    for result in results:
        if result['metrics'] is not None:
            for metric, value in result['metrics'].items():
                df.at[result['idx'], metric] = value
                success_count[metric] += 1
        else:
            failed_records.append(f"line {result['idx']}: {result['error']}")
    
    # Write failed records to file
    with open('failed_records.txt', 'w') as f:
        f.write(f"Total processed rows: {len(df)}\n")
        f.write(f"Failed rows: {len(failed_records)}\n\n")
        f.write("Detailed failure records:\n")
        for record in failed_records:
            f.write(f"{record}\n")
    
    # Save updated file
    df.to_csv("subset_data_with_metrics.csv", sep=",", index=False)
    
    # Output statistics
    print("\nProcessing completion statistics:")
    print(f"Total entries: {len(df)}")
    for metric in metrics_list:
        print(f"Successfully updated {metric} count: {success_count[metric]}")
    print(f"Failed entries: {len(failed_records)}")
    print(f"Failed records written to: failed_records.txt")

if __name__ == "__main__":
    update_sc_file()