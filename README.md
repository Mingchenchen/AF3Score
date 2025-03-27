# AF3Score Pipeline

A pipeline for evaluating protein structure quality using AF3Score.

## Environment Setup

Create the conda environment using the provided YAML file:
```bash
conda env create -f environment.yml
conda activate af3_scoring
```

## Usage Pipeline

### 1. Extract Chains and Generate CIF Files
```bash
python 1_extract_chains.py
```
**Input**: PDB files in `./pdb` directory  
**Output**: 
- Individual chain CIF files in `./complex_chain_cifs/`
- Sequence information in `complex_chain_sequences.csv`

### 2. Convert PDB to JAX Arrays
```bash
python 2_pdb2jax.py
```
**Input**: PDB files in `./pdb` directory  
**Output**: H5 files in `./complex_h5/`

### 3. Generate Configuration Files
```bash
python 3_generate_json.py
```
**Input**: `complex_chain_sequences.csv`  
**Output**: JSON configuration files in `./complex_json_files/`

### 4. Run AlphaFold3 Scoring
```bash
python run_diffusion.py \
  --db_dir=/path/to/alphafold_databases \
  --model_dir=/path/to/alphafold3_model_parameters \
  --json_path=/path/to/complex_json_files/your_protein_data.json \
  --output_dir=/path/to/score_results/ \
  --run_data_pipeline=False \
  --run_inference=true \
  --init_guess=true \
  --path=/path/to/complex_h5/your_protein.h5 \
  --num_samples=1
```

### 5. Extract Scoring Metrics
```bash
python 4_extract_iptm-ipae-pae-interaction.py
```

## Output Metrics

The pipeline generates the following scoring metrics:

| Metric | Description |
|--------|-------------|
| AF3Score_monomer_ca_plddt | pLDDT value for monomer CA atoms |
| AF3Score_monomer_pae | PAE value for monomers |
| AF3Score_monomer_ptm | PTM value for monomers |
| AF3Score_complex_ca_plddt | pLDDT value for complex CA atoms |
| AF3Score_complex_pae | PAE value for complexes |
| AF3Score_complex_ptm | PTM value for complexes |
| AF3Score_complex_iptm | iPTM value for complexes |
| AF3Score_pae_interaction | PAE value for interaction chains |
| AF3Score_ipae | PAE value for interfaces |

## Reference

For more information about AlphaFold3, please visit their [GitHub Repository](https://github.com/google-deepmind/alphafold3)