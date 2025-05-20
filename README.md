# AF3Score Pipeline

A pipeline for evaluating protein structure quality using AF3Score.

## Environment Setup

### 1. Create and Activate Conda Environment
```bash
conda create -n af3score python=3.11
conda activate af3score
conda install gxx linux-64 gxx_impl linux-64 gcc linux-64 gcc_impl linux-64-13.2.0
```

### 2. Install HMMER (Required for MSA Generation)
```bash
mkdir ~/hmmer_build ~/hmmer
wget http://eddylab.org/software/hmmer/hmmer-3.4.tar.gz -P ~/hmmer_build
cd ~/hmmer_build
tar -zxf hmmer-3.4.tar.gz
cd hmmer-3.4
./configure --prefix=~/hmmer
make -j8
make install
```

Add HMMER to your PATH:
```bash
export PATH="~/hmmer/bin:$PATH"
```

Verify installation:
```bash
hmmsearch -h
```

### 3. Install AF3Score and Dependencies
```bash
git clone https://github.com/Mingchenchen/AF3Score.git
cd AF3Score/

# Download required databases
bash fetch_databases.sh <DB_DIR>  # Replace <DB_DIR> with your database directory

# Install Python dependencies
pip install -r dev-requirements.txt
pip install --no-deps -e .
build_data

# Install additional dependencies
conda install -c conda-forge biopython h5py pandas
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

AlphaFold3 scoring can be run in two modes: single file mode or batch mode.

#### Single File Mode

Use this mode when you want to process a single protein structure:

```bash
python run_af3score.py \
  --db_dir=/path/to/alphafold_databases \
  --model_dir=/path/to/alphafold3_model_parameters \
  --json_path=/path/to/complex_json_files/your_protein.json \
  --path=/path/to/complex_h5/your_protein.h5 \
  --output_dir=/path/to/score_results/ \
  --run_data_pipeline=False \
  --run_inference=true \
  --init_guess=true \
  --num_samples=1
```

#### Batch Mode

Use this mode to process multiple protein structures at once:

```bash
python run_af3score.py \
  --db_dir=/path/to/alphafold_databases \
  --model_dir=/path/to/alphafold3_model_parameters \
  --batch_json_dir=/path/to/complex_json_files/ \
  --batch_h5_dir=/path/to/complex_h5/ \
  --output_dir=/path/to/score_results/ \
  --run_data_pipeline=False \
  --run_inference=true \
  --init_guess=true \
  --num_samples=1
```

**Important Notes:**
- In batch mode, JSON and H5 files must have matching filenames (e.g., `protein1.json` and `protein1.h5`)
- You cannot use both modes simultaneously (e.g., specify both `--json_path` and `--batch_json_dir`)
- Batch mode efficiently processes multiple structures by loading the chemical component dictionary (CCD) only once

## Computational Efficiency

Our method consists of two main stages: data preprocessing and model inference. The preprocessing stage runs on CPU and includes two steps: structure processing and coordinate conversion. These CPU-based steps are computationally efficient, taking less than 0.3 seconds combined even for proteins with 1024 residues. For the model inference stage, which runs on GPU (NVIDIA GeForce RTX 4090), the computational time scales with protein sequence length, ranging from 20 seconds for proteins with 256 residues to 60 seconds for those with 1024 residues.

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