# AF3Score Pipeline

A pipeline for evaluating protein structure quality using AF3Score.

## Environment Setup

### 1. Create and Activate Conda Environment
```bash
conda create -n af3score python=3.11
conda activate af3score
conda install gxx_linux-64 gxx_impl_linux-64 gcc_linux-64 gcc_impl_linux-64=13.2.0
```


### 2. Install AF3Score and Dependencies
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


### 3. Optional Install HMMER (Required for MSA Generation)
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

## Usage Pipeline

The **AF3Score pipeline** is designed for high-throughput evaluation of protein structures. It consists of two primary scripts tailored for single-batch or multi-batch processing on high-performance computing (HPC) clusters.

### 1. Main Pipeline Script

`AF3score_pipeline.sh` is the core utility used to process a single directory of PDB files.

**Usage:**

Before running the pipeline on a shell cluster, you must configure the variables within `AF3score_pipeline.sh`.

| Variable | Description | Example Value |
| --- | --- | --- |
| `PYTHON_EXEC` | Path to the specific Conda environment Python binary. | `~/anaconda3/envs/af3score/bin/python` |
| `slurm_partition` | Target GPU partitions for job submission. | `gpu1,gpu2` |
| `slurm_nodelist` | Specific nodes assigned for the computation. | `c06b14n[05-06],c06b19n[05-06]` |

Run the pipeline:
```bash
./AF3score_pipeline.sh <input_pdb_dir> <output_dir> <num_jobs>

```

* **`<input_pdb_dir>`**: Path to the directory containing your input `.pdb` files.
* **`<output_dir>`**: Target directory where AF3Score metrics and results will be saved.
* **`<num_jobs>`**: The number of parallel jobs to launch.

### 2. Batch Processing

For users handling multiple datasets across several directories, use the multi-directory wrapper `AF3score_mutildir.sh`.


## Output Metrics

The pipeline generates the following scoring metrics:

| Metric | Level | Description |
| --- | --- | --- |
| **pTM** | Global / Per-chain | **Predicted TM-score:** Measures the overall topological accuracy of the global structure. |
| **ipTM** | Global / Inter-chain | **Interface pTM:** Assesses the accuracy of the interfaces between different protein chains. |
| **pLDDT** | Per-residue / Per-chain | **Predicted Local Distance Difference Test:** A per-residue confidence score (0-100). Higher values indicate higher local structure stability. |
| **PAE** | Per-chain | **Predicted Aligned Error:** The expected distance error (in Å) between pairs of residues. Lower values indicate higher confidence in relative positioning. |
| **ipSAE** | Inter-chain | **interaction prediction Score from Aligned Errors:** Specifically focuses on the binding interface of two chains. |

Global level metrics are evaluates the quality of the overall structure. Per-chain metrics are focused on the quality of individual chains. Inter-chain metrics are designed to assess the quality of the docking between two chains.

## Reference

For more information about AlphaFold3, please visit their [GitHub Repository](https://github.com/google-deepmind/alphafold3)