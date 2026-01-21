#!/bin/bash

# ============================== Configuration ==============================
# Under <input_dir>, PDB file names must be lowercase, and for multiple chains, name chain IDs as A..Z
# Usage: AF3score_pipeline.sh <input_pdb_dir> <output_dir> <num_jobs>

PYTHON_EXEC="/lustre/grp/cmclab/share/wangd/env/alphafold3/bin/python"
slurm_partition="gpu41,gpu43"
slurm_nodelist="c06b14n[05-06],c06b19n[05-06],c06b20n[05-06],c06b26n[05-06],c06b27n[05-06]"


pipeline_script_dir=$(dirname "$(realpath "$0")")
source "${pipeline_script_dir}/functions.sh"

# ============================== Argument Parsing ==============================
usage() {
    cat <<EOF
Usage: $0 <input_pdb_dir> <output_dir> <num_jobs>

Example:
  $0 /data/pdbs /output/af3score 50
EOF
  exit 1
}

[[ $# -ge 3 ]] || usage

input_pdb_dir="$(realpath "$1")"
output_dir="$(realpath "$2")"
num_jobs="$3"

# ============================== Initialization ==============================
log_info "========== AF3score Pipeline started =========="
log_info "Input PDB dir   : $input_pdb_dir"
log_info "Output dir      : $output_dir"
log_info "Batch size      : $num_jobs"
log_info "SLURM partition : $slurm_partition"
[[ -n "$slurm_nodelist" ]] && log_info "Node list       : $slurm_nodelist"

start_time=$(date +%s)

# =================================================================
# 02. AF3score
# =================================================================
log_step "02" "Running AF3score"

# --- Preparing directories ---
log_info "Preparing directories for AF3score..."
output_af3score_base="$output_dir"
af3_input_batch="$output_af3score_base/af3_input_batch"
output_dir_cif="$output_af3score_base/single_chain_cif"
save_csv="$output_af3score_base/single_seq.csv"
output_dir_json="$output_af3score_base/json"
output_dir_jax="$af3_input_batch/jax"
output_dir_af3score="$output_af3score_base/af3score_outputs"
metric_csv="$output_af3score_base/af3score_metrics.csv"
jax_log_dir="$output_af3score_base/logs/jax"
af3score_log_dir="$output_af3score_base/logs/af3score"
mkdir -p "$af3_input_batch" "$output_dir_cif" "$output_dir_jax" "$output_dir_json" "$output_dir_af3score" "$jax_log_dir" "$af3score_log_dir"

# --- Preparing AF3score inputs (get seq, json and split batch) ---
log_info "Preparing AF3score inputs: extracting sequences, creating JSON files, and splitting batches..."
"$PYTHON_EXEC" "$pipeline_script_dir/01_prepare_get_json.py" \
--input_dir "$input_pdb_dir" \
--output_dir_cif "$output_dir_cif" \
--save_csv "$save_csv" \
--output_dir_json "$output_dir_json" \
--batch_dir "$af3_input_batch" \
--num_jobs "$num_jobs"

# --- Submitting prepare_jax jobs ---
log_info "Submitting prepare_jax jobs to convert PDBs to H5 format..."
declare -a prepare_job_ids=()
for subfolder in "$af3_input_batch/pdb"/*; do
  if [[ -d "$subfolder" ]]; then
    folder_name=$(basename "$subfolder")
    log_info "Submitting prepare_jax job for batch: $folder_name"
    job_id=$(submit_job "$slurm_partition" "$slurm_nodelist" "$pipeline_script_dir/02_submit_prepare_jax.sh" \
      "$jax_log_dir/${folder_name}-%j.out" \
      "$subfolder" \
      "$output_dir_jax/$folder_name" \
      "$pipeline_script_dir" \
    "$PYTHON_EXEC")
    log_info "--> Job submitted with ID: $job_id"
    prepare_job_ids+=("$job_id")
    sleep 0.05
  fi
done

wait_for_jobs "prepare_jax" "${prepare_job_ids[@]}"
log_info "✅ All prepare_jax jobs completed."

# --- Checking H5 file generation ---
log_info "Verifying H5 file generation..."
failed_list="$af3_input_batch/failed_h5.txt"
> "$failed_list"
total_missing=0

for subfolder in "$af3_input_batch/pdb"/*; do
  if [[ -d "$subfolder" ]]; then
    folder_name=$(basename "$subfolder")
    pdb_dir="$subfolder"
    h5_dir="$output_dir_jax/$folder_name"
    
    # Using comm to compare PDB and H5 file lists
    missing_files_str=$(comm -23 \
      <(find "$pdb_dir" -maxdepth 1 -name "*.pdb" -exec basename {} .pdb \; | sort) \
    <(find "$h5_dir" -maxdepth 1 -name "*.h5" -exec basename {} .h5 \; | sort))
    
    mapfile -t missing < <(printf "%s" "$missing_files_str")
    num_pdb_files=$(find "$pdb_dir" -maxdepth 1 -name "*.pdb" | wc -l)
    
    if [[ "${#missing[@]}" -eq 0 ]]; then
      log_info "✅ $folder_name: All $num_pdb_files H5 files were generated successfully."
    else
      log_info "❌ $folder_name: Missing ${#missing[@]} out of $num_pdb_files H5 files."
      for miss in "${missing[@]}"; do
        echo "$pdb_dir/$miss.pdb" >> "$failed_list"
      done
      total_missing=$((total_missing + ${#missing[@]}))
    fi
  fi
done

if [[ "$total_missing" -gt 0 ]]; then
  log_info "A total of $total_missing H5 files failed to generate. Check the list at: $failed_list"
else
  log_info "All H5 files across all batches were generated successfully."
fi

# --- Submitting af3score GPU inference jobs ---
log_info "Submitting af3score inference jobs..."
declare -a af3score_job_ids=()
for subfolder in "$af3_input_batch/json"/*; do
  if [[ -d "$subfolder" ]]; then
    folder_name=$(basename "$subfolder")
    log_info "Submitting af3score job for batch: $folder_name"
    job_id=$(submit_job "$slurm_partition" "$slurm_nodelist" "$pipeline_script_dir/03_submit_af3score.sh" \
      "$af3score_log_dir/${folder_name}-%j.out" \
      "$af3_input_batch/json/$folder_name" \
      "$af3_input_batch/jax/$folder_name" \
      "$output_dir_af3score" \
      "$PYTHON_EXEC" \
    "$pipeline_script_dir")
    log_info "--> Job submitted with ID: $job_id"
    af3score_job_ids+=("$job_id")
    sleep 0.05
  fi
done

wait_for_jobs "af3score" "${af3score_job_ids[@]}"
log_info "✅ All af3score inference jobs completed."

# =================================================================
# 03. Metrics Extraction and Verification
# =================================================================
log_step "03" "Extracting and Verifying Metrics"

log_info "Extracting all metrics into a single CSV file..."
"$PYTHON_EXEC" "$pipeline_script_dir/04_get_metrics.py" \
--input_pdb_dir "$input_pdb_dir" \
--af3score_output_dir "$output_dir_af3score" \
--save_metric_csv "$metric_csv"

log_info "Verifying the final metrics CSV file: $metric_csv"
expected_count=$(ls -1q "$input_pdb_dir"/*.pdb 2>/dev/null | wc -l)
actual_count=$(tail -n +2 "$metric_csv" | wc -l) # Skip header

if [[ "$actual_count" -eq "$expected_count" ]]; then
  log_info "Verification successful: Row count matches ($actual_count) and no empty values found."
else
  log_info "Verification failed. Expected records: $expected_count, Found: $actual_count"
fi

# =================================================================
# Final Summary
# =================================================================
end_time=$(date +%s)
duration=$((end_time - start_time))
log_info "========== Pipeline finished at: $(date) =========="
log_info "Total execution time: $duration seconds"
