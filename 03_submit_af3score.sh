#!/bin/bash
#SBATCH -p gpu41,gpu43
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -J af3score

echo "========== Job started at: $(date) =========="
start_time=$(date +%s)


export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
export PATH=/lustre/grp/cmclab/share/wangd/hmmer/bin:$PATH
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=true"
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_CLIENT_MEM_FRACTION=0.95
export PATH=$(python -c "import site; print(site.getsitepackages()[0] + '/nvidia/cuda_nvcc/bin')"):$PATH


batch_json_dir=$1        
batch_h5_dir=$2    
output_dir=$3  
buckets=$(basename "$batch_json_dir" | grep -oE '[0-9]+$')
# buckets=256


echo "Running on: $batch_json_dir  $batch_h5_dir  $buckets -> $output_dir" 

$4 "$5/run_af3score.py"\
  --db_dir=/lustre/grp/cmclab/share/wangd/af3_data \
  --model_dir=/lustre/grp/cmclab/share/chenmc/Alphafold3params \
  --batch_json_dir="$batch_json_dir" \
  --batch_h5_dir="$batch_h5_dir" \
  --output_dir="$output_dir" \
  --run_data_pipeline=False \
  --run_inference=true \
  --init_guess=true \
  --num_samples=1 \
  --buckets="$buckets" \
  --write_cif_model=False \
  --write_summary_confidences=true \
  --write_full_confidences=true \
  --write_best_model_root=false \
  --write_ranking_scores_csv=false \
  --write_terms_of_use_file=false \
  --write_fold_input_json_file=false


end_time=$(date +%s)
elapsed=$((end_time - start_time))

echo "========== Job finished at: $(date) =========="
echo "========== Total runtime: ${elapsed} seconds =========="