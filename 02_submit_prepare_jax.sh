#!/bin/bash
#SBATCH -p gpu41,gpu43
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -J jax

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

$4 "$3/02_prepare_pdb2jax.py" \
      --pdb_folder $1 \
      --output_folder $2 \
      --num_workers 12
