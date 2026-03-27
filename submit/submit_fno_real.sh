#!/bin/bash
#SBATCH --partition=regularshort
#SBATCH --mem-per-cpu=4G
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

source /cvmfs/hpc.rug.nl/versions/2023.01/rocky8/x86_64/amd/zen3/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate gza

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

export CUDA_VISIBLE_DEVICES=""

python src/ns_model.py
