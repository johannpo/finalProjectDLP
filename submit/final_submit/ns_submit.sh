#!/bin/bash
#SBATCH --job-name=base_model
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/s4099265/dlp_project/final_project/logs/%x_%j.out

set -euo pipefail

# Activate environment
source /cvmfs/hpc.rug.nl/versions/2023.01/rocky8/x86_64/amd/zen3/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate gza

# Project setup
PROJROOT="/scratch/s4099265/dlp_project/final_project"
cd $PROJROOT

# Create a timestamped directory for this specific run's artifacts
RUN_DIR="${PROJROOT}/output/final_data"
mkdir -p "$RUN_DIR/model" "$RUN_DIR/eval"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Pass the RUN_DIR to your python script as an argument
srun python src/reg_model/ns_model.py --out_dir "$RUN_DIR"
