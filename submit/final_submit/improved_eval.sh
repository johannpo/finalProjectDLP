#!/bin/bash
#SBATCH --job-name=eval_baseline
#SBATCH --partition=regularshort
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/s4099265/dlp_project/final_project/logs/%x_%j.out

set -euo pipefail

# Activate environment
source /cvmfs/hpc.rug.nl/versions/2023.01/rocky8/x86_64/amd/zen3/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate gza

# Project setup
PROJROOT="/scratch/s4099265/dlp_project/final_project"
cd $PROJROOT

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Pass the RUN_DIR to your python script as an argument
srun python src/improved_model/eval.py
