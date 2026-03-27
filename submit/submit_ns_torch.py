#!/bin/bash
#SBATCH --partition=gpumedium
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/s4099265/dlp_project/final_project/logs/%x_%j.out

set -euo pipefail

source /cvmfs/hpc.rug.nl/versions/2023.01/rocky8/x86_64/amd/zen3/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate gza

cd /scratch/s4099265/dlp_project/final_project

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python src/ns_model.py
