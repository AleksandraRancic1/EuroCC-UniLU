#!/bin/bash
#SBATCH --job-name=EuroCC_exc2
#SBATCH --output=logs2/exc2_cpu_%j.out
#SBATCH --error=logs2/exc2_cpu_%j.err
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16          # Number of CPU cores per task
#SBATCH --mem=64G                  # Total memory
#SBATCH --time=04:00:00            # Wall time limit

# Optional: helpful diagnostics
echo "Job started on $(hostname) at $(date)"
echo "Running on $SLURM_CPUS_ON_NODE CPU core(s)"

# Initialize Conda (important for SLURM batch jobs)
eval "$(/home/users/arancic/miniconda3/bin/conda shell.bash hook)"

# Activate your environment
conda activate ml_env_39

python -u Exercise2_pt.py

echo "Job finished at $(date)"

