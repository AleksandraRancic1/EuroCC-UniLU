#!/bin/bash
#SBATCH --job-name=EuroCC_exc2_gpu
#SBATCH --output=logs2/exc2_gpu_%j.out
#SBATCH --error=logs2/exc2_gpu_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1              
#SBATCH --cpus-per-task=16          # Number of CPU cores per task
#SBATCH --mem=64G                  # Total memory
#SBATCH --time=02:00:00            # Wall time limit

# Optional: helpful diagnostics
echo "Job started on $(hostname) at $(date)"
echo "Running on $SLURM_GPUS_ON_NODE GPU(s)"
nvidia-smi

# Initialize Conda (important for SLURM batch jobs)
eval "$(/home/users/arancic/miniconda3/bin/conda shell.bash hook)"

# Activate your environment
conda activate ml_env_39

python Exercise2_pt_GPU.py

echo "Job finished at $(date)"


