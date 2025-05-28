#!/bin/bash
#SBATCH --job-name=EuroCC_exc3_DDP
#SBATCH --output=logs3/exc3_%j.out
#SBATCH --error=logs3/exc3_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8         # Number of CPU cores per task
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

export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((12000 + RANDOM % 1000))

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo,docker0

srun python Exercise3.py

echo "Job finished at $(date)"
