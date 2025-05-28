#!/bin/bash
#SBATCH --job-name=EuroCC_exc1
#SBATCH --output=logs1/exc1_%j.out
#SBATCH --error=logs1/exc1_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1               # Request 1 GPUs
#SBATCH --cpus-per-task=4          # Number of CPU cores per task
#SBATCH --mem=64G                  # Total memory
#SBATCH --time=02:00:00            # Wall time limit

# Optional: helpful diagnostics
echo "Job started on $(hostname) at $(date)"
echo "Running on $SLURM_GPUS_ON_NODE GPU(s)"
nvidia-smi

source /etc/profile
module --force purge
module use /opt/apps/resif/iris/2020b/gpu/modules/all
module use /opt/apps/resif/iris/2020b/skylake/modules/all
module load env/deprecated/2020b
module load lib/TensorFlow/2.5.0-fosscuda-2020b

python Exercise1_tf.py

echo "Job finished at $(date)"
