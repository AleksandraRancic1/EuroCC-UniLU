#!/bin/bash
#SBATCH --job-name=ex1
#SBATCH --partition=cpu
#SBATCH --account=p201165
#SBATCH --qos=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=logs/ex1_%j.out
#SBATCH --error=logs/ex1_%j.err

mkdir -p logs
 
source /etc/profile

module --force purge

module load env/release/2024.1
module load Python/3.12.3-GCCcore-13.3.0
module load SciPy-bundle/2024.05-gfbf-2024a

python exercise1_baseline.py
