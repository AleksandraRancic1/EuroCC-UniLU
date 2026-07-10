salloc \
  --account=em4hpc_school_2026 \
  --reservation=hpda_cpu \
  --partition=batch \
  --qos=normal \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=2 \
  --mem=8G \
  --time=00:30:00
Load modules:

module –force purge
module load env/development/2024a
module load lang/Python/3.12.3-GCCcore-13.3.0
module load lang/SciPy-bundle/2024.05-gfbf-2024a
module load data/Arrow/17.0.0-gfbf-2024a

Create a virtual environment:

mkdir -p ~/venvs
python -m venv ~/venvs/hpda-2024a –system-site-packages

Activate the virtual environment
source ~/venvs/hpda-2024a/bin/activate
Check which Python is now active:
Which python 
Python --version
Install Dask:
python -m pip install "dask[distributed]==2024.9.0"

rsync -avz path-to-your-file/file-name.py iris-cluster:/path-to-your-folder


scripts
exercise 1
#!/bin/bash
#SBATCH --job-name=ex1
#SBATCH --account=em4hpc_school_2026
#SBATCH --reservation=hpda_cpu
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=112G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ex1_%j.out
#SBATCH --error=logs/ex1_%j.err

mkdir -p logs
 
source /etc/profile

module --force purge

module load env/development/2024a
module load lang/Python/3.12.3-GCCcore-13.3.0
module load lang/SciPy-bundle/2024.05-gfbf-2024a
module load data/Arrow/17.0.0-gfbf-2024a

python exercise1_baseline.py

exercise 2
#!/bin/bash
#SBATCH --job-name=ex2
#SBATCH --account=em4hpc_school_2026
#SBATCH --reservation=hpda_cpu
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=112G
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=logs/ex2_%j.out
#SBATCH --error=logs/ex2_%j.err

 
source /etc/profile

module --force purge

module load env/development/2024a
module load lang/Python/3.12.3-GCCcore-13.3.0
module load lang/SciPy-bundle/2024.05-gfbf-2024a
module load data/Arrow/17.0.0-gfbf-2024a
module load tools/dask/2024.9.0-gfbf-2024a

python exercise2_dask_singlenode.py
