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



<img width="468" height="55" alt="image" src="https://github.com/user-attachments/assets/6b494c73-468d-41d9-b286-5e29e44be253" />

<img width="468" height="295" alt="image" src="https://github.com/user-attachments/assets/abde1f9a-b7e0-4fad-95ba-ca6b15fbcc54" />

<img width="468" height="152" alt="image" src="https://github.com/user-attachments/assets/b9a4f986-c6ea-4c17-a6ea-d1524e8d2bea" />


<img width="468" height="200" alt="image" src="https://github.com/user-attachments/assets/f16677b1-3699-4b1f-a5a8-2e9091984a14" />

