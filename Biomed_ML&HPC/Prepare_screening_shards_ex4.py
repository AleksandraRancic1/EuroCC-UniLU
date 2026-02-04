# Exercise 4 - HPC screening with Slurm job arrays

# prepare_screening_shards.py

"""
Large molecular libraries cannot be processed by a single job or a single GPU.

Therefore, we must:
- split the library into independent chunks
- allow parallel processing
-avoid duplication or omission of molecules

This script:
- loads a large molecular dataset (ZINC)
- selects a manageable subset (budget-aware)
- splits molecules into N shards
-saves each shard as a .pt file
- enables one shard-> one slurm job -> one GPU


ZINC is more or less realistic chemical library;
it is much larger than ESOL;
represents what screening actually looks like
PyTorch geometrics already provides molecular graphs

ESOL was for training and debugging
ZINC is for screening
"""

import argparse
import os
import torch
from tqdm import tqdm
from torch_geometric.datasets import ZINC

from utils import ensure_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--out_dir", type=str, default="./screen_shards")
    ap.add_argument("--max_mols", type=int, default=200000) # adjust to the budget
    ap.add_argument("--num_shards", type=int, default=4)
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    ds = ZINC(root=args.data_root, subset=True) #subset=True keeps it lighter
    # ZINC in PyG provides splits via indices; subset=True is manageable.

    # use first max_mols for training practicality
    n = min(args.max_mols, len(ds))
    indices = list(range(n))

    shard_size = (n + args.num_shards - 1) // args.num_shards
    print(f"Preparing {args.num_shards} shards from {n} molecules (shard_size ~ {shard_size})")

    for shard_id in range(args.num_shards):
        start = shard_id * shard_size
        end = min(n, (shard_id + 1) * shard_size)
        shard_idx = indices[start:end]

        shard = [ds[i] for i in tqdm(shard_idx, desc=f"Shard {shard_id}")]
        out_path = os.path.join(args.out_dir, f"shard_{shard_id:04d}.pt")
        torch.save({"shard_id": shard_id, "data_list": shard}, out_path)
        print(f"Saved {len(shard)} -> {out_path}")
    
    print("Done.")

if __name__ == "__main__":
    main()
