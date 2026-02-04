# exercise3_screen_shard.py


"""
The goal of this exercise is to show how a trained AI model is applied
to a large chemical library using:
- slurm job arrays
- one GPU per node
- independent data shards
- no inter-process communication
- fully scalable inference

Workflow:
1. Prepare shards of large molecular library (ZINC)
    previous step (prepare_screening_shards.py)
2. Launch a Slurm job array
    * Each job:
        - loads one shard
        - loads the trained model
        - runs inference on GPU
        - writes predictions to disk
3. Aggregate and rank results
    next step (aggregate_rank_hits.py)

This script runs AI inference on one shard of molecules and writes predicted scores to disk,
designed to be executed as one Slurm job in a job array.

"""

import argparse
import os
import csv
import torch
from torch.geometric.loader import DataLoader
from tqdm import tqdm

from models import GINRegressor
from utils import get_device, ensure_dir

def load_model(ckpt_path: str, device):
    """
    loads the trained model checkpoint and prepares it for inference
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # loads the .pt file produced in Exercise 2
    # ensures portability

    model = GINRegressor(
        in_dim=ckpt["in_dim"],
        hidden_dim=ckpt["hidden_dim"],
        num_layers=ckpt["num_layers"],
    )
    # reconstructs exact the same architecture
    # no hard-coded parameters
    # ensures reproducibility

    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    # loads learned weights
    # moves model to CPU or GPU
    # sets inference mode

    return model

@torch.no_grad()
def main():
    # disables gradient tracking
    # reduces memory usage
    # faster inference
    # correct for screening (no learning)

    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--shard_path", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="./screen_outputs")
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    device = get_device(args.device)    # device setup
    ensure_dir(args.out_dir)

    payload = torch.load(args.shard_path)   # loading the shard
    shard_id = payload["shard_id"]
    data_list = payload["data_list"]

    model = load_model(args.ckpt, device)
    loader = DataLoader(data_list, batch_size=args.batch_size, shuffle=False)
    # loads trained model
    # creates batched inference loader
    # no shuffling (order doesn't matter)

    out_csv = os.path.join(args.out_dir, f"preds_shard_{shard_id:04d}.csv")
    # this ensures one output file per shard
    # deterministic naming
    # easy aggregation later

    # we output (local_index_in_shard, score); if you have SMILES add it similarly
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["shard_id", "idx_in_shard", "score"])

        idx_base = 0
        for bacth in tqdm(loader, desc=f"Screen shard {shard_id}"):
            batch = batch.to(device)
            scores = model(batch).detach().cpu().numpy()

            # batch.num_graphs molecules in this batch
            for i, s in enumerate(scores):
                w.writerow([shard_id, idx_base + i, float(s)])
            
            idx_base += batch.num_graphs

    print(f"Wrote: {out_csv}")

if __name__ == "__main__":
    main()