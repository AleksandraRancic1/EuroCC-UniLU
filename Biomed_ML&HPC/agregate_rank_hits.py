# agregate_rank_hits

"""
This script performs the final step of large scale AI based virtual screening:
Aggregating predictions produced in parallel on HPC and ranking molecules to produce a final hit list.

It is used after screening a large molecular library in parallel using Slurm job arrays;
this script is a closing step of Exercise 4.

Screening should be distributed across many GPUs;
Each GPU processes a subset (shard) of molecules
Results are written independently
No single job has a global view

Aggregation and ranking are mandatory to turn raw predictions into actionable decisions.


After running AI screening in parallel on GPUs, we aggregate all predictions and rank molecules
to obtain a final hit list for downstream analysis.
"""

import argparse
import glob
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default="./screen_outputs")
    ap.add_argument("--out_csv", type=str, default="./hits_ranked.csv")
    ap.add_argument("--top_k", type=int, default=1000)  
    # top k: never inspect milions of molecules
    # candidates are shorlisted for:
    # - docking
    # - synthesis
    # - expert review

    args = ap.parse_args()

    files = sorted(glob.glob(f"{args.in_dir}/preds_shard_*.csv"))
    # finds all shard outputs automatically
    # no-hard coded shard count
    # works for 10 but also for 1000 shards
    # HPC robust design
    if not files:
        raise RuntimeError(f"No prediction files found in {args.in_dir}")   # prevents silent failures
    
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    # Aggregation:
    # - reads each shard file
    # - merges them into one table
    # - each row = one molecule
    # this recreates the global screening result

    # for ESOL model: "score" is a regression output: treat it as a ranking score.
    df = df.sort_values("score", ascending=False).head(args.top_k)
    # higher score = more promising (according to the model)
    # exact meaning of score is irrelevant
    # only relative ordering matters

    # this is ligand based virtual screening logic

    df.to_csv(args.out_csv, index=False) # output = deliverable of screening

    print(f"Aggregated {len(files)} shards -> {args.out_csv}")
    print(df.head(10).to_string(index=False))
    # logging allows:
    # - quick inspection
    # - sanity check
    # - live demo output

if __name__ == "__main__":
    main()
