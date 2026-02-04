# exercise3_benchmark_infer.py

"""
This exercise demonstrates how a trained AI model is used for virtual screening and why
inference throughput is the dominant cost at scale.

* The model is already trained (exercise 2)
* we score many unseen molecules
* we measure how fast this can be done

Workflow:
1. load a trained molecular GNN
2. Load a set of molecules (ESOL subset, stand-in for a library)
3. Run forward passes only (no learning)
4. Meassure:
    - elapsed time
    - number of molecules processed
5. Compute throughput
6. Compare CPU vs GPU
"""

import argparse 
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import MoleculeNet

from models import GINRegressor
from utils import set_seed, get_device, Timer 

import matplotlib.pyplot as plt

@torch.no_grad()
# disables gradient-tracking, reduces memory, inference-only mode
def run_inference(model, loader, device):
    """
    - runs the model on a dataset without gradients
    - measures wall-clock time
    - counts how many molecules were processed
    """
    model.eval()
    # warmup for GPU
    if device.type == "cuda":   # gpu
        for _ in range(3):
            for batch in loader:
                batch = batch.to(device)
                _ = model(batch)
                break
        torch.cuda.synchronize()

    n = 0
    with Timer() as t:      # only the forward pass loop is being timed
        # we do not time model and dataset loading; thus we isolate full inference cost
        for batch in loader:
            batch = batch.to(device)
            _ = model(batch)
            n += batch.num_graphs
        if device.type == "cuda":
            torch.cuda.synchronize()

    return t.elapsed, n

def load_model(ckpt_path: str, device):
    # loads the saved .pt checkpoint
    # reconstructs the exact same model
    # moves it to CPU or GPU
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = GINRegressor(
        in_dim=ckpt["in_dim"],  #architecture parameters are stored so the model can be recreated exactly without hard-coding
        hidden_dim=ckpt["hidden_dim"],
        num_layers=ckpt["num_layers"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--ckpt", type=str, default="./artifacts/gin_esol_best.pt")
    ap.add_argument("--n_mols", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)

    ds = MoleculeNet(root=args.data_root, name="ESOL")
    # Dataset loading
    # ESOL is reused only as a convenient molecular source
    # we are not evaluating accuracy
    # we are measuring speed
    n = min(args.n_mols, len(ds))
    subset = ds[:n]
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False)

    # CPU run

    cpu = get_device("cpu")
    # establlishes a baseline
    # demonstrates that CPU inference is possible
    # shows it does not scale

    model_cpu = load_model(args.ckpt, cpu)
    cpu_time, cpu_n = run_inference(model_cpu, loader, cpu)

    # GPU run
    # automatically enables GPU on HPC
    if torch.cuda.is_available():
        gpu = get_device("cuda")
        model_gpu = load_model(args.ckpt, gpu)
        gpu_time, gpu_n = run_inference(model_gpu, loader, gpu)
    else:
        gpu_time, gpu_n = None, None

    print("\n=== Inference Benchmark ===")
    print(f"Molecules: {n} | batch_size={args.batch_size}")
    print(f"CPU: {cpu_time:.3f} s  ({cpu_n/cpu_time:.1f} mol/s)")
    if gpu_time is not None:
        print(f"GPU: {gpu_time:.3f} s  ({gpu_n/gpu_time:.1f} mol/s)")
        print(f"Speedup: {cpu_time/gpu_time:.2f}x")
    else:
        print("GPU: not available")

    labels = ["CPU"]
    throughputs = [cpu_n / cpu_time]

    if gpu_time is not None:
        labels.append("GPU")
        throughputs.append(gpu_n / gpu_time)

    # This plot shows how many molecules per second can be scored during inference;
    # it ilustrates why GPUs and HPC are required for large-scale virtual screening;

    plt.figure()
    plt.bar(labels, throughputs)
    plt.ylabel("Molecules per second")
    plt.title("Inference Throughput Comparison")
    plt.tight_layout()
    plt.savefig("inference_throughput.png")
    plt.close()

if __name__ == "__main__":
    main()


