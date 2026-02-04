#exercise2_train_gnn.py

"""
This exercise demonstrates how a Graph Neural Network learns a molecular property from structure,
and why GPUs/HPC are needed for drug-discovery AI task.

We train a graph neural network to predict aqueous solubility (ESOL) from molecular structure, 
then evaluate how well it generalizes to unseen molecules.

* Workflow of the exercise:
1. Load a molecular dataset (ESOL)
- molecules are given as SMILES
- internally converted to graphs (atoms = nodes, bonds = edges)
2. Split data
- 80% training
- 10% validation
- 10% test
3. Train a GNN
- input: molecular graph
- output: one real value (predicted solubility)
4. Monitor learning
- training loss
- validation RMSE
5. Save the best model
- based on validation performance
6. Evaluate on test set
- final generalization check
7. Visualize learning
- Loss/ RMSE vs epoch

The model learns structure -> property relationships
After training, the same model can score new molecules
In later exercises, the model is used for virtual screening at scale
"""

import argparse
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import MoleculeNet
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from models import GINRegressor
from utils import set_seed, get_device, ensure_dir

import matplotlib.pyplot as plt

def split_indices(n, seed=42):
    """
    - creates a deterministic random split of the dataset;
    - ensures reproducibility
    - separates data into:
        - training (learning)
        - validation (model selection)
        - test (final evaluation)
    """
    # simple deterministic split: 80/10/10
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    return train_idx, val_idx, test_idx

@torch.no_grad()
# disables gradient computation
# fast and memory efficient
# correct for evaluation

def evaluate(model, loader, device):
    """
    - runs the model in inference mode
    - computes RMSE between predictions and true values
    """
    model.eval()
    ys, yhats = [], []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch).detach().cpu()
        y = batch.y.view(-1).cpu()
        ys.append(y)
        yhats.append(pred)
    y = torch.cat(ys).numpy()
    yhat = torch.cat(yhats).numpy()
    rmse = mean_squared_error(y, yhat) ** 0.5
    # RMSE is a standard metric for regressionl it is interpretable error scale
    return rmse

def main():
    """
    The full training pipeline

    """
    ap = argparse.ArgumentParser()
    # argument parsing
    # this allows the same script to run locally on CPU, on GPU, on HPC via Slurm
    # critical for HPC portability
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--out_dir", type=str, default="./artifacts")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)   # device selection: cpu, cuda gpu -> same code, diff hardware
    ensure_dir(args.out_dir)

    # ESOL dataset (regression)
    ds = MoleculeNet(root=args.data_root, name="ESOL")
    # dataset loading
    # Downloads ESOL if missing
    # Converts SMILES into molecular graphs
    # Each molecule becomes: 1. node features (atoms) and 2. edge list (bonds)

    # Important; the model never sees SMILES just graphs

    train_idx, val_idx, test_idx = split_indices(len(ds), seed=args.seed)

    train_loader = DataLoader(ds[train_idx], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ds[val_idx], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(ds[test_idx], batch_size=args.batch_size, shuffle=False)
    # Data loaders: batch multiple molecules, enable efficient training, hide variable graph sizes

    in_dim = ds.num_node_features
    model = GINRegressor(in_dim=in_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    # This model is a graph isomorphism network
    # message passing between atoms
    # aggregation into a molecule-level vector
    # final MLP outputs one scalar

    # strong theoretical properties
    # stable and widely used in molecular ML

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_path = f"{args.out_dir}/gin_esol_best.pt"

    train_losses = []
    val_rmses = []

    for epoch in range(1, args.epochs + 1):
        """
        Inside each epoch:
        1. forward pass-predict solubility
        2. compute loss (MSE)
        3. backpropagation
        4. parameter update
        
        this is a standard supervised learning
        """

        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            batch = batch.to(device)
            pred = model(batch)
            y = batch.y.view(-1)
            loss = loss_fn(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        val_rmse = evaluate(model, val_loader, device)
        # validation is important because detects overfitting; 
        # it selects the best model
        # it mimics real model selection

        train_loss = sum(losses) / max(1, len(losses))

        print(f"[epoch {epoch}] train_loss = {train_loss:.4f} val_RMSE={val_rmse:.4f}")

        train_losses.append(train_loss)
        val_rmses.append(val_rmse)

        if val_rmse < best_val:
            """
            - keep the best generalizing model
            - not necessarily the last epoch
            """
            best_val = val_rmse
            torch.save(
                {"model_state": model.state_dict(),
                 "in_dim": in_dim,
                 "hidden_dim": args.hidden_dim,
                 "num_layers": args.num_layers},
                 best_path
            )
            print(f" saved best -> {best_path}")

    # Visualization

    # Training loss decreasing -> learning
    # Validation RMSE stability -> generalization

    plt.figure()
    plt.plot(range(1, args.epochs + 1), train_losses, label="Train loss")
    plt.plot(range(1, args.epochs + 1), val_rmses, label="Val RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.title("Training Progress (ESOL)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.out_dir}/training_curve.png")
    plt.close()
        
    # final test evaluation

    ckpt = torch.load(best_path, map_location = "cpu")
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    test_rmse = evaluate(model, test_loader, device)\
    # final, unbiased performance estimate
    # used only once
    print(f"\nBest val_RMSE={best_val:.4f} | test_RMSE = {test_rmse:.4f}")
    print(f"Model checkpoint: {best_path}")

if __name__ == "__main__":
    main()
    
