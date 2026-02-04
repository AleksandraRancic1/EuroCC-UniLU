# train bbbp classifier

import argparse
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import MoleculeNet
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from models import GINRegressor

from utils import set_seed, get_device, ensure_dir

def split_indices(n, seed=42):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    return train_idx, val_idx, test_idx

@torch.no_grad()
def evaluate_auc(model, loader, device):
    model.eval()
    ys, yhats = [], []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        probs = torch.sigmoid(logits).cpu()
        y = batch.y.view(-1).cpu()

        mask = ~torch.isnan(y)
        ys.append(y[mask])
        yhats.append(probs[mask])

    y = torch.cat(ys).numpy()
    yhat = torch.cat(yhats).numpy()

    return roc_auc_score(y,yhat)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--out_dir", type=str, default="./artifacts_bbbp")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)

    ensure_dir(args.out_dir)

    # Load BBBP dataset
    dataset = MoleculeNet(root=args.data_root, name="BBBP")
    train_idx, val_idx, test_idx = split_indices(len(dataset), args.seed)

    train_loader = DataLoader(dataset[train_idx], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset[val_idx], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(dataset[test_idx], batch_size=args.batch_size, shuffle=False)

    model = GINRegressor(       
        in_dim=dataset.num_node_features,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    best_path = f"{args.out_dir}/gin_bbbp_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            batch = batch.to(device)
            y = batch.y.view(-1)

            mask = ~torch.isnan(y)
            if mask.sum() == 0:
                continue

            logits = model(batch)[mask]
            loss = criterion(logits, y[mask])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        val_auc = evaluate_auc(model, val_loader, device)
        print(f"[epoch {epoch}] train_loss={sum(losses)/len(losses):.4f} val_AUC={val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "in_dim": dataset.num_node_features,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                },
                best_path
            )
            print(f"  saved best model → {best_path}")

    # Test evaluation
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    test_auc = evaluate_auc(model, test_loader, device)
    print(f"\nBest validation AUC: {best_val_auc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Model saved at: {best_path}")


if __name__ == "__main__":
    main()


