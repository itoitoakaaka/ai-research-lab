"""Reproducible synthetic regression benchmarks for Lab A."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

CONDITIONS = {
    "baseline": dict(n_train=60, noise=0.2, depth=2, width=32, lr=0.01),
    "n20": dict(n_train=20, noise=0.2, depth=2, width=32, lr=0.01),
    "n300": dict(n_train=300, noise=0.2, depth=2, width=32, lr=0.01),
    "noise06": dict(n_train=60, noise=0.6, depth=2, width=32, lr=0.01),
    "depth4": dict(n_train=60, noise=0.2, depth=4, width=32, lr=0.01),
    "lr00001": dict(n_train=60, noise=0.2, depth=2, width=32, lr=0.0001),
    "lr005": dict(n_train=60, noise=0.2, depth=2, width=32, lr=0.05),
    "width128": dict(n_train=60, noise=0.2, depth=2, width=128, lr=0.01),
}

def target(x):
    return np.sin(3 * x) + 0.3 * x

def net(depth, width):
    layers = []
    size = 1
    for _ in range(depth):
        layers += [nn.Linear(size, width), nn.Tanh()]
        size = width
    layers += [nn.Linear(size, 1)]
    return nn.Sequential(*layers)

def run(name, cfg, seed, epochs):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    xt = rng.uniform(-2, 2, size=(cfg["n_train"], 1)).astype("float32")
    yt = (target(xt) + rng.normal(0, cfg["noise"], xt.shape)).astype("float32")
    xv = rng.uniform(-2, 2, size=(200, 1)).astype("float32")
    yv = (target(xv) + rng.normal(0, cfg["noise"], xv.shape)).astype("float32")
    grid = np.linspace(-2, 2, 400, dtype="float32").reshape(-1, 1)
    model = net(cfg["depth"], cfg["width"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    tx, ty = torch.from_numpy(xt), torch.from_numpy(yt)
    vx, vy = torch.from_numpy(xv), torch.from_numpy(yv)
    history = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(tx), ty)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            tr = nn.functional.mse_loss(model(tx), ty).item()
            va = nn.functional.mse_loss(model(vx), vy).item()
        history.append(dict(condition=name, seed=seed, epoch=epoch+1,
                            train_mse=tr, val_mse=va))
    with torch.no_grad():
        pred = model(torch.from_numpy(grid)).numpy()
    truth_mse = float(np.mean((pred - target(grid)) ** 2))
    metrics = dict(condition=name, seed=seed, train_mse=tr,
                   val_mse=va, truth_mse=truth_mse, **cfg)
    predictions = [dict(condition=name, seed=seed, x=float(x),
                        prediction=float(p), truth=float(y))
                   for x, p, y in zip(grid[:, 0], pred[:, 0], target(grid)[:, 0])]
    return metrics, history, predictions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    if args.epochs < 1 or args.seeds < 1:
        parser.error("epochs and seeds must be positive")
    torch.set_num_threads(1)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    metrics, history, predictions = [], [], []
    for name, cfg in CONDITIONS.items():
        for seed in range(args.seeds):
            m, h, p = run(name, cfg, seed, args.epochs)
            metrics.append(m); history.extend(h); predictions.extend(p)
            print(name, seed, "validation MSE:", round(m["val_mse"], 5))
    pd.DataFrame(metrics).to_csv(out / "run_metrics.csv", index=False)
    pd.DataFrame(history).to_csv(out / "learning_curves.csv", index=False)
    pd.DataFrame(predictions).to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(metrics).groupby("condition")[["train_mse", "val_mse", "truth_mse"]].agg(["mean", "std"]).to_csv(out / "condition_summary.csv")
    print("Saved CSV files to", out)

if __name__ == "__main__":
    main()
