"""Train the weak CNN on CIFAR-10 and log to MLflow.

Hardcoded hyperparameters on purpose — autoresearch edits the source, not CLI flags.

Known weak choices (leave them; autoresearch will improve):
- SGD without momentum
- lr 0.01 fixed (no scheduler)
- batch_size 16
- epochs 5
"""

import os
import subprocess
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim

from baseline.data import get_loaders
from baseline.model import WeakCNN


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _git_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True)
        return out.strip()
    except Exception:
        return "unknown"


def main() -> None:
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("autoresearch-class")

    device = _device()
    model = WeakCNN().to(device)
    train_loader, val_loader = get_loaders(batch_size=16, download=True)

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    epochs = 5

    with mlflow.start_run() as run:
        mlflow.log_params({
            "model": "WeakCNN",
            "optimizer": "SGD",
            "lr": 0.01,
            "batch_size": 16,
            "epochs": epochs,
        })
        mlflow.set_tag("commit_sha", _git_sha())
        mlflow.set_tag("experiment_iter", os.environ.get("AUTORESEARCH_ITER", "0"))

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * xb.size(0)
            train_loss /= len(train_loader.dataset)

            model.eval()
            correct, total, val_loss = 0, 0, 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    val_loss += criterion(logits, yb).item() * xb.size(0)
                    correct += (logits.argmax(1) == yb).sum().item()
                    total += xb.size(0)
            val_loss /= total
            val_acc = correct / total

            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc},
                step=epoch,
            )
            print(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        ckpt_path = Path("data/last_model.pt")
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        mlflow.log_artifact(str(ckpt_path))
        mlflow.log_artifact("baseline/model.py")

        Path(".autoresearch").mkdir(exist_ok=True)
        Path(".autoresearch/last_val_acc.txt").write_text(f"{val_acc:.6f}\n")
        print(f"[FINAL] val_acc={val_acc:.4f}  run_id={run.info.run_id}")


if __name__ == "__main__":
    main()
