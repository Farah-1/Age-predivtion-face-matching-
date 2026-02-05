import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from typing import List

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import TrainConfig
from src.transforms import get_val_transforms
from src.data import UTKFaceFolderDataset
from src.model import AgePredictionModel

def _load_model(cfg: TrainConfig, device: torch.device) -> AgePredictionModel:
    primary = os.path.join(cfg.output_dir, "best_model_2nd_trial.pt")
    fallback = os.path.join(cfg.output_dir, "best_model.pt")
    ckpt_path = primary if os.path.exists(primary) else fallback
    model = AgePredictionModel(use_soft_labels=True, max_age=cfg.max_age, dropout=cfg.dropout, pretrained=False).to(device)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state)
    else:
        print("Warning: checkpoint not found; using randomly initialized model.")
    model.eval()
    return model

@torch.no_grad()
def compute_metrics(model: AgePredictionModel, loader, device: torch.device, cs_thresholds: List[int] = None):
    if cs_thresholds is None:
        cs_thresholds = list(range(1, 11))
    all_preds = []
    all_targets = []
    for batch in loader:
        images = batch["image"].to(device)
        ages = batch["age"].to(device)
        age_pred, _ = model(images)
        all_preds.append(age_pred.cpu())
        all_targets.append(ages.cpu())
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    mae = float(np.mean(np.abs(preds - np.floor(targets))))
    cs_scores = {}
    abs_err = np.abs(preds - np.floor(targets))
    n = len(abs_err)
    for t in cs_thresholds:
        cs_scores[f"CS_{t}"] = float((abs_err <= t).sum() / n)
    return mae, cs_scores, int(n)

def create_charts():
    cfg = TrainConfig()
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dir = os.path.join(cfg.split_path, "test")
    if not os.path.exists(test_dir):
        print(f"Test split not found at {test_dir}.")
        return
    test_ds = UTKFaceFolderDataset(test_dir, transform=get_val_transforms(cfg.img_size))
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    model = _load_model(cfg, device)
    mae, cs_scores, n = compute_metrics(model, test_loader, device)
    metrics_path = os.path.join(plots_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"mae": mae, "cs_scores": cs_scores, "samples": n}, f, indent=2)
    print(f"Saved metrics to {metrics_path}")
    labels = list(cs_scores.keys())
    values = [cs_scores[k] * 100 for k in labels]
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, color="#3b82f6")
    plt.ylim(0, 100)
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Cumulative Score Threshold")
    plt.title(f"CS Accuracy (Test) | MAE: {mae:.2f} years | N={n}")
    plt.grid(axis="y", alpha=0.3)
    save_path = os.path.join(plots_dir, "cs_bar.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved CS bar chart to {save_path}")

if __name__ == "__main__":
    create_charts()
