import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import TrainConfig
from src.data import UTKFaceFolderDataset
from src.transforms import get_val_transforms
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
def create_gallery():
    cfg = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    test_dir = os.path.join(cfg.split_path, "test")
    ds = UTKFaceFolderDataset(test_dir, transform=get_val_transforms(cfg.img_size))
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    model = _load_model(cfg, device)
    for batch in loader:
        images_batch = batch["image"].to(device)
        ages_batch = batch["age"].cpu().numpy()
        break
    preds, _ = model(images_batch)
    preds = preds.cpu().numpy()
    k = min(16, images_batch.size(0))
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i in range(k):
        img = batch["image"][i].cpu().numpy().transpose(1, 2, 0)
        img = np.clip(img, 0, 1)
        ax = axes[i // 4, i % 4]
        ax.imshow(img)
        ax.set_title(f"T:{ages_batch[i]:.0f} | P:{preds[i]:.1f}", fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    save_path = os.path.join(plots_dir, "sample_predictions_batch2.png")
    plt.savefig(save_path)
    print(f"Saved age prediction gallery to {save_path}")

if __name__ == "__main__":
    create_gallery()
