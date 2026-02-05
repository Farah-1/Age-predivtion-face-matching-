import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import AgePredictionModel
from src.data import UTKFaceFolderDataset
from src.transforms import get_val_transforms
from src.config import TrainConfig

def create_plots():
    cfg = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    test_dir = os.path.join("Dataset", "test")
    if not os.path.exists(test_dir):
        print(f"Test directory not found: {test_dir}")
        return
    ds = UTKFaceFolderDataset(test_dir, transform=get_val_transforms(cfg.img_size))
    subset_indices = list(range(min(len(ds), 200))) 
    ds_subset = torch.utils.data.Subset(ds, subset_indices)
    loader = DataLoader(ds_subset, batch_size=32, shuffle=False, num_workers=0)
    ckpt_primary = os.path.join(cfg.output_dir, "best_model_2nd_trial.pt")
    ckpt_fallback = os.path.join(cfg.output_dir, "best_model.pt")
    ckpt_path = ckpt_primary if os.path.exists(ckpt_primary) else ckpt_fallback
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return
    ckpt = torch.load(ckpt_path, map_location=device)
    model = AgePredictionModel(use_soft_labels=True, max_age=100, dropout=0.3, pretrained=False).to(device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.eval()
    all_preds = []
    all_targets = []
    print("Running inference for plots...")
    with torch.no_grad():
        for batch in tqdm(loader):
            images = batch["image"].to(device)
            ages = batch["age"].to(device)
            preds, _ = model(images)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(ages.cpu().numpy())
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    mae = np.mean(np.abs(all_preds - all_targets))
    plt.figure(figsize=(8, 8))
    plt.scatter(all_targets, all_preds, alpha=0.5, color='blue', label='Predictions')
    plt.plot([0, 100], [0, 100], 'r--', label='Ideal (x=y)')
    plt.xlabel('True Age')
    plt.ylabel('Predicted Age')
    plt.title(f'True vs Predicted Age (Test Set)\nMAE: {mae:.2f} years')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, 'scatter_plot.png'))
    plt.close()
    print("Saved scatter_plot.png")
    samples_indices = np.random.choice(len(ds), 6, replace=False)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    for i, idx in enumerate(samples_indices):
        sample = ds[idx]
        img_tensor = sample["image"].unsqueeze(0).to(device)
        true_age = sample["age"].item()
        with torch.no_grad():
            pred_age, _ = model(img_tensor)
            pred_age = pred_age.item()
        img_np = sample["image"].permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        axes[i].imshow(img_np)
        axes[i].set_title(f"True: {int(true_age)} | Pred: {pred_age:.1f}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'sample_predictions.png'))
    plt.close()
    print("Saved sample_predictions.png")

if __name__ == "__main__":
    create_plots()
