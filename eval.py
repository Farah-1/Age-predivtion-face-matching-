from utils import *
from data import *
from transforms import *
from config import *
from model import *
@torch.no_grad()
def compute_metrics(
    model: nn.Module,
    loader,
    device: torch.device,
    cs_thresholds: List[int] = None,
) -> Tuple[float, dict, list]:
    if cs_thresholds is None:
        cs_thresholds = list(range(1, 11))  # 1..10 years

    model.eval()

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

    mae = np.mean(np.abs(preds - np.floor(targets)))

    cs_scores = {}
    abs_err = np.abs(preds - np.floor(targets))
    n = len(abs_err)
    for t in cs_thresholds:
        cs_scores[f"CS_{t}"] = float((abs_err <= t).sum() / n)

    return float(mae), cs_scores, list(zip(preds, np.floor(targets)))
def visualize_prediction(
    model: nn.Module,
    dataset,
    idx: int,
    device: torch.device,
    img_size: int,
):
    model.eval()
    sample = dataset[idx]
    image = sample["image"].unsqueeze(0).to(device)
    true_age = sample["age"].item()

    with torch.no_grad():
        pred_age, _ = model(image)
        pred_age = pred_age.item()

# Convert tensor (C,H,W) to numpy
    img_np = sample["image"].cpu().numpy().transpose(1, 2, 0)


    # Ensure the final values are strictly between 0 and 1
    img_np = np.clip(img_np, 0, 1)

        

    plt.figure(figsize=(6, 6))
    plt.imshow(img_np)
    plt.axis("off")
    plt.title(f"True Age: {true_age:.0f} | Predicted Age: {pred_age:.1f} | Error: {abs(true_age - pred_age):.1f} years",
              fontsize=12)
    plt.tight_layout()
    plt.show()

def main():
    cfg = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    test_ds = UTKFaceFolderDataset(os.path.join(cfg.split_path, 'test'), transform=get_val_transforms(cfg.img_size))

    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
# Load best model
    ckpt_path = os.path.join(cfg.output_dir, "best_model_2nd_trial.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_cfg_dict = ckpt.get("config", {})
    use_soft_labels = True  # default; can be extended to persist in config

    model = AgePredictionModel(
        use_soft_labels=use_soft_labels,
        max_age=ckpt_cfg_dict.get("max_age", cfg.max_age),
        dropout=ckpt_cfg_dict.get("dropout", cfg.dropout),
        pretrained=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded best model from epoch {ckpt['epoch']} with Val MAE {ckpt['best_val_mae']:.4f}")

    # Evaluate on test set
    mae, cs_scores, pred_target_pairs = compute_metrics(model, test_loader, device)

    print(f"\n=== Test Set Results ===")
    print(f"MAE: {mae:.4f} years")
    print(f"\nCumulative Scores:")
    for k, v in cs_scores.items():
        print(f"  {k}: {v * 100:.2f}%")

    # Visualize some predictions
    print("Visualizing sample predictions...")
    for idx in [0, 10, 20, 30, 40]:
        if idx < len(test_ds):
            visualize_prediction(model, test_ds, idx, device, cfg.img_size)
        

if __name__ == "__main__":
    main()

