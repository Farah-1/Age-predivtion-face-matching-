from utils import * 
from data import *
from transforms import *
from losses import *
from config import *
from model import *
def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    scheduler,
    criterion: AgeLoss,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    running_l1 = 0.0

    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1} [Train]", leave=False)
    for step, batch in pbar:
        images = batch["image"].to(device)
        ages = batch["age"].to(device)
        age_dists = batch["age_dist"].to(device)

        optimizer.zero_grad()
        age_pred, logits = model(images)

        loss, loss_dict = criterion(age_pred, logits, ages, age_dists)
        loss.backward()
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * images.size(0)
        running_l1 += loss_dict["l1"].item() * images.size(0)

        # Update progress bar text with current batch loss
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "l1": f"{loss_dict['l1'].item():.2f}"})

        if step % 50 == 0:
            global_step = epoch * len(loader) + step

    epoch_loss = running_loss / len(loader.dataset)
    epoch_l1 = running_l1 / len(loader.dataset)
    return {"loss": epoch_loss, "l1_mae": epoch_l1}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: AgeLoss,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    running_l1 = 0.0

    # Wrap the loader with tqdm for validation progress
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Val]", leave=False)

    for batch in pbar:
        images = batch["image"].to(device)
        ages = batch["age"].to(device)
        age_dists = batch["age_dist"].to(device)

        age_pred, logits = model(images)
        loss, loss_dict = criterion(age_pred, logits, ages, age_dists)

        running_loss += loss.item() * images.size(0)
        running_l1 += loss_dict["l1"].item() * images.size(0)
        
        # Update progress bar
        pbar.set_postfix({"val_l1": f"{loss_dict['l1'].item():.2f}"})

    epoch_loss = running_loss / len(loader.dataset)
    epoch_l1 = running_l1 / len(loader.dataset)
    return {"loss": epoch_loss, "l1_mae": epoch_l1}
def main():
    cfg = TrainConfig()
    ensure_dirs(cfg)
    cfg.split_path = prepare_split_folders(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

 
    train_ds = UTKFaceFolderDataset(os.path.join(cfg.split_path, 'train'), transform=get_train_transforms(cfg.img_size))
    val_ds = UTKFaceFolderDataset(os.path.join(cfg.split_path, 'val'), transform=get_val_transforms(cfg.img_size))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    use_soft_labels = True 
# Model
    model = AgePredictionModel(
        use_soft_labels=use_soft_labels,
        max_age=cfg.max_age,
        dropout=cfg.dropout,
        pretrained=True,
    ).to(device)

    # Loss
    criterion = AgeLoss(
        max_age=cfg.max_age,
        lambda_l1=cfg.lambda_l1,
        lambda_meanvar=cfg.lambda_meanvar,
        lambda_kl=cfg.lambda_kl,
    ).to(device)

    # Optimizer
    optimizer = AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    # Scheduler
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg.lr,
        steps_per_epoch=steps_per_epoch,
        epochs=cfg.epochs,
        pct_start=0.1,
        div_factor=25.0,
        final_div_factor=1e4,
    )

    best_val_mae = float("inf")
    patience_counter = 0
    ckpt_path = os.path.join(cfg.output_dir, "best_model.pt")

    for epoch in range(cfg.epochs):
        # Use tqdm.write to keep the batch progress bars from jumping around
        tqdm.write(f"\nEpoch {epoch + 1}/{cfg.epochs}")

        # 1. Train 
        # (The batch progress bar will appear here because of the tqdm we added to the function)
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            device,
            epoch,
        
        )
        tqdm.write(
            f"Train - Loss: {train_metrics['loss']:.4f}, "
            f"MAE: {train_metrics['l1_mae']:.4f}"
        )

        # 2. Validate
        val_metrics = validate(
            model, val_loader, criterion, device, epoch
        )
        tqdm.write(
            f"Val    - Loss: {val_metrics['loss']:.4f}, "
            f"MAE: {val_metrics['l1_mae']:.4f}"
        )

        # 3. Checkpointing logic (Keep exactly as you had it)
        val_mae = val_metrics["l1_mae"]
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val_mae": best_val_mae,
                    "config": cfg.__dict__,
                },
                ckpt_path,
            )
            tqdm.write(f" Saved new best model with MAE {best_val_mae:.4f}")
        else:
            patience_counter += 1
            tqdm.write(
                f"No improvement in MAE for {patience_counter} epochs "
                f"(best: {best_val_mae:.4f})"
            )

        # Early stopping
        if patience_counter >= cfg.early_stopping_patience:
            tqdm.write("Early stopping triggered.")
            break

    print(f"\nTraining completed! Best Val MAE: {best_val_mae:.4f}")
    print(f"Best model saved to: {ckpt_path}")


if __name__ == "__main__":
    main()

