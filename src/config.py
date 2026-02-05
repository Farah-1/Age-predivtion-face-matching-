from utils import *
@dataclass
class TrainConfig:
    data_dir: str = r"C:\Users\fa715\OneDrive\Desktop\assesment+tasks\cyshield\age\data\UTKFace\UTKFace"
    output_dir: str = "outputs"
    split_path: str = "Dataset"
    seed: int = 42
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 8
    max_age: int = 100
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-2
    dropout: float = 0.3
    lambda_l1: float = 1.0
    lambda_meanvar: float = 0.2
    lambda_kl: float = 0.5
    early_stopping_patience: int = 10
    log_dir: str = "logs"

def ensure_dirs(cfg: TrainConfig) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.split_path, exist_ok=True)
