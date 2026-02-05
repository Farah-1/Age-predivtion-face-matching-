from utils import *
def prepare_split_folders(cfg):
    base_data_path = cfg.split_path
    subfolders = ['train', 'val', 'test']
    if os.path.exists(base_data_path):
        shutil.rmtree(base_data_path)
    for sub in subfolders:
        os.makedirs(os.path.join(base_data_path, sub), exist_ok=True)
    all_files = glob.glob(os.path.join(cfg.data_dir, "*.jpg"))
    train_f, temp_f = train_test_split(all_files, test_size=0.2, random_state=cfg.seed)
    val_f, test_f = train_test_split(temp_f, test_size=0.5, random_state=cfg.seed)
    for files, folder in zip([train_f, val_f, test_f], subfolders):
        for f in files:
            dest = os.path.join(base_data_path, folder, os.path.basename(f))
            shutil.copy(f, dest)
    print(f"Data split complete: Train({len(train_f)}), Val({len(val_f)}), Test({len(test_f)})")
    return base_data_path

class UTKFaceFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None, max_age=100):
        self.files = glob.glob(os.path.join(folder_path, "*.jpg"))
        self.transform = transform
        self.max_age = max_age
    def _create_age_distribution(self, age):
        ages = np.arange(0, self.max_age + 1)
        dist = np.exp(-0.5 * ((ages - age) / 2.0) ** 2)
        return (dist / dist.sum()).astype(np.float32)
    def __len__(self): 
        return len(self.files)
    def __getitem__(self, idx):
        path = self.files[idx]
        try:
            age = int(os.path.basename(path).split('_')[0])
        except:
            age = 25
        age = min(age, self.max_age)
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(image=np.array(img))["image"]
        return {
            "image": img,
            "age": torch.tensor(age, dtype=torch.float32),
            "age_dist": torch.from_numpy(self._create_age_distribution(age))
        }
