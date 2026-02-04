
from utils import * 

def get_train_transforms(img_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(img_size, img_size), # Keeps your 224 resizing
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=0,
                p=0.5,
            ),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            # CoarseDropout for partial face occlusion simulation
            A.CoarseDropout(
                max_holes=1,
                max_height=int(0.2 * img_size),
                max_width=int(0.2 * img_size),
                p=0.5,
            ),
            # NEW: Replaced ImageNet normalization with simple 0-1 scaling
            A.ToFloat(max_value=255.0), 
            ToTensorV2(),
        ]
    )


def get_val_transforms(img_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            # NEW: Replaced ImageNet normalization with simple 0-1 scaling
            A.ToFloat(max_value=255.0), 
            ToTensorV2(),
        ]
    )


def get_test_transforms(img_size: int) -> A.Compose:
    return get_val_transforms(img_size)

