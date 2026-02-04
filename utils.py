import os
import glob
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm.auto import tqdm
import argparse
import os
from typing import Dict

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

import shutil
# print(f"PyTorch: {torch.__version__}")
# print(f"CUDA Available: {torch.cuda.is_available()}")
# if torch.cuda.is_available():
#     print(f"GPU: {torch.cuda.get_device_name(0)}")