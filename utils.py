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

import shutil
