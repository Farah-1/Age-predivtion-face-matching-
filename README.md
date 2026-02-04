# Age Prediction and Face Matching Project

This project implements an Age Prediction model using Deep Learning (EfficientNet-B4 backbone) and provides a Face Matching module that combines age prediction with face verification.

## Features

- **Age Prediction**: Predicts age from facial images using a regression model with Soft Label distribution learning.
- **Face Matching**: Verifies if two images belong to the same person using ArcFace (DeepFace) and predicts the age of both faces.
- **Robustness**: Designed to handle large age gaps (2-40 years) between images.
- **Dataset Support**: Compatible with UTKFace and FG-NET dataset formats.

## Project Structure

- `train.py`: Main script to train the age prediction model.
- `eval.py`: Script to evaluate the trained model on the test set.
- `facematching.py`: Module for face matching and age prediction on image pairs.
- `model.py`: Defines the `AgePredictionModel` architecture.
- `data.py`: Dataset loading and preprocessing (UTKFaceFolderDataset).
- `losses.py`: Custom loss functions (L1 Loss + KL Divergence for soft labels).
- `config.py`: Configuration parameters for training and evaluation.
- `transforms.py`: Image augmentations and transformations.
- `utils.py`: Utility functions for logging and seeding.
- `requirements.txt`: List of Python dependencies.

## Installation and Setup

### Prerequisites

- Python 3.10+
- Windows (as configured in this environment)

### Setting up the Environment

The project is designed to run in a specific virtual environment (`.venv_age_pred`).

1.  **Activate the Virtual Environment**:
    Open your terminal (PowerShell) and run:
    ```powershell
    .\.venv_age_pred\Scripts\Activate.ps1
    ```

2.  **Install Dependencies** (if not already installed):
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The environment should already have necessary packages like `torch`, `deepface`, `numpy`, etc.*

## Usage

### 1. Face Matching & Age Prediction (Demo)

The `facematching.py` script takes two image paths, predicts the age of the person in each image, and verifies if they match.

**How to run:**
1. Open `facematching.py`.
2. Scroll to the bottom `if __name__ == "__main__":` block.
3. Update the `path1` and `path2` variables with your image paths:
   ```python
   path1 = r"C:\path\to\your\image1.jpg"
   path2 = r"C:\path\to\your\image2.jpg"
   ```
4. Run the script:
   ```bash
   python facematching.py
   ```
*(Note: The script outputs the predicted age for both images, the true age (if available in filename), and a similarity score/verification result.)*

### 2. Training the Model

To train the age prediction model from scratch:

1.  Ensure your dataset is prepared (UTKFace format).
2.  Adjust parameters in `config.py` if needed (batch size, epochs, learning rate).
3.  Run the training script:
    ```bash
    python train.py
    ```
    - Checkpoints will be saved in the `outputs/` directory.
    - TensorBoard logs are saved in `outputs/logs/`.

### 3. Evaluation

To evaluate the best trained model on the test dataset:

```bash
python eval.py
```
- This loads the model `outputs/best_model_2nd_trial.pt` (ensure this file exists or update the path in `eval.py`).
- Outputs Mean Absolute Error (MAE) and Cumulative Score (CS) metrics.
- Visualizes sample predictions.

## Dataset Format

The system supports parsing ages from filenames:
- **UTKFace**: `[age]_[gender]_[race]_[date].jpg` (e.g., `20_1_0_20170116174525125.jpg`)
- **FG-NET**: `[ID]A[AGE][suffix].JPG` (e.g., `001A02.JPG`)

## Dependencies

Key libraries used:
- `torch`, `torchvision` (Deep Learning)
- `deepface` (Face Verification)
- `tf-keras` (Required by DeepFace)
- `efficientnet_pytorch` / `timm` (Backbone)
- `numpy`, `pandas`, `matplotlib` (Data processing & Viz)
