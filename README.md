# Age Prediction and Face Matching Project

This project implements an Age Prediction model using Deep Learning (EfficientNet-B4 backbone) and provides a Face Matching module that combines age prediction with face verification.

## Features

- **Age Prediction**: Predicts age from facial images using a regression model with Soft Label distribution learning.
- **Face Matching**: Verifies if two images belong to the same person using ArcFace (DeepFace) and predicts the age of both faces.
- **Robustness**: Designed to handle large age gaps (2-40 years) between images.
- **Dataset Support**: Compatible with UTKFace and FG-NET dataset formats.

## Project Structure

- Entry points:
  - `train.py`: Train the age prediction model.
  - `eval.py`: Evaluate the trained model on the test set.
- Core modules (imported by entry points):
  - `src/` with `config.py`, `data.py`, `model.py`, `losses.py`, `transforms.py`, `utils.py`, `eval.py`
- Auxiliary scripts:
  - `auxiliary/` with plotting, galleries, and report generation scripts
- Outputs:
  - `outputs/` stores checkpoints and generated charts/images

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

Use the auxiliary face matching module to compare two images and predict ages.

**How to run:**
1. Open `auxiliary/face_matching.py`.
2. In the `if __name__ == "__main__":` block, set your image paths.
3. Run:
   ```bash
   python auxiliary\face_matching.py
   ```

### 2. Training the Model

To train the age prediction model from scratch:

1.  Ensure your dataset is prepared (UTKFace format).
2.  Adjust parameters in `src/config.py` if needed (batch size, epochs, learning rate, paths).
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
- Loads `outputs/best_model_2nd_trial.pt` (update path in `eval.py` if needed).
- Prints MAE and CS metrics; shows sample prediction visualizations.
- If you face Windows worker issues, `eval.py` uses `num_workers=0` for stability.

## Dataset Format

The system supports parsing ages from filenames:
- **UTKFace**: `[age]_[gender]_[race]_[date].jpg` (e.g., `20_1_0_20170116174525125.jpg`)
- **FG-NET**: `[ID]A[AGE][suffix].JPG` (e.g., `001A02.JPG`)

## Model Weights & Large Files

The trained model weights (`best_model_2nd_trial.pt`) are approximately 223MB, which exceeds GitHub's standard 100MB file limit. To upload this repository to GitHub, you **must use Git LFS (Large File Storage)**.

### How to Upload to GitHub

1.  **Install Git LFS**:
    - Download and install from [git-lfs.com](https://git-lfs.com/).
    - Or run: `git lfs install`

2.  **Initialize the Repository**:
    ```bash
    git init
    git lfs install
    ```

3.  **Track Large Files** (Already configured in `.gitattributes`):
    The repository includes a `.gitattributes` file that automatically tracks `*.pt` files using LFS.
    ```bash
    git add .gitattributes
    ```

4.  **Commit and Push**:
    ```bash
    git add .
    git commit -m "Initial commit with model weights"
    git branch -M main
    git remote add origin <YOUR_GITHUB_REPO_URL>
    git push -u origin main
    ```

## Dependencies

Key libraries used:
- `torch`, `torchvision` (Deep Learning)
- `deepface` (Face Verification)
- `tf-keras` (Required by DeepFace)
- `timm` (EfficientNet backbone)
- `numpy`, `pandas`, `matplotlib` (Data processing & Viz)
