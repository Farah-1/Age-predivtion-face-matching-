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
  ## the model weights and the virtual environment are published on google drive
  - https://drive.google.com/drive/folders/1b-pfmko95O8QRi7goUBSVfE0eHI82XhT?usp=sharing

## Installation and Setup


### Setting up the Environment

The project is designed to run in a specific virtual environment (`.venv_age_pred`).

1.  **Activate the Virtual Environment**:
    Open your terminal (PowerShell) and run:
    ```powershell
    .\.venv_age_pred\Scripts\Activate.ps1
    ```


## Usage

### 1. Face Matching & Age Prediction (Demo)

Use the auxiliary face matching module to compare two images and predict ages.

**How to run:**
1. Open `src/create_facematching_plot.py`.
2. set your image paths.
3. Run:
   ```bash
   python src\create_facematching_plot.py
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

### 3. Evaluation

To evaluate the best trained model on the test dataset:

```bash
python src/eval.py
```
- Loads `outputs/best_model_2nd_trial.pt` (update path in `eval.py` if needed).
- Prints MAE and CS metrics; shows sample prediction visualizations.
- If you face Windows worker issues, `eval.py` uses `num_workers=0` for stability.

## Dataset Format

The system supports parsing ages from filenames:
- **UTKFace**: `[age]_[gender]_[race]_[date].jpg` (e.g., `20_1_0_20170116174525125.jpg`)
- **FG-NET**: `[ID]A[AGE][suffix].JPG` (e.g., `001A02.JPG`)

## Model Weights & Large Files

The trained model weights (`best_model_2nd_trial.pt`) are approximately 223MB, which exceeds GitHub's standard 100MB file limit.
so i uploaded to google drive   - https://drive.google.com/drive/folders/1b-pfmko95O8QRi7goUBSVfE0eHI82XhT?usp=sharing

