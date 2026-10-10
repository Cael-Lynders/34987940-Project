# MMA3001 Project: Automated Severity Scoring for Packaged Pork Rasher Defects

Author: Cael Lynders (Student ID 34987940)

## Problem statement

Manual visual inspection of packaged meat is slow, inconsistent between inspectors, and does not scale on a production line. This project addresses a narrow part of that problem: **predicting a continuous defect-severity score for an image of a packaged pork rasher tray, based on the types and number of defects detected in that image.**

Rather than a binary pass/fail decision, the model outputs a graded severity score that could feed a downstream pass/fail or review-queue decision in an automated quality report.

- **Input:** per-image defect counts (total detections plus counts for `loose-meat`, `packaging-error`, `twisted-meat`, `unsealed`, `wrinkle`), derived from the bounding-box annotations of the Roboflow dataset *pork-rasher-error-packaging* (workspace `hello-2aqe0`, CC BY 4.0, Version 4).
- **Output:** `severity_score`, the mean bounding-box area (pixel²) of all annotated defects in an image (0 if none).
- **Methods compared:** mean-prediction baseline, decision tree regression, SVR (RBF), and neural network regression, evaluated with MAE, RMSE and R², 5-fold cross-validation, runtime and memory profiling, and hyperparameter tuning (`GridSearchCV`).
- **Validation:** the dataset's predefined train/valid/test splits, a mean baseline, residual diagnostics and failure-mode analysis. Validation is against the dataset's own annotations, not independently verified inspection outcomes.

## Repository structure

```
notebooks/   Main Colab notebook (34987940_Project.ipynb)
src/         Reusable code (features.py: build_feature_table)
tests/       pytest tests and saved test report
docs/        Generated HTML documentation and diagnostic plots
data/        Placeholder only; the dataset is downloaded at runtime and not committed
```

## Setup

The project is designed to run in Google Colab.

1. **Open the notebook** `notebooks/34987940_Project.ipynb` in Google Colab.
2. **Add Colab Secrets** (key icon in the left sidebar, with notebook access enabled):
   - `GH_PAT`: a GitHub personal access token (only needed for cloning or pushing to a private repo)
   - `ROBO_API`: your Roboflow API key (Roboflow Settings > API Keys)
3. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```
   Core packages: `roboflow`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `pytest`, `pdoc`.
4. **Run the notebook top to bottom** (Runtime > Run all). It will:
   - connect to Roboflow and download Version 4 of the dataset in COCO format (into the Colab runtime, not the repo)
   - build per-image feature tables for the train/valid/test splits
   - train and compare the candidate models, run cross-validation and tuning, and produce residual and failure-mode diagnostics
5. **Set your git identity** if committing from a fresh Colab session:
   ```
   git config --global user.email "your-email@example.com"
   git config --global user.name "Your Name"
   ```

## Running the tests

```
python -m pytest tests/ -v
```

`python -m pytest` is used so the project root is on the import path and `from src.features import ...` resolves.

## Generating the documentation

```
pip install pdoc
pdoc --html src -o docs
```

## Data and licence

The dataset is not included in this repository. It is downloaded at runtime from Roboflow Universe under CC BY 4.0. Code in this repository is released under the licence in `LICENSE`.
