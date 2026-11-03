# MMA3001 Project: Automated Severity Scoring for Packaged Pork Rasher Defects

Author: Cael Lynders (Student ID 34987940)

## Problem statement

Manual visual inspection of packaged meat is slow, inconsistent between inspectors, and does not scale on a production line. This project addresses a narrow part of that problem: **predicting a continuous defect-severity score for an image of a packaged pork rasher tray, based on the types and number of defects detected in that image.**

Rather than a binary pass/fail decision, the model outputs a graded severity score that could feed a downstream pass/fail or review-queue decision in an automated quality report.

- **Input:** per-image defect counts (total detections plus counts for `loose-meat`, `packaging-error`, `twisted-meat`, `unsealed`, `wrinkle`), derived from the bounding-box annotations of the Roboflow dataset *pork-rasher-error-packaging* (workspace `hello-2aqe0`, CC BY 4.0, Version 4). Dataset page: https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging
- **Output:** `severity_score`, the mean bounding-box area (pixel²) of all annotated defects in an image (0 if none).
- **Methods compared:** mean-prediction baseline, decision tree regression, SVR (RBF), and neural network regression, evaluated with MAE, RMSE and R², group-aware 5-fold cross validation, runtime and memory profiling, and hyperparameter tuning (`GridSearchCV`).
- **Validation:** the dataset's predefined train/valid/test splits, a mean baseline, residual diagnostics and failure-mode analysis. Validation is against the dataset's own annotations, not independently verified inspection outcomes.

## Inputs and outputs

| Item | Name | Format | Units and domain | Meaning |
| --- | --- | --- | --- | --- |
| Input | `num_detections` | integer | count, 0 or more | Total annotated defects in the image |
| Input | `loose_meat_count` | integer | count, 0 or more | Defects of class `loose-meat` |
| Input | `packaging_error_count` | integer | count, 0 or more | Defects of class `packaging-error` |
| Input | `twisted_meat_count` | integer | count, 0 or more | Defects of class `twisted-meat` |
| Input | `unsealed_count` | integer | count, 0 or more | Defects of class `unsealed` |
| Input | `wrinkle_count` | integer | count, 0 or more | Defects of class `wrinkle` |
| Output | `severity_score` | float, one per image | pixels², 0 to 388,800 (a 720×540 image) | Mean bounding-box area of the image's annotated defects; 0 if there are none |

The inputs are built by `build_feature_table` in `src/features.py` from the COCO annotation file of each dataset split. The score is never used as an input.

**Invalid or missing input**

- An annotation whose `category_id` is not in the categories mapping raises a `KeyError` (covered by a test).
- An image with no annotations is kept and given a severity score of 0 (covered by a test).

## Repository structure

```
notebooks/   Main Colab notebook (34987940_Project_Final.ipynb)
src/         Reusable code (features.py: build_feature_table)
tests/       pytest tests and saved test report
docs/        Generated HTML documentation and diagnostic plots
data/        Placeholder only; the dataset is downloaded at runtime and not committed
```

## Setup

The project is designed to run in Google Colab.

1. **Open the notebook** `notebooks/34987940_Project_Final.ipynb` in Google Colab (File > Open notebook > GitHub tab, repository `Cael-Lynders/34987940-Project`).
2. **Add Colab Secrets** (key icon in the left sidebar, with notebook access enabled):
   - `ROBO_API`: your Roboflow API key (Roboflow Settings > API Keys)
   - `GH_PAT`: a GitHub personal access token (only needed for pushing to the repository; the repository is public, so cloning needs no token)
3. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```
   Core packages: `roboflow`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `pytest`, `pdoc`.
4. **Run the notebook top to bottom** (Runtime > Run all). It will:
   - connect to Roboflow and download Version 4 of the dataset in COCO format (downloaded into data/ (git-ignored, not committed))
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

## Acceptance criteria

The ground truth is the severity score computed from the dataset's own annotations, so these criteria show consistency with the labels, not agreement with real inspectors. 

| # | Criterion | Result | Met |
| --- | --- | --- | --- |
| 1 | All pytest tests pass (`python -m pytest`), including a known-answer test: boxes of 10×10 and 20×20 give a severity of 250 | 5 of 5 pass (`tests/test_report.txt`) | Yes |
| 2 | Validation MAE (120 images) beats the mean baseline by more than the fold-to-fold spread of group-aware cross-validation | 11,329 vs 24,085; spread ± 382 | Yes |
| 3 | Test MAE (80 withheld images, scored once) is below the test baseline | 10,406 vs 19,568 | Yes |
| 4 | Validation MAE beats a two-rule benchmark (0 if the image has no detections, otherwise the mean of the non-zero training scores) | 11,329 vs 20,738 | Yes |

MAE is in pixels². Known limits: the counts carry no defect size, so images with the same counts receive the same prediction, and the validation and test sets are small (120 and 80 images).

## Generating the documentation

```
pip install pdoc
pdoc --docformat numpy src -o docs
```

Open `docs/index.html` in a browser. It redirects to the generated `src` page; the `features` module page has the function documentation.

## Data and licence

The dataset is not included in this repository. It is downloaded at runtime from Roboflow Universe under CC BY 4.0. Code in this repository is released under the licence in `LICENSE`.
