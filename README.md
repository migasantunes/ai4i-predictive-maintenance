# Predicting Machine Failures – AI4I 2020 Predictive Maintenance

This project trains a Multilayer Perceptron (MLP) and a Random Forest (RF) on the UCI AI4I 2020 dataset for two tasks:

- **Binary:** detect a failure (`Machine failure`, 3.39% positives).
- **Multiclass:** predict the failure mode among *No failure, TWF, HDF, PWF, OSF* (RNF is excluded because it is random).

It compares how well each model catches failures (missed failures vs. false alarms), using a 75/25 stratified train/test split (`random_state=42`) and repeated stratified 5-fold CV on the training set for every choice. The test set is used once, in notebook 05.

## Group Members

- Nuno Leal - No. 2026145552
- Adriel Lima - No. 2026144080
- José Antunes - No. 2023211288

The **AI-use and contribution declaration** is at the end of [notebooks/05_final_eval.ipynb](notebooks/05_final_eval.ipynb).

## Repository Structure

```text
ai4i-predictive-maintenance/
├── README.md                        # this file: setup and execution order
├── requirements.txt                 # pinned versions (Python 3.14.3)
├── data/
│   ├── raw/ai4i2020.csv             # original data (not in git, see "Data Download")
│   └── splits/                      # saved split indices (.npy): binary_train/test, multi_train/test
├── src/                             # shared code, imported by every notebook
│   ├── data.py                      # load, quality checks, labels, engineered features, preprocess(), make_preprocessor()
│   ├── splits.py                    # create, save and load the 75/25 splits
│   ├── evaluate.py                  # repeated CV, metrics (zero_division=0), threshold/cut-off tables, tuning log
│   ├── models_rf.py                 # Random Forest builder, sweeps and random-search spaces
│   └── models_mlp.py                # MLP pipeline (scaling + one-hot Type + MLPClassifier)
├── notebooks/                       # run in this order (see "Execution Order")
│   ├── 01_data_eda.ipynb            # data quality, label decisions, EDA figures, saves the splits
│   ├── 02_feature_selection.ipynb   # RF importance, mutual information, feature-set comparison
│   ├── 03_random_forest.ipynb       # RF tuning, binary + multiclass
│   ├── 04_mlp.ipynb                 # MLP tuning, binary + multiclass
│   └── 05_final_eval.ipynb          # test evaluation, error analysis, AI-use and contribution declaration
├── results/
│   ├── tuning_log.csv               # binary task: every configuration tried (validation mean and std)
│   ├── tuning_log_multi.csv         # multiclass task: every configuration tried
│   ├── feature_selection_binary.csv # feature sets vs. validation F1
│   ├── selected_features.json       # the 9 features used by the RF
│   ├── rf_binary_finalists.csv      # RF finalists re-evaluated over 4 CV seeds
│   ├── rf_multi_finalists.csv
│   ├── rf_binary_candidate.json     # RF configuration proposed by notebook 03
│   ├── rf_multi_candidate.json
│   ├── best_params.json             # the 4 frozen configurations and decision values (model-selection meeting)
│   ├── final_metrics.csv            # validation (mean, std) and test metrics, all 4 models
│   └── figures/                     # EDA plots, tuning curves, learning curves, test confusion matrices
└── report/
    └── ML_report.pdf                # results report (3 pages)
```

## Data Download

Download the dataset from [UCI](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset), unzip it and place `ai4i2020.csv` in `data/raw/`. The file is not tracked by git.

## Environment Setup

Create an environment with Python 3.14.3 (the version the project was tested on) and install the requirements:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
nbstripout --install               # optional, for development: strips notebook outputs on commit
```

## Execution Order

Open the notebooks from the `notebooks/` folder with the `.venv` kernel and run each one with **Restart → Run All**, in this order:

1. `01_data_eda.ipynb`: data quality checks, label decisions and EDA figures. It also saves the train/test split indices to `data/splits/`.
2. `02_feature_selection.ipynb`: compares feature sets and saves the selected features to `results/selected_features.json`.
3. `03_random_forest.ipynb`: tunes the RF for both tasks and saves the RF candidates, finalists and tuning figures.
4. `04_mlp.ipynb`: tunes the MLP for both tasks and saves the learning-curve figures.
5. `05_final_eval.ipynb`: trains the four frozen models on the full training set, evaluates them once on the test set and runs the error analysis. It writes `results/final_metrics.csv` and the test confusion matrices.

Notes:

- Notebooks 03 and 04 are the slow ones (tuning with repeated CV). They are independent of each other, and only 03 needs the output of 02.
- Between 04 and 05 is the **model-selection meeting**: the four configurations and their decision values (threshold or cut-off), chosen from notebooks 03 and 04 on validation only, were frozen by hand in `results/best_params.json`. Notebook 05 reads them from there and makes no choices.
- Everything is seeded (`random_state=42` for splits, folds and models), so re-running a notebook reproduces the same numbers. Notebook 01 rewrites the same split indices.
- Every configuration tried is logged in `results/tuning_log.csv` (binary) and `results/tuning_log_multi.csv` (multiclass). Re-running a notebook replaces its rows instead of duplicating them.
- To only reproduce the final results, run notebooks 01 and 05: all the other files they need are already in `results/`.

## Additional Information

- To import `src` from the notebooks, `src/` has an `__init__.py`, and each notebook starts with:

  ```python
  import sys, pathlib
  sys.path.insert(0, str(pathlib.Path.cwd().parent))
  ```

  so the notebooks must be run from the `notebooks/` folder.
- If anything in `src/` changes, restart the kernel so the notebooks pick up the change.
- **nbstripout:** `.gitattributes` strips notebook outputs on commit in any clone where `nbstripout --install` was run. The executed notebooks for submission must therefore be taken from the working folder, not from a fresh `git clone`.
