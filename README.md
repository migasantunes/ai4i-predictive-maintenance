# Predicting Machine Failures – AI4I 2020 Predictive Maintenance

This project, trains a Multilayer Perceptron (MLP) and a Random Forest on the UCI AI4I 2020 dataset for two tasks: binary failure detection (`Machine failure`) and multiclass prediction of the failure mode. It compares how well each model catches failures.

## Group Members
- Nuno Leal - No. 2026145552
- Adriel Lima - No. 2026144080
- José Antunes - No. 2023211288

## Repository Structure

```
ai4i-predictive-maintenance/
├── README.md                    # setup + execution order
├── requirements.txt             # pinned versions
├── main.ipynb                   # final notebook, including AI-use and contribution declaration
├── data/
│   ├── raw/ai4i2020.csv         # original data
│   └── splits/                  # saved indices: binary_train/test, multi_train/test
├── src/
│   ├── data.py                  # load, quality checks, preprocess(), engineered features
│   ├── splits.py                # create, save and load the 75/25 splits
│   ├── evaluate.py              # metrics, repeated CV mean ± std, confusion matrices
│   ├── models_rf.py             # Random Forest builder + tuning grid
│   └── models_mlp.py            # MLP architecture + training loop 
├── notebooks/
│   ├── 01_data_eda.ipynb
│   ├── 02_feature_selection.ipynb 
│   ├── 03_random_forest.ipynb   
│   ├── 04_mlp.ipynb 
│   └── 05_final_eval.ipynb
├── results/
│   ├── tuning_log.csv           # every config tried, with mean ± std
│   ├── best_params.json         # final configs for each model
│   ├── final_metrics.csv        # validation + test, all 4 models
│   └── figures/                 # EDA plots, tuning curves, confusion matrices
└── report/
    ├── report.md                # markdown version of the report
    └── report.pdf               # max 3 pages
```

## Data Download
Download the dataset from [UCI](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset), unzip it and place `ai4i2020.csv` in the `data/raw` folder.

## Environment Setup

Create an environment with Python 3.14.3 (Tested on Python 3.14.3) and install the requirements:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Notebooks

1. Activate the virtual environment
2. Open the notebooks in your preferred IDE or Jupyter Notebook
3. Select the kernel from .venv
4. Run the notebooks by clicking `Restart` and then `Run All` in the notebook interface.

## Additional Information

- To be able to use the `src` module in the notebooks, we added `__init__.py` to the src directory, making it importable. Then, we used the following code snippet at the beginning of each notebook:

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))
```

`main.ipynb` sits at the repository root, so it uses `pathlib.Path.cwd()` instead of `pathlib.Path.cwd().parent`.

- If any changes are made to the src module, restart the kernel to ensure the changes are reflected in the notebooks.