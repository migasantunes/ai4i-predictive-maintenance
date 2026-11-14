from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "ai4i2020.csv"
MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]

# This is so that when there are multiple modes, the rarest one is chosen as the label, 
# this way no failure is dropped and we can gain a small balance in the dataset.
PRIORITY = ["TWF", "OSF", "PWF", "HDF"] 

RAW_FEATURES = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
ENGINEERED = ["Temp diff [K]", "Power [W]", "Strain [min Nm]"]
NUMERIC = RAW_FEATURES + ENGINEERED
TYPES = ["L", "M", "H"]
CLASSES = ["No failure", "TWF", "HDF", "PWF", "OSF"]  # position = integer label

def load_raw(path=RAW_PATH):
    return pd.read_csv(path)

def quality_report(df):
    """General data quality checks, one count per check."""

    return pd.Series({
        "rows": len(df),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_measurements": int(df.drop(columns=["UDI", "Product ID"]).duplicated().sum()),
        "duplicate_UDI": int(df["UDI"].duplicated().sum()),
        "type_vs_product_id_mismatch": int((df["Product ID"].str[0] != df["Type"]).sum()),
        "negative_values": int((df.select_dtypes("number") < 0).sum().sum()),
        "process_temp_below_air": int((df["Process temperature [K]"] < df["Air temperature [K]"]).sum()),
    })

def label_issues(df):
    """Rows whose labels are inconsistent with the reason."""

    n_modes = df[MODES].sum(axis=1)
    flags = pd.DataFrame({
        "failure_no_mode": (df["Machine failure"] == 1) & (n_modes == 0),
        "mode_no_failure": (df["Machine failure"] == 0) & (n_modes > 0),
        "multiple_modes": n_modes > 1,
        "rnf": df["RNF"] == 1,
    })
    bad = flags.any(axis=1)
    out = df.loc[bad, ["UDI", "Machine failure", *MODES]].copy()
    out["reason"] = flags[bad].apply(lambda r: ", ".join(r.index[r]), axis=1)
    return out

def rows_to_drop(df, task):
    """
    Parameters:
    df: pd.DataFrame 
    task: str, either "binary" or "multi"

    Binary keeps every row
    Multiclass drops rows with no learnable label.
    
    Returns:
    pd.Index of rows to drop
    """
    
    if task == "binary":
        return pd.Index([])
    if task == "multi":
        no_failure = (df["Machine failure"] == 0) & (df[MODES].sum(axis=1) == 0)
        learnable = (df["Machine failure"] == 1) & (df[PRIORITY].sum(axis=1) > 0)
        return df.index[~(no_failure | learnable)]
    raise ValueError("task must be 'binary' or 'multi'")

def multiclass_label(df):
    """One label per row. With several modes, the rarest wins."""

    label = pd.Series("No failure", index=df.index)
    for mode in reversed(PRIORITY):
        label[df[mode] == 1] = mode
    return label

def add_features(df):
    """Physics-based features that match how the failure modes are defined."""

    out = df.copy()
    out["Temp diff [K]"] = out["Process temperature [K]"] - out["Air temperature [K]"]
    out["Power [W]"] = out["Torque [Nm]"] * out["Rotational speed [rpm]"] * 2 * np.pi / 60 # Watts = Nm * rad/s | rpm to rad/s = rpm * 2pi/60
    out["Strain [min Nm]"] = out["Tool wear [min]"] * out["Torque [Nm]"]
    return out

def preprocess(df, task):
    """
    Parameters:
    df: pd.DataFrame
    task: str, either "binary" or "multi"

    Returns:
    X: pd.DataFrame, raw + engineered features
    y: pd.Series, target depending on binary or multiclass
    """

    df = add_features(df.drop(rows_to_drop(df, task)))
    X = df[NUMERIC + ["Type"]].copy()
    X["Type"] = pd.Categorical(X["Type"], categories=["L", "M", "H"])
    X = pd.get_dummies(X, columns=["Type"], dtype=int)
    if task == "binary":
        y = df["Machine failure"]
    else:
        y = multiclass_label(df).map({c: i for i, c in enumerate(CLASSES)})
    return X, y.rename("target")

def make_preprocessor(scale=True):
    """One-hot Type, and standardise the numeric columns if scale=True. Fit on training data only."""

    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([
        ("num", num, NUMERIC),
        ("type", OneHotEncoder(categories=[TYPES], sparse_output=False), ["Type"]),
    ], verbose_feature_names_out=False)
