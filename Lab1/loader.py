import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, scale
from sklearn.decomposition import PCA
import warnings
from phik import resources, report
from phik.report import plot_correlation_matrix
from scipy.io import arff


def load_and_clean_data(
    file_path,
    cat_cols,
    num_cols,
    handle_missing=True,
    scale=True,
    scale_time_based=False,
    onehot=True,
    return_cols=False,
):
    """
    Load the ARFF data and perform initial cleaning
    """
    print("Loading and cleaning the data...")

    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)

    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = df[col].str.decode("utf-8")
            except Exception:
                pass

    # Only keep columns specified in cat_cols and num_cols
    keep_cols = [col for col in cat_cols + num_cols if col in df.columns]
    df = df[keep_cols]

    print(f"\nDataset dimensions: {df.shape[0]} rows and {df.shape[1]} columns")
    print("\nFirst few rows:")
    print(df.head())

    for col in cat_cols:
        if col in df.columns:
            print(f"\nConverting {col} to categorical")
            df[col] = df[col].astype("category")

    for col in num_cols:
        if col in df.columns:
            print(f"\nConverting {col} to numeric")
            df[col] = pd.to_numeric(df[col], errors="coerce")

    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("\nMissing values found:")
        print(missing_values[missing_values > 0])
    else:
        print("\nNo missing values found.")

    if scale or scale_time_based:
        # special handling for time based event columns
        event_cols = ["time_to_aGvHD_III_IV", "PLTrecovery", "ANCrecovery"]
        for col in event_cols:
            if col in df.columns:
                never_occurred_col = f"{col}_never_occurred"
                df[never_occurred_col] = (df[col] == 1_000_000).astype(int)
                df[never_occurred_col] = df[never_occurred_col].astype("category")
                # replace 1_000_000 with nan
                df[col] = df[col].replace(1_000_000, np.nan)
                # add the new binary col to cat_cols for one-hot encoding
                if never_occurred_col not in cat_cols:
                    cat_cols.append(never_occurred_col)
                if col not in num_cols:
                    num_cols.append(col)

    if handle_missing:
        for col in num_cols:
            print(f"\nFilling missing values in {col} with mean")
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mean())
        for col in cat_cols:
            print(f"\nFilling missing values in {col} with mode")
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0])

    y = None
    X = df
    if "survival_status" in df.columns:
        y = df["survival_status"]
        X = df.drop(columns=["survival_status"])

    if scale:
        scaler = StandardScaler()
        cols_to_scale = [
            col for col in num_cols if col in df.columns and col != "survival_status"
        ]
        df[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])
        if y is not None and "survival_status" not in df.columns:
            df["survival_status"] = y

    if onehot:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    if return_cols:
        return df, cat_cols, num_cols

    return df
