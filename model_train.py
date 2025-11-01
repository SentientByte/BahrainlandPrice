from pathlib import Path
import math

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from xgboost import XGBRegressor, plot_importance
import matplotlib.pyplot as plt
import category_encoders as ce

from utils import get_project_paths, print_header


def _load_training_df():
    """
    Try to load model_ready.xlsx from sensible locations.
    Order:
      1) /data/model_ready.xlsx
      2) /output/model_ready.xlsx
      3) /data/data.xlsx   <-- fallback
    """
    base_dir, venv_dir, data_dir, output_dir = get_project_paths()

    # 1) the ideal location
    p1 = data_dir / "model_ready.xlsx"
    if p1.exists():
        print(f"[INFO] Using training data from {p1}")
        return pd.read_excel(p1)

    # 2) sometimes cleaning scripts save it in /output
    p2 = output_dir / "model_ready.xlsx"
    if p2.exists():
        print(f"[INFO] Using training data from {p2}")
        return pd.read_excel(p2)

    # 3) last resort: raw data.xlsx in /data
    p3 = data_dir / "data.xlsx"
    if p3.exists():
        print(f"[WARN] model_ready.xlsx not found. Using {p3} instead.")
        return pd.read_excel(p3)

    # nothing found
    raise FileNotFoundError(
        f"model_ready.xlsx not found in {data_dir} or {output_dir}, and data.xlsx also missing."
    )


def train_and_test():
    print_header("TEACH / TRAIN & TEST THE MODEL")

    # -------------------------------------------------
    # 1) load data (with the smarter loader)
    # -------------------------------------------------
    base_dir, venv_dir, data_dir, output_dir = get_project_paths()
    df = _load_training_df()

    # expected columns (from your original script)
    expected_cols = ["Location", "Size", "Classification", "Roads", "Broker", "price"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in training data: {missing}")

    # -------------------------------------------------
    # 2) Basic cleaning / casting
    # -------------------------------------------------
    df = df[expected_cols].copy()
    df = df.dropna(subset=expected_cols).copy()

    df["Size"] = pd.to_numeric(df["Size"], errors="coerce")
    df["Roads"] = pd.to_numeric(df["Roads"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")  # in kBHD (or close)

    df = df.dropna(subset=["Size", "Roads", "price"]).copy()

    # -------------------------------------------------
    # 3) OUTLIER TRIMMING (1% & 99%)  <-- from your file
    # -------------------------------------------------
    low_q, high_q = df["price"].quantile([0.01, 0.99])
    before_rows = len(df)
    df = df[(df["price"] >= low_q) & (df["price"] <= high_q)].copy()
    after_rows = len(df)
    print(f"[INFO] Outlier trimming (1% tails): {before_rows} -> {after_rows} rows kept")

    # -------------------------------------------------
    # 4) ENGINEERED FEATURES (ALL of them)
    # -------------------------------------------------
    # 4.1 size per classification
    avg_size_by_class = df.groupby("Classification")["Size"].transform("mean")
    df["Size_per_Classification"] = df["Size"] / (avg_size_by_class + 1e-6)

    # 4.2 size per location
    avg_size_by_loc = df.groupby("Location")["Size"].transform("mean")
    df["Size_per_Location"] = df["Size"] / (avg_size_by_loc + 1e-6)

    # 4.3 NEW: average price for each (Location, Classification) pair
    df["LocClass_avg_price"] = (
        df.groupby(["Location", "Classification"])["price"]
          .transform("mean")
    )

    # -------------------------------------------------
    # 5) Train/test split
    # -------------------------------------------------
    FEATURE_COLS_CATEG = ["Location", "Classification", "Broker"]
    FEATURE_COLS_NUM = [
        "Size",
        "Roads",
        "Size_per_Classification",
        "Size_per_Location",
        "LocClass_avg_price",
    ]

    X = df[FEATURE_COLS_CATEG + FEATURE_COLS_NUM].copy()
    y = df["price"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    print(f"[INFO] Train rows: {len(X_train)}, Test rows: {len(X_test)}")

    # -------------------------------------------------
    # 6) TARGET ENCODING (fit on TRAIN only)
    # -------------------------------------------------
    print("[INFO] Applying target encoding on: Location, Classification, Broker ...")
    te = ce.TargetEncoder(cols=FEATURE_COLS_CATEG, smoothing=0.3)
    X_train_te = te.fit_transform(X_train, y_train)
    X_test_te = te.transform(X_test)

    # -------------------------------------------------
    # 7) Add interaction AFTER encoding
    # -------------------------------------------------
    X_train_te["Loc_x_Class"] = X_train_te["Location"] * X_train_te["Classification"]
    X_test_te["Loc_x_Class"] = X_test_te["Location"] * X_test_te["Classification"]

    final_features = list(X_train_te.columns)

    # -------------------------------------------------
    # 8) SIMPLE XGBOOST (only 3 knobs)
    # -------------------------------------------------
    model = XGBRegressor(
        n_estimators=600,      # you can set to 900 later
        learning_rate=0.05,    # similar scale to original file
        max_depth=7,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
    )

    print("[INFO] Training RAW model (kBHD target)...")
    model.fit(X_train_te[final_features], y_train)

        # -------------------------------------------------
    # 9) Evaluate
    # -------------------------------------------------
    y_pred = model.predict(X_test_te[final_features])

    # old sklearn does NOT support squared=False, so do it manually
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"[RAW MODEL] R²:   {r2:.4f}")
    print(f"[RAW MODEL] RMSE: {rmse:,.2f} kBHD")


    # -------------------------------------------------
    # 10) Save predictions to /output
    # -------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_df = X_test.copy()
    pred_df["Actual Price (kBHD)"] = y_test.values.round(0).astype(int)
    pred_df["Predicted Price (kBHD)"] = y_pred.round(0).astype(int)

    out_file = output_dir / "predicted_results.xlsx"
    pred_df.to_excel(out_file, index=False)
    print(f"[INFO] Saved predictions to {out_file}")

    # -------------------------------------------------
    # 11) Feature importance to /output
    # -------------------------------------------------
    fig_path = output_dir / "feature_importance.png"
    plt.figure(figsize=(10, 6))
    plot_importance(model, max_num_features=12)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    print(f"[INFO] Saved feature importance plot to {fig_path}")
