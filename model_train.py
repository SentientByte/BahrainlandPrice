from itertools import product
import math

import pandas as pd

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


def _prepare_base_dataframe():
    df = _load_training_df()

    expected_cols = ["Location", "Size", "Classification", "Roads", "Broker", "price"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in training data: {missing}")

    df = df[expected_cols].copy()
    df = df.dropna(subset=expected_cols).copy()

    df["Size"] = pd.to_numeric(df["Size"], errors="coerce")
    df["Roads"] = pd.to_numeric(df["Roads"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df = df.dropna(subset=["Size", "Roads", "price"]).copy()

    low_q, high_q = df["price"].quantile([0.01, 0.99])
    before_rows = len(df)
    df = df[(df["price"] >= low_q) & (df["price"] <= high_q)].copy()
    after_rows = len(df)
    print(f"[INFO] Outlier trimming (1% tails): {before_rows} -> {after_rows} rows kept")

    return df


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    avg_size_by_class = df.groupby("Classification")["Size"].transform("mean")
    df["Size_per_Classification"] = df["Size"] / (avg_size_by_class + 1e-6)

    avg_size_by_loc = df.groupby("Location")["Size"].transform("mean")
    df["Size_per_Location"] = df["Size"] / (avg_size_by_loc + 1e-6)

    df["LocClass_avg_price"] = (
        df.groupby(["Location", "Classification"])["price"].transform("mean")
    )

    return df


def _train_model(df: pd.DataFrame, *, save_outputs: bool = False, verbose: bool = True):
    _, _, _, output_dir = get_project_paths()

    feature_cols_categ = ["Location", "Classification", "Broker"]
    feature_cols_num = [
        "Size",
        "Roads",
        "Size_per_Classification",
        "Size_per_Location",
        "LocClass_avg_price",
    ]

    X = df[feature_cols_categ + feature_cols_num].copy()
    y = df["price"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    if verbose:
        print(f"[INFO] Train rows: {len(X_train)}, Test rows: {len(X_test)}")

    if verbose:
        print("[INFO] Applying target encoding on: Location, Classification, Broker ...")
    te = ce.TargetEncoder(cols=feature_cols_categ, smoothing=0.3)
    X_train_te = te.fit_transform(X_train, y_train)
    X_test_te = te.transform(X_test)

    X_train_te["Loc_x_Class"] = X_train_te["Location"] * X_train_te["Classification"]
    X_test_te["Loc_x_Class"] = X_test_te["Location"] * X_test_te["Classification"]

    final_features = list(X_train_te.columns)

    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=7,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
    )

    if verbose:
        print("[INFO] Training model (kBHD target)...")
    model.fit(X_train_te[final_features], y_train)

    y_pred = model.predict(X_test_te[final_features])

    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    if verbose:
        print(f"[MODEL] R²:   {r2:.4f}")
        print(f"[MODEL] RMSE: {rmse:,.2f} kBHD")

    if save_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        pred_df = X_test.copy()
        pred_df["Actual Price (kBHD)"] = y_test.values.round(0).astype(int)
        pred_df["Predicted Price (kBHD)"] = y_pred.round(0).astype(int)

        out_file = output_dir / "predicted_results.xlsx"
        pred_df.to_excel(out_file, index=False)
        if verbose:
            print(f"[INFO] Saved predictions to {out_file}")

        fig_path = output_dir / "feature_importance.png"
        plt.figure(figsize=(10, 6))
        plot_importance(model, max_num_features=12)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        if verbose:
            print(f"[INFO] Saved feature importance plot to {fig_path}")

    return {
        "model": model,
        "rmse": rmse,
        "r2": r2,
    }


def _generate_trim_options(series: pd.Series):
    counts = series.value_counts()
    categories = list(counts.index)
    total_unique = len(categories)
    options = []
    seen = set()

    for k in range(total_unique, -1, -1):
        keep = tuple(categories[:k])
        key = tuple(sorted(keep))
        if key in seen:
            continue
        seen.add(key)
        options.append({
            "keep": keep,
            "total": total_unique,
        })

    return options


def _apply_grouping(df: pd.DataFrame, column: str, keep: tuple) -> pd.DataFrame:
    if keep is None:
        return df

    keep_set = set(keep)
    if len(keep_set) == 0:
        df[column] = "Others"
    else:
        if len(keep_set) == df[column].nunique():
            return df
        df[column] = df[column].where(df[column].isin(keep_set), "Others")
    return df


def _describe_keep(keep: tuple, total: int) -> str:
    if len(keep) == total:
        return f"All ({total})"
    if len(keep) == 0:
        return "All grouped"
    return f"Top {len(keep)}"


def auto_tune_trimming():
    print_header("AUTO TUNE TRIMMING")

    base_df = _prepare_base_dataframe()

    trim_targets = ["Broker", "Classification", "Location"]
    trim_options = {col: _generate_trim_options(base_df[col]) for col in trim_targets}

    total_combinations = 1
    for col in trim_targets:
        total_combinations *= len(trim_options[col])

    print(f"[INFO] Evaluating {total_combinations} trimming configurations ...")

    results = []

    for broker_opt, class_opt, loc_opt in product(
        trim_options["Broker"], trim_options["Classification"], trim_options["Location"]
    ):
        df_variant = base_df.copy()
        df_variant = _apply_grouping(df_variant, "Broker", broker_opt["keep"])
        df_variant = _apply_grouping(df_variant, "Classification", class_opt["keep"])
        df_variant = _apply_grouping(df_variant, "Location", loc_opt["keep"])

        engineered_df = _engineer_features(df_variant)
        metrics = _train_model(engineered_df, save_outputs=False, verbose=False)

        results.append({
            "broker": broker_opt,
            "classification": class_opt,
            "location": loc_opt,
            "r2": metrics["r2"],
            "rmse": metrics["rmse"],
        })

    results.sort(key=lambda item: (-item["r2"], item["rmse"]))

    print("\nTop 10 trimming configurations (sorted by R² desc, RMSE asc):")
    header = f"{'Rank':<6}{'Broker':<18}{'Classification':<20}{'Location':<18}{'R²':>10}{'RMSE':>14}"
    print(header)
    print("-" * len(header))

    for idx, entry in enumerate(results[:10], start=1):
        broker_label = _describe_keep(entry["broker"]["keep"], entry["broker"]["total"])
        class_label = _describe_keep(entry["classification"]["keep"], entry["classification"]["total"])
        loc_label = _describe_keep(entry["location"]["keep"], entry["location"]["total"])
        print(
            f"{idx:<6}{broker_label:<18}{class_label:<20}{loc_label:<18}"
            f"{entry['r2']:>10.4f}{entry['rmse']:>14,.2f}"
        )


def train_and_test():
    print_header("TEACH / TRAIN & TEST THE MODEL")

    base_df = _prepare_base_dataframe()
    engineered_df = _engineer_features(base_df)
    _train_model(engineered_df, save_outputs=True, verbose=True)
