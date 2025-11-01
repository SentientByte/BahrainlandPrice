from itertools import product
import math

import numpy as np
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


def _build_feature_adder(train_df: pd.DataFrame):
    train_df = train_df.copy()
    train_df["price_per_m2"] = train_df["price"] / train_df["Size"]

    loc_ppm2 = train_df.groupby("Location")["price_per_m2"].mean()
    cls_ppm2 = train_df.groupby("Classification")["price_per_m2"].mean()
    loccls_ppm2 = train_df.groupby(["Location", "Classification"])["price_per_m2"].mean()
    broker_ppm2 = train_df.groupby("Broker")["price_per_m2"].mean()
    global_ppm2 = train_df["price_per_m2"].mean()

    loc_count = train_df.groupby("Location").size()
    cls_count = train_df.groupby("Classification").size()
    broker_count = train_df.groupby("Broker").size()

    loccls_ppm2_dict = loccls_ppm2.to_dict()

    def add_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["loc_ppm2"] = df["Location"].map(loc_ppm2).fillna(global_ppm2)
        df["cls_ppm2"] = df["Classification"].map(cls_ppm2).fillna(global_ppm2)
        df["broker_ppm2"] = df["Broker"].map(broker_ppm2).fillna(global_ppm2)
        loccls_keys = pd.Series(
            list(zip(df["Location"], df["Classification"])), index=df.index
        )
        df["loccls_ppm2"] = loccls_keys.map(loccls_ppm2_dict)
        df["loccls_ppm2"] = df["loccls_ppm2"].fillna(df["loc_ppm2"])

        df["baseline_loc_price"] = df["loc_ppm2"] * df["Size"]
        df["baseline_cls_price"] = df["cls_ppm2"] * df["Size"]
        df["baseline_broker_price"] = df["broker_ppm2"] * df["Size"]

        df["log_size"] = np.log1p(df["Size"])
        df["sqrt_size"] = np.sqrt(df["Size"])

        df["has_road"] = (df["Roads"] > 0).astype(int)
        df["roads_capped"] = np.minimum(df["Roads"], 3)

        df["loc_count"] = df["Location"].map(loc_count).fillna(0)
        df["class_count"] = df["Classification"].map(cls_count).fillna(0)
        df["broker_count"] = df["Broker"].map(broker_count).fillna(0)
        df["broker_count"] = df["broker_count"].clip(upper=200)

        return df

    return add_features


def _fit_default_grouping(train_df: pd.DataFrame) -> dict:
    thresholds = {"Location": 25, "Classification": 15, "Broker": 100}
    keep_map = {}
    for column, limit in thresholds.items():
        counts = train_df[column].value_counts()
        keep_values = counts.nlargest(limit).index
        keep_map[column] = set(keep_values)
    return keep_map


def _apply_grouping_map(df: pd.DataFrame, keep_map: dict) -> pd.DataFrame:
    df = df.copy()
    for column, keep_values in keep_map.items():
        df[column] = df[column].where(df[column].isin(keep_values), "Others")
    return df


def _prepare_train_test_features(df: pd.DataFrame):
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
    )

    feature_adder = _build_feature_adder(train_df)
    train_feat = feature_adder(train_df)
    test_feat = feature_adder(test_df)

    grouping_map = _fit_default_grouping(train_feat)
    train_grouped = _apply_grouping_map(train_feat, grouping_map)
    test_grouped = _apply_grouping_map(test_feat, grouping_map)

    feature_cols_categ = ["Location", "Classification", "Broker"]
    feature_cols_num = [
        "Size",
        "Roads",
        "loc_ppm2",
        "cls_ppm2",
        "broker_ppm2",
        "loccls_ppm2",
        "baseline_loc_price",
        "baseline_cls_price",
        "baseline_broker_price",
        "log_size",
        "sqrt_size",
        "has_road",
        "roads_capped",
        "loc_count",
        "class_count",
        "broker_count",
    ]

    final_features = feature_cols_categ + feature_cols_num

    X_train = train_grouped[final_features].copy()
    y_train = train_grouped["price"].copy()
    X_test = test_grouped[final_features].copy()
    y_test = test_grouped["price"].copy()

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "train_table": train_grouped,
        "test_table": test_grouped,
        "feature_cols_categ": feature_cols_categ,
        "final_features": final_features,
    }


def _train_model(df: pd.DataFrame, *, save_outputs: bool = False, verbose: bool = True):
    _, _, _, output_dir = get_project_paths()

    prepared = _prepare_train_test_features(df)

    X_train = prepared["X_train"]
    X_test = prepared["X_test"]
    y_train = prepared["y_train"]
    y_test = prepared["y_test"]
    feature_cols_categ = prepared["feature_cols_categ"]
    final_features = prepared["final_features"]

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
        pred_df = prepared["test_table"].copy()
        pred_df["Actual Price (kBHD)"] = y_test.values.round(0).astype(int)
        pred_df["Predicted Price (kBHD)"] = y_pred.round(0).astype(int)

        out_file = output_dir / "predicted_results.xlsx"
        pred_df.to_excel(out_file, index=False)
        if verbose:
            print(f"[INFO] Saved predictions to {out_file}")

        train_table_path = output_dir / "train_table.xlsx"
        train_table_encoded = X_train_te.copy()
        train_table_encoded["price"] = y_train.values
        train_table_encoded.to_excel(train_table_path, index=False)
        if verbose:
            print(f"[INFO] Saved encoded training table to {train_table_path}")

        test_table_path = output_dir / "test_table.xlsx"
        test_table_encoded = X_test_te.copy()
        test_table_encoded["price"] = y_test.values
        test_table_encoded.to_excel(test_table_path, index=False)
        if verbose:
            print(f"[INFO] Saved encoded testing table to {test_table_path}")

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


def _generate_trim_options(series: pd.Series, *, max_thresholds: int = 25):
    counts = series.value_counts()
    categories = list(counts.index)
    total_unique = len(categories)
    options = []
    seen = set()

    candidate_sizes = list(range(total_unique, -1, -1))

    if len(candidate_sizes) > max_thresholds:
        # Sample at most ``max_thresholds`` values (including the extremes)
        sampled_indices = set()
        last_index = len(candidate_sizes) - 1
        if max_thresholds == 1:
            sampled_indices.add(0)
        else:
            for i in range(max_thresholds):
                raw_idx = round(i * last_index / (max_thresholds - 1))
                sampled_indices.add(raw_idx)
        candidate_sizes = [candidate_sizes[idx] for idx in sorted(sampled_indices)]

    for k in candidate_sizes:
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

    completed = 0
    for broker_opt, class_opt, loc_opt in product(
        trim_options["Broker"], trim_options["Classification"], trim_options["Location"]
    ):
        df_variant = base_df.copy()
        base_rows = len(df_variant)
        df_variant = _apply_grouping(df_variant, "Broker", broker_opt["keep"])
        df_variant = _apply_grouping(df_variant, "Classification", class_opt["keep"])
        df_variant = _apply_grouping(df_variant, "Location", loc_opt["keep"])

        # Sanity check: grouping should never remove rows.
        if len(df_variant) != base_rows:
            raise RuntimeError("Auto tune grouping removed rows unexpectedly.")

        metrics = _train_model(df_variant, save_outputs=False, verbose=False)

        results.append({
            "broker": broker_opt,
            "classification": class_opt,
            "location": loc_opt,
            "r2": metrics["r2"],
            "rmse": metrics["rmse"],
        })

        completed += 1
        remaining = total_combinations - completed
        print(
            f"[PROGRESS] Simulations completed: {completed}/{total_combinations} "
            f"({remaining} remaining)"
        )

    results.sort(key=lambda item: (-item["r2"], item["rmse"]))

    leaderboard_limit = 10
    print(
        f"\nTop {leaderboard_limit} trimming configurations (sorted by R² desc, RMSE asc):"
    )

    header_cols = [
        ("Rank", 6),
        ("Broker", 20),
        ("Classification", 24),
        ("Location", 20),
        ("R²", 10),
        ("RMSE", 14),
    ]

    def _render_row(rank_label, broker_label, class_label, loc_label, r2_value, rmse_value):
        return (
            f"| {rank_label:<{header_cols[0][1]-2}}"
            f"| {broker_label:<{header_cols[1][1]-2}}"
            f"| {class_label:<{header_cols[2][1]-2}}"
            f"| {loc_label:<{header_cols[3][1]-2}}"
            f"| {r2_value:>{header_cols[4][1]-2}}"
            f"| {rmse_value:>{header_cols[5][1]-2}} |"
        )

    horizontal_rule = "+" + "+".join("-" * (width) for _, width in header_cols) + "+"

    print(horizontal_rule)
    header_row = (
        f"| {'Rank':^{header_cols[0][1]-2}}"
        f"| {'Broker':^{header_cols[1][1]-2}}"
        f"| {'Classification':^{header_cols[2][1]-2}}"
        f"| {'Location':^{header_cols[3][1]-2}}"
        f"| {'R²':^{header_cols[4][1]-2}}"
        f"| {'RMSE':^{header_cols[5][1]-2}} |"
    )
    print(header_row)
    print(horizontal_rule)

    for idx, entry in enumerate(results[:leaderboard_limit], start=1):
        broker_label = _describe_keep(entry["broker"]["keep"], entry["broker"]["total"])
        class_label = _describe_keep(entry["classification"]["keep"], entry["classification"]["total"])
        loc_label = _describe_keep(entry["location"]["keep"], entry["location"]["total"])
        print(
            _render_row(
                idx,
                broker_label,
                class_label,
                loc_label,
                f"{entry['r2']:.4f}",
                f"{entry['rmse']:,.2f}",
            )
        )

    print(horizontal_rule)


def train_and_test():
    print_header("TEACH / TRAIN & TEST THE MODEL")

    base_df = _prepare_base_dataframe()
    _train_model(base_df, save_outputs=True, verbose=True)
