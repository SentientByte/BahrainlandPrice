from itertools import product
import math

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from xgboost import XGBRegressor, plot_importance
import matplotlib.pyplot as plt
import category_encoders as ce

from data_pipeline import engineer_features, prepare_base_dataframe
from utils import get_project_paths, print_header


def _train_model(df: pd.DataFrame, *, save_outputs: bool = False, verbose: bool = True):
    # STEP 1: Locate the output directory so trained artefacts can be persisted.
    _, _, _, output_dir = get_project_paths()

    # STEP 2: Separate categorical and numerical predictors used by the model.
    feature_cols_categ = ["Location", "Classification", "Broker"]
    feature_cols_num = [
        "Size",
        "Roads",
        "Price_per_m2_per_Classification",
        "Price_per_m2_per_Location",
        "LocClass_avg_price_per_m2",
        "locclsbrk_ppm2",
    ]

    # STEP 3: Split the engineered dataframe into predictors and target.
    X = df[feature_cols_categ + feature_cols_num].copy()
    y = df["price"].copy()

    # STEP 4: Create train/test splits to evaluate performance.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    if verbose:
        print(f"[INFO] Train rows: {len(X_train)}, Test rows: {len(X_test)}")

    # STEP 5: Target encode the categorical predictors using training targets.
    if verbose:
        print("[INFO] Applying target encoding on: Location, Classification, Broker ...")
    te = ce.TargetEncoder(cols=feature_cols_categ, smoothing=0.3)
    X_train_te = te.fit_transform(X_train, y_train)
    X_test_te = te.transform(X_test)

    # STEP 6: Add an interaction feature capturing the synergy between encoded columns.
    X_train_te["Loc_x_Class"] = X_train_te["Location"] * X_train_te["Classification"]
    X_test_te["Loc_x_Class"] = X_test_te["Location"] * X_test_te["Classification"]

    final_features = list(X_train_te.columns)

    # STEP 7: Instantiate the gradient boosting regressor with tuned hyperparameters.
    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=7,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
    )

    # STEP 8: Fit the model on the encoded training data.
    if verbose:
        print("[INFO] Training model (kBHD target)...")
    model.fit(X_train_te[final_features], y_train)

    # STEP 9: Generate predictions for the held-out test set.
    y_pred = model.predict(X_test_te[final_features])

    # STEP 10: Compute standard regression metrics to measure performance.
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    if verbose:
        print(f"[MODEL] R²:   {r2:.4f}")
        print(f"[MODEL] RMSE: {rmse:,.2f} kBHD")

    # STEP 11: Optionally export predictions, encoded tables, and feature importance.
    if save_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        pred_df = X_test.copy()
        pred_df["Actual Price (kBHD)"] = y_test.values.round(0).astype(int)
        pred_df["Predicted Price (kBHD)"] = y_pred.round(0).astype(int)

        out_file = output_dir / "predicted_results.xlsx"
        pred_df.to_excel(out_file, index=False)
        if verbose:
            print(f"[INFO] Saved predictions to {out_file}")

        train_table_path = output_dir / "train_table.xlsx"
        train_table = X_train_te.copy()
        train_table["price"] = y_train.values
        train_table.to_excel(train_table_path, index=False)
        if verbose:
            print(f"[INFO] Saved encoded training table to {train_table_path}")

        test_table_path = output_dir / "test_table.xlsx"
        test_table = X_test_te.copy()
        test_table["price"] = y_test.values
        test_table.to_excel(test_table_path, index=False)
        if verbose:
            print(f"[INFO] Saved encoded testing table to {test_table_path}")

        fig_path = output_dir / "feature_importance.png"
        plt.figure(figsize=(10, 6))
        plot_importance(model, max_num_features=12)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        if verbose:
            print(f"[INFO] Saved feature importance plot to {fig_path}")

    # STEP 12: Return the fitted model and its evaluation metrics to the caller.
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
    # STEP 1: Announce the beginning of the trimming grid-search process.
    print_header("AUTO TUNE TRIMMING")

    # STEP 2: Prepare the base dataframe that every simulation will start from.
    base_df = prepare_base_dataframe()

    # STEP 3: Build trimming option grids for each categorical column of interest.
    trim_targets = ["Broker", "Classification", "Location"]
    trim_options = {col: _generate_trim_options(base_df[col]) for col in trim_targets}

    # STEP 4: Calculate how many total configurations will be tested.
    total_combinations = 1
    for col in trim_targets:
        total_combinations *= len(trim_options[col])

    print(f"[INFO] Evaluating {total_combinations} trimming configurations ...")

    results = []

    # STEP 5: Iterate through every possible combination of trimming options.
    completed = 0
    for broker_opt, class_opt, loc_opt in product(
        trim_options["Broker"], trim_options["Classification"], trim_options["Location"]
    ):
        df_variant = base_df.copy()
        base_rows = len(df_variant)

        # STEP 6: Apply the chosen grouping rules to the working dataframe.
        df_variant = _apply_grouping(df_variant, "Broker", broker_opt["keep"])
        df_variant = _apply_grouping(df_variant, "Classification", class_opt["keep"])
        df_variant = _apply_grouping(df_variant, "Location", loc_opt["keep"])

        # STEP 7: Ensure the grouping logic does not accidentally drop rows.
        if len(df_variant) != base_rows:
            raise RuntimeError("Auto tune grouping removed rows unexpectedly.")

        # STEP 8: Engineer features and train a model for this configuration.
        engineered_df = engineer_features(df_variant)
        metrics = _train_model(engineered_df, save_outputs=False, verbose=False)

        # STEP 9: Record the trimming configuration alongside evaluation metrics.
        results.append({
            "broker": broker_opt,
            "classification": class_opt,
            "location": loc_opt,
            "r2": metrics["r2"],
            "rmse": metrics["rmse"],
        })

        # STEP 10: Emit a progress update for visibility.
        completed += 1
        remaining = total_combinations - completed
        print(
            f"[PROGRESS] Simulations completed: {completed}/{total_combinations} "
            f"({remaining} remaining)"
        )

    # STEP 11: Order the results to surface the best-performing combinations.
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

    # STEP 12: Print a table displaying the leading configurations.
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
    # STEP 1: Announce the primary training/testing routine.
    print_header("TEACH / TRAIN & TEST THE MODEL")

    # STEP 2: Prepare the cleaned dataset and engineer features for modelling.
    base_df = prepare_base_dataframe()
    engineered_df = engineer_features(base_df)

    # STEP 3: Train the model and persist outputs for stakeholders.
    _train_model(engineered_df, save_outputs=True, verbose=True)
