import json
import math

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from xgboost import XGBRegressor, plot_importance
import matplotlib.pyplot as plt
import category_encoders as ce

from data_pipeline import FeatureLookupTables, engineer_features, prepare_base_dataframe
from utils import get_project_paths, print_header


FEATURE_COLS_CATEG = ["Location", "Classification", "Broker"]
FEATURE_COLS_NUM = [
    "Size",
    "Roads",
    "Price_per_m2_per_Classification",
    "Price_per_m2_per_Location",
    "LocClass_avg_price_per_m2",
    "locclsbrk_ppm2",
]

MODEL_ARTIFACTS_DIRNAME = "model_artifacts"
MODEL_FILENAME = "xgb_model.joblib"
ENCODER_FILENAME = "target_encoder.joblib"
FEATURE_LIST_FILENAME = "feature_columns.json"


def _train_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    save_outputs: bool = False,
    verbose: bool = True,
):
    # STEP 1: Locate the output directory so trained artefacts can be persisted.
    _, _, _, output_dir = get_project_paths()

    # STEP 2: Separate categorical and numerical predictors used by the model.
    # STEP 3: Split the engineered dataframe into predictors and target.
    X_train = train_df[FEATURE_COLS_CATEG + FEATURE_COLS_NUM].copy()
    y_train = train_df["price"].copy()
    X_test = test_df[FEATURE_COLS_CATEG + FEATURE_COLS_NUM].copy()
    y_test = test_df["price"].copy()

    if verbose:
        print(f"[INFO] Train rows: {len(X_train)}, Test rows: {len(X_test)}")

    # STEP 5: Target encode the categorical predictors using training targets.
    if verbose:
        print("[INFO] Applying target encoding on: Location, Classification, Broker ...")
    te = ce.TargetEncoder(cols=FEATURE_COLS_CATEG, smoothing=0.3)
    X_train_te = te.fit_transform(X_train, y_train)
    X_test_te = te.transform(X_test)

    # STEP 6: Add an interaction feature capturing the synergy between encoded columns.
    X_train_te["Loc_x_Class"] = X_train_te["Location"] * X_train_te["Classification"]
    X_test_te["Loc_x_Class"] = X_test_te["Location"] * X_test_te["Classification"]

    final_features = FEATURE_COLS_CATEG + FEATURE_COLS_NUM + ["Loc_x_Class"]

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
        pred_df = test_df[FEATURE_COLS_CATEG + FEATURE_COLS_NUM].copy()
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

        artifacts_dir = output_dir / MODEL_ARTIFACTS_DIRNAME
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        model_path = artifacts_dir / MODEL_FILENAME
        encoder_path = artifacts_dir / ENCODER_FILENAME
        feature_path = artifacts_dir / FEATURE_LIST_FILENAME

        joblib.dump(model, model_path)
        joblib.dump(te, encoder_path)
        with feature_path.open("w", encoding="utf-8") as fh:
            json.dump(final_features, fh)
        if verbose:
            print(f"[INFO] Saved model artefacts to {artifacts_dir}")

    # STEP 12: Return the fitted model and its evaluation metrics to the caller.
    return {
        "model": model,
        "rmse": rmse,
        "r2": r2,
    }


def train_and_test():
    # STEP 1: Announce the primary training/testing routine.
    print_header("TEACH / TRAIN & TEST THE MODEL")

    # STEP 2: Prepare the cleaned dataset and split before feature engineering.
    base_df = prepare_base_dataframe()
    train_df, test_df = train_test_split(base_df, test_size=0.2, shuffle=True)

    # STEP 3: Fit feature lookups on the training data and apply them to both splits.
    train_engineered, lookups = engineer_features(train_df, return_lookups=True)
    test_engineered = engineer_features(test_df, lookups=lookups)

    # STEP 4: Persist lookup tables for reuse by inference code and analysts.
    _, _, _, output_dir = get_project_paths()
    lookup_dir = output_dir / "feature_lookups"
    lookups.save(lookup_dir, export_excel=True)

    # STEP 5: Train the model and persist outputs for stakeholders.
    _train_model(train_engineered, test_engineered, save_outputs=True, verbose=True)


def real_world_test():
    # STEP 1: Guide the user towards training the model if artefacts are missing.
    print_header("REAL WORLD TEST")

    _, _, _, output_dir = get_project_paths()

    artifacts_dir = output_dir / MODEL_ARTIFACTS_DIRNAME
    lookups_dir = output_dir / "feature_lookups"

    model_path = artifacts_dir / MODEL_FILENAME
    encoder_path = artifacts_dir / ENCODER_FILENAME
    feature_path = artifacts_dir / FEATURE_LIST_FILENAME

    missing_paths = [
        path
        for path in [artifacts_dir, lookups_dir, model_path, encoder_path, feature_path]
        if not path.exists()
    ]

    if missing_paths:
        print("[WARN] Trained model artefacts were not found. Please run option 4 first.")
        for path in missing_paths:
            print(f"       Missing: {path}")
        return

    lookups = FeatureLookupTables.load(lookups_dir)
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    with feature_path.open("r", encoding="utf-8") as fh:
        final_features = json.load(fh)

    print(
        "Enter real-world plot details as: Location, Classification, Size, Roads, Broker"
    )
    print("Type 'q' to return to the menu.")

    while True:
        raw = input("Input: ").strip()
        if raw.lower() in {"q", "quit", "exit", ""}:
            break

        parts = [part.strip() for part in raw.split(",")]
        if len(parts) != 5:
            print("[ERROR] Please provide exactly 5 values separated by commas.")
            continue

        location, classification, size_str, roads_str, broker = parts

        try:
            size = float(size_str)
            roads = float(roads_str)
        except ValueError:
            print("[ERROR] Size and Roads must be numeric values.")
            continue

        input_df = pd.DataFrame(
            [
                {
                    "Location": location,
                    "Classification": classification,
                    "Size": size,
                    "Roads": roads,
                    "Broker": broker,
                }
            ]
        )

        enriched = lookups.apply_to_frame(input_df)
        features_df = enriched[FEATURE_COLS_CATEG + FEATURE_COLS_NUM].copy()
        encoded_df = encoder.transform(features_df)

        if not isinstance(encoded_df, pd.DataFrame):
            encoded_df = pd.DataFrame(encoded_df, columns=features_df.columns)

        encoded_df["Loc_x_Class"] = (
            encoded_df["Location"] * encoded_df["Classification"]
        )

        for column in final_features:
            if column not in encoded_df.columns:
                encoded_df[column] = 0.0

        encoded_df = encoded_df[final_features]

        prediction = float(model.predict(encoded_df)[0])
        print(f"[PREDICTION] Estimated price: {prediction:,.2f} k BHD")
