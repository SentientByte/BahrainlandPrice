"""Data preparation pipeline for the Bahrain land price model."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from utils import get_project_paths


def _resolve_training_paths() -> Tuple[Path, Path]:
    """Return the data and output directories used to locate training data."""

    # STEP 1: Fetch project-level paths so we can locate the data folders.
    _, _, data_dir, output_dir = get_project_paths()

    # STEP 2: Hand back the directories that other helpers will rely on.
    return data_dir, output_dir


def load_training_dataframe() -> pd.DataFrame:
    """Load the training dataframe from the expected locations."""

    # STEP 1: Resolve the directories that might contain training artefacts.
    data_dir, output_dir = _resolve_training_paths()

    # STEP 2: Try the canonical, model-ready export created by the cleaning pipeline.
    p1 = data_dir / "model_ready.xlsx"
    if p1.exists():
        print(f"[INFO] Using training data from {p1}")
        return pd.read_excel(p1)

    # STEP 3: If the cleaned file was saved to the output directory, use that instead.
    p2 = output_dir / "model_ready.xlsx"
    if p2.exists():
        print(f"[INFO] Using training data from {p2}")
        return pd.read_excel(p2)

    # STEP 4: Fall back to the original raw export if no cleaned file is available.
    p3 = data_dir / "data.xlsx"
    if p3.exists():
        print(f"[WARN] model_ready.xlsx not found. Using {p3} instead.")
        return pd.read_excel(p3)

    # STEP 5: Abort with a clear error when no source file can be located.
    raise FileNotFoundError(
        f"model_ready.xlsx not found in {data_dir} or {output_dir}, and data.xlsx also missing."
    )


def prepare_base_dataframe() -> pd.DataFrame:
    """Clean the raw dataframe and prepare the base features for modelling."""

    # STEP 1: Load whichever input dataset is available.
    df = load_training_dataframe()

    # STEP 2: Verify and keep only the columns that downstream modelling expects.
    expected_cols = ["Location", "Size", "Classification", "Roads", "Broker", "price"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in training data: {missing}")

    df = df[expected_cols].copy()

    # STEP 3: Drop fully empty records before converting data types.
    df = df.dropna(subset=expected_cols).copy()

    # STEP 4: Coerce numeric inputs to numbers so calculations work reliably.
    df["Size"] = pd.to_numeric(df["Size"], errors="coerce")
    df["Roads"] = pd.to_numeric(df["Roads"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # STEP 5: Remove any rows that still have missing numeric values after coercion.
    df = df.dropna(subset=["Size", "Roads", "price"]).copy()

    # STEP 6: Trim extreme price outliers so the model is trained on stable targets.
    low_q, high_q = df["price"].quantile([0.01, 0.99])
    before_rows = len(df)
    df = df[(df["price"] >= low_q) & (df["price"] <= high_q)].copy()
    after_rows = len(df)
    print(f"[INFO] Outlier trimming (1% tails): {before_rows} -> {after_rows} rows kept")

    # STEP 7: Return the cleaned dataset to be fed into feature engineering.
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the engineered features required by the model."""

    # STEP 1: Start from a defensive copy to avoid mutating caller data.
    df = df.copy()

    # STEP 2: Remove legacy columns that may still exist on older exports.
    legacy_road_cols = ["has_road", "roads_capped"]
    existing_legacy_cols = [col for col in legacy_road_cols if col in df.columns]
    if existing_legacy_cols:
        df = df.drop(columns=existing_legacy_cols)

    # STEP 3: Compute price-per-square-metre to normalise target values.
    df["price_per_m2"] = df["price"] / df["Size"]

    # STEP 4: Aggregate average prices per classification to capture high-level trends.
    df["Price_per_m2_per_Classification"] = (
        df.groupby("Classification")["price_per_m2"].transform("mean")
    )

    # STEP 5: Aggregate average prices per location to incorporate regional signals.
    df["Price_per_m2_per_Location"] = (
        df.groupby("Location")["price_per_m2"].transform("mean")
    )

    # STEP 6: Blend location and classification for a finer-grained average.
    df["LocClass_avg_price_per_m2"] = (
        df.groupby(["Location", "Classification"])["price_per_m2"].transform("mean")
    )

    # STEP 7: Extend the aggregation to also capture broker-specific behaviour.
    df["locclsbrk_ppm2"] = (
        df.groupby(["Location", "Classification", "Broker"])["price_per_m2"].transform("mean")
    )

    # STEP 8: Return the enriched dataframe ready for training.
    return df
