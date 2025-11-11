"""Data preparation pipeline for the Bahrain land price model."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd

from utils import get_project_paths


@dataclass
class FeatureLookupTables:
    """Container for aggregate price-per-m² lookups fitted on training data."""

    classification: pd.Series
    location: pd.Series
    loc_class: pd.Series
    loc_class_broker: pd.Series
    global_mean: float

    @classmethod
    def from_training_frame(cls, df: pd.DataFrame) -> "FeatureLookupTables":
        """Create lookup tables based solely on the provided training frame."""

        working = df.copy()
        working["price_per_m2"] = working["price"] / working["Size"]

        classification = working.groupby("Classification")["price_per_m2"].mean()
        location = working.groupby("Location")["price_per_m2"].mean()
        loc_class = working.groupby(["Location", "Classification"])["price_per_m2"].mean()
        loc_class_broker = working.groupby(
            ["Location", "Classification", "Broker"]
        )["price_per_m2"].mean()
        global_mean = working["price_per_m2"].mean()

        return cls(
            classification=classification,
            location=location,
            loc_class=loc_class,
            loc_class_broker=loc_class_broker,
            global_mean=global_mean,
        )

    def apply_to_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted lookup tables to a dataframe without refitting."""

        enriched = df.copy()

        if {"price", "Size"}.issubset(enriched.columns):
            enriched["price_per_m2"] = enriched["price"] / enriched["Size"].replace(0, pd.NA)

        enriched = enriched.merge(
            self.classification.rename("Price_per_m2_per_Classification").reset_index(),
            on="Classification",
            how="left",
        )

        enriched = enriched.merge(
            self.location.rename("Price_per_m2_per_Location").reset_index(),
            on="Location",
            how="left",
        )

        enriched = enriched.merge(
            self.loc_class.rename("LocClass_avg_price_per_m2").reset_index(),
            on=["Location", "Classification"],
            how="left",
        )

        enriched = enriched.merge(
            self.loc_class_broker.rename("locclsbrk_ppm2").reset_index(),
            on=["Location", "Classification", "Broker"],
            how="left",
        )

        class_vals = enriched["Price_per_m2_per_Classification"].fillna(self.global_mean)
        loc_vals = enriched["Price_per_m2_per_Location"].fillna(self.global_mean)

        loc_class_vals = (
            enriched["LocClass_avg_price_per_m2"]
            .fillna(class_vals)
            .fillna(loc_vals)
            .fillna(self.global_mean)
        )

        locclsbrk_vals = (
            enriched["locclsbrk_ppm2"]
            .fillna(loc_class_vals)
            .fillna(class_vals)
            .fillna(loc_vals)
            .fillna(self.global_mean)
        )

        enriched["Price_per_m2_per_Classification"] = class_vals
        enriched["Price_per_m2_per_Location"] = loc_vals
        enriched["LocClass_avg_price_per_m2"] = loc_class_vals
        enriched["locclsbrk_ppm2"] = locclsbrk_vals

        return enriched

    def save(self, directory: Path, *, export_excel: bool = False) -> None:
        """Persist lookup tables for reuse during inference."""

        directory.mkdir(parents=True, exist_ok=True)

        classification_df = self.classification.rename(
            "Price_per_m2_per_Classification"
        ).reset_index()
        location_df = self.location.rename("Price_per_m2_per_Location").reset_index()
        loc_class_df = self.loc_class.rename("LocClass_avg_price_per_m2").reset_index()
        locclsbrk_df = self.loc_class_broker.rename("locclsbrk_ppm2").reset_index()
        global_df = pd.DataFrame({"global_price_per_m2_mean": [self.global_mean]})

        classification_df.to_csv(directory / "classification.csv", index=False)
        location_df.to_csv(directory / "location.csv", index=False)
        loc_class_df.to_csv(directory / "location_classification.csv", index=False)
        locclsbrk_df.to_csv(directory / "location_classification_broker.csv", index=False)
        global_df.to_csv(directory / "global.csv", index=False)

        if export_excel:
            excel_path = directory / "feature_lookups.xlsx"
            with pd.ExcelWriter(excel_path) as writer:
                classification_df.to_excel(writer, sheet_name="classification", index=False)
                location_df.to_excel(writer, sheet_name="location", index=False)
                loc_class_df.to_excel(writer, sheet_name="loc_class", index=False)
                locclsbrk_df.to_excel(writer, sheet_name="loc_class_broker", index=False)
                global_df.to_excel(writer, sheet_name="global", index=False)


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


def engineer_features(
    df: pd.DataFrame,
    *,
    lookups: FeatureLookupTables | None = None,
    return_lookups: bool = False,
) -> Tuple[pd.DataFrame, FeatureLookupTables] | pd.DataFrame:
    """Apply feature engineering using training-derived lookup tables.

    When ``lookups`` is ``None`` the function will fit lookup tables on the
    provided dataframe. To avoid data leakage, callers must request the fitted
    tables via ``return_lookups=True`` so they can be re-used for validation or
    inference splits.
    """

    df = df.copy()
    legacy_road_cols = ["has_road", "roads_capped"]
    existing_legacy_cols = [col for col in legacy_road_cols if col in df.columns]
    if existing_legacy_cols:
        df = df.drop(columns=existing_legacy_cols)

    if lookups is None:
        if not return_lookups:
            raise ValueError(
                "Feature lookups were not provided. Call engineer_features with "
                "return_lookups=True on the training split to obtain them."
            )
        lookups = FeatureLookupTables.from_training_frame(df)

    enriched = lookups.apply_to_frame(df)

    if return_lookups:
        return enriched, lookups
    return enriched
