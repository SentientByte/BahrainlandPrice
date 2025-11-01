import pandas as pd
import numpy as np
from pathlib import Path
from utils import get_project_paths, print_header


M2_TO_FT2 = 10.7639  # 1 m² = 10.7639 ft²


def _load_data():
    base_dir, venv_dir, data_dir, output_dir = get_project_paths()
    src_file = data_dir / "data.xlsx"
    if not src_file.exists():
        raise FileNotFoundError(f"Input file not found: {src_file}")
    df = pd.read_excel(src_file)
    return df


def clean_data_pipeline():
    base_dir, venv_dir, data_dir, output_dir = get_project_paths()
    df = _load_data()
    original_len = len(df)

    # Normalize column names and build lowercase map
    df.columns = [c.strip() for c in df.columns]
    colmap = {c.lower(): c for c in df.columns}

    # ---------------------------------------------------------------------
    # 1) COLUMN CLEANUP STEPS
    # ---------------------------------------------------------------------

    # Location: remove "Land For Sale in "
    if "location" in colmap:
        loc_col = colmap["location"]
        df[loc_col] = (
            df[loc_col]
            .astype(str)
            .str.replace(r"(?i)^land for sale in\s*", "", regex=True)
            .str.strip()
        )

        # 🔹 Standardize some known location variants
        replacements = {
            "Hawrat A'ali": "A'ali",
            "Hawrat Sanad": "Sanad",
            "Riffa - Al Hajiyat": "Riffa",
            "Riffa - Al Hunainiyah": "Riffa",
            "Riffa - Buhair": "Riffa",
            "Riffa Al Shamali": "Riffa",
            "West Riffa": "Riffa",
            "Sitra Industrail Area": "Sitra",
        }

        df[loc_col] = df[loc_col].replace(replacements)
    else:
        print("[WARN] 'location' column not found, skipping location cleanup.")

    # Size: remove " m²" and convert to numeric
    if "size" in colmap:
        size_col = colmap["size"]
        df[size_col] = (
            df[size_col]
            .astype(str)
            .str.replace(" m²", "", regex=False)
            .str.strip()
            .replace("", pd.NA)
        )
        df[size_col] = pd.to_numeric(df[size_col], errors="coerce")

    # Price per foot: remove "BHD. " and convert to numeric
    if "price per foot" in colmap:
        ppf_col = colmap["price per foot"]
        df[ppf_col] = (
            df[ppf_col]
            .astype(str)
            .str.replace("BHD. ", "", regex=False)
            .str.strip()
            .replace("", pd.NA)
        )
        df[ppf_col] = pd.to_numeric(df[ppf_col], errors="coerce")

    # Classification: remove "*" and drop "Undefined"
    deleted_undefined = 0
    if "classification" in colmap:
        cls_col = colmap["classification"]
        df[cls_col] = df[cls_col].astype(str).str.replace("*", "", regex=False).str.strip()
        before_cls = len(df)
        df = df[~df[cls_col].str.match(r"(?i)^undefined$", na=False)]
        deleted_undefined = before_cls - len(df)

    # ---------------------------------------------------------------------
    # 2) CREATE "calculated price" IF MISSING
    # ---------------------------------------------------------------------
    if "calculated price" not in colmap:
        if "size" in colmap and "price per foot" in colmap:
            size_col = colmap["size"]
            ppf_col = colmap["price per foot"]
            df["calculated price"] = df[size_col] * M2_TO_FT2 * df[ppf_col]
            colmap["calculated price"] = "calculated price"

    # ---------------------------------------------------------------------
    # 3) Delete rows where Location contains "Land For Rent"
    # ---------------------------------------------------------------------
    deleted_land_for_rent = 0
    if "location" in colmap:
        loc_col = colmap["location"]
        mask_land = df[loc_col].astype(str).str.contains("land for rent", case=False, na=False)
        deleted_land_for_rent = mask_land.sum()
        df = df[~mask_land]

    # ---------------------------------------------------------------------
    # 4) Delete rows with any missing value
    # ---------------------------------------------------------------------
    before_dropna = len(df)
    df = df.dropna(how="any")
    deleted_missing = before_dropna - len(df)

    # ---------------------------------------------------------------------
    # 5) Delete rows where Size < 20
    # ---------------------------------------------------------------------
    deleted_size = 0
    if "size" in colmap:
        size_col = colmap["size"]
        before_size = len(df)
        df = df[df[size_col] >= 20]
        deleted_size = before_size - len(df)

    # ---------------------------------------------------------------------
    # 6) Price: remove commas and make numeric
    # ---------------------------------------------------------------------
    if "price" in colmap:
        price_col = colmap["price"]
        df[price_col] = (
            df[price_col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .replace("", pd.NA)
        )
        df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    else:
        raise KeyError("Data must have a 'price' column.")

    # ---------------------------------------------------------------------
    # 7) Remove rows if |price - calculated price| > 5000
    # ---------------------------------------------------------------------
    deleted_price_diff = 0
    if "calculated price" in colmap:
        calc_col = colmap["calculated price"]
        price_col = colmap["price"]
        before_price_diff = len(df)
        diff = (df[price_col] - df[calc_col]).abs()
        df = df[diff <= 5000]
        deleted_price_diff = before_price_diff - len(df)

    # ---------------------------------------------------------------------
    # 8) Save cleaned data
    # ---------------------------------------------------------------------
    cleaned_path = output_dir / "cleaned_data.xlsx"
    df.to_excel(cleaned_path, index=False)

    # ---------------------------------------------------------------------
    # 9) Create model_ready.xlsx
    # ---------------------------------------------------------------------
    df_model = df.copy()
    for col_to_drop in ["price per foot", "calculated price"]:
        if col_to_drop in colmap and colmap[col_to_drop] in df_model.columns:
            df_model = df_model.drop(columns=[colmap[col_to_drop]])

    price_col = colmap["price"]
    df_model[price_col] = (df_model[price_col] / 1000.0).round(0).astype(int)

    model_ready_path = output_dir / "model_ready.xlsx"
    df_model.to_excel(model_ready_path, index=False)

    # ---------------------------------------------------------------------
    # 10) Print summary
    # ---------------------------------------------------------------------
    print_header("CLEANING SUMMARY")
    print(f"Original rows: {original_len}")
    print(f"1) 'Land For Rent' removed: {deleted_land_for_rent}")
    print(f"2) Rows with any missing value removed: {deleted_missing}")
    print(f"3) Rows with size < 20 removed: {deleted_size}")
    print(f"4) Rows with |price - calculated price| > 5000 removed: {deleted_price_diff}")
    print(f"5) Rows with classification 'Undefined' removed: {deleted_undefined}")
    print(f"[DONE] Cleaned data saved to: {cleaned_path}")
    print(f"[DONE] Model-ready data saved to: {model_ready_path}")
