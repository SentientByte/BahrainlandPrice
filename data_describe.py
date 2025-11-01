import pandas as pd
from pathlib import Path
from utils import get_project_paths, print_header

DEFAULT_THIN_THRESHOLD = 5
EXCLUDE_COLUMNS = {"size"}

TOP_N = 10
BOTTOM_N = 10


def _is_categorical(series: pd.Series) -> bool:
    return (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_categorical_dtype(series)
        or pd.api.types.is_bool_dtype(series)
    )


def _print_top_bottom_table(vc: pd.Series, top_n: int = TOP_N, bottom_n: int = BOTTOM_N):
    top_part = vc.head(top_n)
    bottom_part = vc.tail(bottom_n)

    top_rows = [(str(idx), int(cnt)) for idx, cnt in top_part.items()]
    bottom_rows = [(str(idx), int(cnt)) for idx, cnt in bottom_part.items()]

    max_rows = max(len(top_rows), len(bottom_rows))

    left_name_w = max([len(r[0]) for r in top_rows] + [len("Top values")])
    left_cnt_w = max([len(str(r[1])) for r in top_rows] + [len("Count")])
    right_name_w = max([len(r[0]) for r in bottom_rows] + [len("Bottom values")])
    right_cnt_w = max([len(str(r[1])) for r in bottom_rows] + [len("Count")])

    def line():
        return (
            "+" + "-" * (left_name_w + 2)
            + "+" + "-" * (left_cnt_w + 2)
            + "+" + "-" * (right_name_w + 2)
            + "+" + "-" * (right_cnt_w + 2)
            + "+"
        )

    print(line())
    print(
        "| "
        + "Top values".ljust(left_name_w)
        + " | "
        + "Count".ljust(left_cnt_w)
        + " | "
        + "Bottom values".ljust(right_name_w)
        + " | "
        + "Count".ljust(right_cnt_w)
        + " |"
    )
    print(line())

    for i in range(max_rows):
        if i < len(top_rows):
            t_name, t_cnt = top_rows[i]
        else:
            t_name, t_cnt = "", ""
        if i < len(bottom_rows):
            b_name, b_cnt = bottom_rows[i]
        else:
            b_name, b_cnt = "", ""
        print(
            "| "
            + t_name.ljust(left_name_w)
            + " | "
            + str(t_cnt).ljust(left_cnt_w)
            + " | "
            + b_name.ljust(right_name_w)
            + " | "
            + str(b_cnt).ljust(right_cnt_w)
            + " |"
        )

    print(line())


def describe_data():
    base_dir, venv_dir, data_dir, output_dir = get_project_paths()

    src_file = output_dir / "model_ready.xlsx"
    if not src_file.exists():
        raise FileNotFoundError(
            f"[ERROR] model_ready.xlsx not found in {output_dir}. "
            "Run 'Clean the data' first so it produces model_ready.xlsx."
        )

    df = pd.read_excel(src_file)
    print_header("DESCRIBE DATA (ON MODEL-READY DATA)")
    print(f"[INFO] Loaded model_ready data with shape: {df.shape}")

    # global default
    try:
        user_thr = input(f"Enter default thin-category threshold [default={DEFAULT_THIN_THRESHOLD}]: ").strip()
        if user_thr == "":
            global_thin_threshold = DEFAULT_THIN_THRESHOLD
        else:
            global_thin_threshold = int(user_thr)
    except Exception:
        global_thin_threshold = DEFAULT_THIN_THRESHOLD

    for col in df.columns:
        if col in EXCLUDE_COLUMNS:
            continue

        s = df[col]
        if not _is_categorical(s):
            continue

        print("\n" + "-" * 60)
        print(f"Column: {col}")
        vc = s.value_counts(dropna=False).sort_values(ascending=False)
        print(f"Unique values: {len(vc)}")

        # show ASCII top/bottom
        _print_top_bottom_table(vc)

        # per-column threshold
        try:
            col_thr_in = input(
                f"Thin-category threshold for '{col}' [default={global_thin_threshold}]: "
            ).strip()
            if col_thr_in == "":
                thin_threshold = global_thin_threshold
            else:
                thin_threshold = int(col_thr_in)
        except Exception:
            thin_threshold = global_thin_threshold

        thin_mask = vc <= thin_threshold
        thin_cats = vc[thin_mask]

        if len(thin_cats) == 0:
            print(f"[INFO] No thin categories (≤ {thin_threshold}) in '{col}'.")
            continue

        print(f"[INFO] Found {len(thin_cats)} thin categories (≤ {thin_threshold}) in '{col}':")
        for v, c in thin_cats.items():
            print(f"  {repr(v)} -> {c}")

        # 👉 calculate impact for R/G options
        thin_values = list(thin_cats.index)
        current_rows = len(df)
        rows_affected = df[col].isin(thin_values).sum()

        col_lower = col.lower()
        suggested_action = "G" if ("broker" in col_lower or "classif" in col_lower) else "S"

        print("\nWhat do you want to do with thin categories in this column?")
        print(
            f"  [R] Remove rows that contain these thin categories "
            f"(current rows: {current_rows}, will remove: {rows_affected})"
        )
        print(
            f"  [G] Group thin categories into 'Other' "
            f"(current rows: {current_rows}, will change: {rows_affected})"
        )
        print("  [S] Skip / do nothing")
        choice = input(f"Choose action for '{col}' [default={suggested_action}]: ").strip().upper()
        if choice == "":
            choice = suggested_action

        if choice == "R":
            before = len(df)
            df = df[~df[col].isin(thin_values)]
            after = len(df)
            print(f"[DONE] Removed {before - after} rows due to thin categories in '{col}'.")
        elif choice == "G":
            df[col] = df[col].where(~df[col].isin(thin_values), other="Other")
            print(f"[DONE] Grouped {rows_affected} rows into 'Other' in '{col}'.")
        else:
            print(f"[SKIP] No changes made to '{col}'.")

    # save
    print("\n" + "=" * 60)
    print("Finished processing all categorical columns.")
    print("How do you want to save the updated data?")
    print("  [O] Overwrite model_ready.xlsx")
    print("  [N] Save as model_ready_described.xlsx")
    save_choice = input("Choose [O/N, default=N]: ").strip().upper()
    if save_choice == "O":
        out_path = src_file
    else:
        out_path = output_dir / "model_ready_described.xlsx"

    df.to_excel(out_path, index=False)
    print(f"[DONE] Data saved to: {out_path}")
