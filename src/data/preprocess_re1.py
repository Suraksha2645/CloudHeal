"""preprocess_re1.py

Cleans ONE RE1 Online Boutique case and saves a processed copy.
Raw data is only read, never modified.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# 1. PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# 2. DATASET PATHS
# ============================================================

RAW_OB_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "rcaeval_re1"
    / "RE1"
    / "RE1"
    / "RE1-OB"
)

PROCESSED_OB_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rcaeval_re1"
    / "RE1-OB"
)


# ============================================================
# 3. LOAD ONE CASE
# ============================================================

def load_case(case_dir):
    """Read data.csv and inject_time.txt from one case."""

    df = pd.read_csv(case_dir / "data.csv")

    inject_time = int(
        (case_dir / "inject_time.txt").read_text().strip()
    )

    return df, inject_time


# ============================================================
# 4. CLEAN ONE CASE
# ============================================================

def clean_case(df):
    """Remove duplicate time column and sort by time."""

    df = df.copy()

    if "time.1" in df.columns:

        # Check that time.1 is exactly the same as time
        if not (df["time"] == df["time.1"]).all():
            raise ValueError(
                "'time.1' differs from 'time'; not dropping it."
            )

        df = df.drop(columns=["time.1"])

    # Sort rows according to timestamp
    df = df.sort_values("time").reset_index(drop=True)

    return df


# ============================================================
# 5. VALIDATE PROCESSED CASE
# ============================================================

def summarize_case(df):
    """Run basic quality checks."""

    time_steps = (
        df["time"]
        .diff()
        .dropna()
        .unique()
    )

    return {
        "rows": len(df),

        "columns": df.shape[1],

        "missing_values": int(
            df.isna().sum().sum()
        ),

        "duplicate_timestamps": int(
            df["time"].duplicated().sum()
        ),

        "non_numeric_columns": [
            column
            for column in df.columns
            if not pd.api.types.is_numeric_dtype(
                df[column]
            )
        ],

        "time_steps_seconds": sorted(
            time_steps.tolist()
        ),
    }


# ============================================================
# 6. SAVE PROCESSED CASE
# ============================================================

def save_case(df, output_path):
    """Save cleaned data as CSV."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False
    )


# ============================================================
# 7. MAIN
# ============================================================

def main():

    # One case only for now
    case_dir = (
        RAW_OB_DIR
        / "adservice_cpu"
        / "1"
    )

    # Load raw data
    df_raw, inject_time = load_case(
        case_dir
    )

    # Clean data
    df_clean = clean_case(
        df_raw
    )

    # Display information
    print("===== PREPROCESSING RE1 CASE =====")

    print("\nRaw shape:")
    print(df_raw.shape)

    print("\nClean shape:")
    print(df_clean.shape)

    print("\nInjection time:")
    print(inject_time)

    print("\n===== VALIDATION =====")

    summary = summarize_case(
        df_clean
    )

    for name, value in summary.items():
        print(f"{name}: {value}")

    # Output path
    output_path = (
        PROCESSED_OB_DIR
        / "adservice_cpu_1.csv"
    )

    # Save processed data
    save_case(
        df_clean,
        output_path
    )

    print("\nSaved to:")
    print(output_path)


# ============================================================
# 8. RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()