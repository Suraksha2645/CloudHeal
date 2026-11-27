
from pathlib import Path
import pandas as pd

from preprocess_re1 import (
    RAW_OB_DIR,
    PROCESSED_OB_DIR,
    load_case,
    clean_case,
    summarize_case,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rcaeval_re1"
    / "manifest_re1_ob.csv"
)


def main():

    manifest = pd.read_csv(MANIFEST_PATH)

    # Select only clean Group L cases
    cases = manifest[
        (manifest["group"] == "L")
        & (manifest["is_clean"] == True)
    ]

    print("Clean Group L cases:", len(cases))

    processed = 0
    skipped = 0
    errors = 0

    for _, row in cases.iterrows():

        case_id = row["case_id"]

        case_dir = RAW_OB_DIR / case_id

        output_name = case_id.replace("/", "_") + ".csv"
        output_path = PROCESSED_OB_DIR / output_name

        # Never overwrite an existing processed file
        if output_path.exists():
            print("SKIPPED (already exists):", case_id)
            skipped += 1
            continue

        try:
            df_raw, inject_time = load_case(case_dir)

            df_clean = clean_case(df_raw)

            summary = summarize_case(df_clean)

            save_dir = output_path.parent
            save_dir.mkdir(parents=True, exist_ok=True)

            df_clean.to_csv(output_path, index=False)

            print(
                f"PROCESSED: {case_id} | "
                f"Rows: {summary['rows']} | "
                f"Columns: {summary['columns']}"
            )

            processed += 1

        except Exception as error:

            print(
                f"ERROR: {case_id} | "
                f"{type(error).__name__}: {error}"
            )

            errors += 1

    print("\n========== SUMMARY ==========")
    print("Eligible cases:", len(cases))
    print("Newly processed:", processed)
    print("Already existing:", skipped)
    print("Errors:", errors)


if __name__ == "__main__":
    main()
