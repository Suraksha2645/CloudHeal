from pathlib import Path

from preprocess_re1 import load_case, clean_case, summarize_case


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_OB_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "rcaeval_re1"
    / "RE1"
    / "RE1"
    / "RE1-OB"
)


SAMPLE_CASES = [
    ("adservice", "cpu", 1),
    ("cartservice", "mem", 1),
    ("checkoutservice", "delay", 1),
    ("currencyservice", "loss", 1),
    ("productcatalogservice", "disk", 1),
]


def inspect_sample_case(service, fault_type, repetition):
    case_dir = RAW_OB_DIR / f"{service}_{fault_type}" / str(repetition)

    print("\n" + "=" * 60)
    print(f"CASE: {service}_{fault_type} / repetition {repetition}")
    print("=" * 60)

    if not case_dir.exists():
        print("STATUS: CASE NOT FOUND")
        print(f"Path: {case_dir}")
        return

    try:
        df_raw, inject_time = load_case(case_dir)
        df_clean = clean_case(df_raw)
        summary = summarize_case(df_clean)

        print(f"Raw shape: {df_raw.shape}")
        print(f"Clean shape: {df_clean.shape}")
        print(f"Injection time: {inject_time}")
        print(f"Rows: {summary['rows']}")
        print(f"Columns: {summary['columns']}")
        print(f"Missing values: {summary['missing_values']}")
        print(f"Duplicate timestamps: {summary['duplicate_timestamps']}")
        print(f"Non-numeric columns: {summary['non_numeric_columns']}")
        print(f"Time steps (seconds): {summary['time_steps_seconds']}")
        print("STATUS: OK")

    except Exception as error:
        print("STATUS: ERROR")
        print(f"{type(error).__name__}: {error}")


def main():
    print("===== RE1 DATASET SAMPLE SUMMARY =====")
    print(f"Dataset path:\n{RAW_OB_DIR}")
    print(f"\nCases to inspect: {len(SAMPLE_CASES)}")

    for service, fault_type, repetition in SAMPLE_CASES:
        inspect_sample_case(service, fault_type, repetition)

    print("\n" + "=" * 60)
    print("SAMPLE CHECK COMPLETE")
    print("=" * 60)
    print("No processed files were created.")
    print("No raw files were modified.")


if __name__ == "__main__":
    main()