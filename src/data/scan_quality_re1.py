"""scan_quality_re1.py

Read-only quality scan of every RE1-OB case.
Loads ONE case at a time. Creates and modifies no files.
Does not call clean_case(), because clean_case() refuses some cases.
"""

from collections import Counter

from preprocess_re1 import load_case
from scan_missing_re1 import list_case_dirs, RAW_OB_DIR


def describe_time_columns(df):
    """Describe 'time' and 'time.1'.

    Output: (blank_time_rows, time1_state). time1_state is one of:
      'absent'                - the case has no 'time.1' column
      'no-overlap'            - 'time' and 'time.1' are never both present
      'same'                  - one distinct offset, and it is 0
      'shifted(1 distinct)'   - one distinct offset, and it is not 0
      'variable(n distinct)'  - more than one distinct offset
    '+gaps' is appended if 'time.1' has blank values.
    """
    time_blank = int(df["time"].isna().sum())
    if "time.1" not in df.columns:
        return time_blank, "absent"

    both = df["time"].notna() & df["time.1"].notna()
    if not both.any():
        state = "no-overlap"
    else:
        offsets = sorted((df.loc[both, "time.1"] - df.loc[both, "time"]).unique())
        if len(offsets) == 1:
            # One distinct offset: 0 means identical timestamps.
            state = "same" if offsets[0] == 0 else "shifted(1 distinct)"
        else:
            state = f"variable({len(offsets)} distinct)"

    if df["time.1"].isna().any():
        state += "+gaps"
    return time_blank, state


def scan_case(case_dir):
    """Return a dictionary of quality measures for one case."""
    df, inject_time = load_case(case_dir)
    time_blank, time1_state = describe_time_columns(df)

    # Where 'time' is blank, borrow 'time.1' only to locate the injection row.
    if "time.1" in df.columns:
        best_time = df["time"].fillna(df["time.1"])
    else:
        best_time = df["time"]
    injection_row = int((best_time < inject_time).sum())

    data = df.drop(columns=[c for c in ("time", "time.1") if c in df.columns])
    missing_per_column = data.isna().sum()
    columns_missing = missing_per_column[missing_per_column > 0]
    complete_row = ~data.isna().any(axis=1)

    max_missing_pct = 0.0
    if len(columns_missing):
        max_missing_pct = round(100 * columns_missing.max() / len(df), 1)

    is_clean = (
        time_blank == 0
        and time1_state in ("absent", "same")
        and len(columns_missing) == 0
    )
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "injection_row": injection_row,
        "time_blank": time_blank,
        "time1_state": time1_state,
        "columns_with_missing": len(columns_missing),
        "max_missing_pct": max_missing_pct,
        "complete_before": int(complete_row.iloc[:injection_row].sum()),
        "complete_after": int(complete_row.iloc[injection_row:].sum()),
        "is_clean": is_clean,
    }


def main():
    case_dirs = list_case_dirs(RAW_OB_DIR)
    print(f"Scanning {len(case_dirs)} cases (one at a time)...\n")

    clean_count = 0
    errors = 0
    time1_states = Counter()
    shapes = Counter()

    for name, case_dir in case_dirs:
        try:
            r = scan_case(case_dir)
        except Exception as error:
            errors += 1
            print(f"ERROR {name}: {type(error).__name__}: {error}")
            continue

        shapes[(r["rows"], r["columns"])] += 1
        time1_states[r["time1_state"]] += 1
        if r["is_clean"]:
            clean_count += 1
            continue

        print(
            f"{name}: rows={r['rows']} cols={r['columns']} inj_row={r['injection_row']} "
            f"time_blank={r['time_blank']} time.1={r['time1_state']} "
            f"cols_missing={r['columns_with_missing']} max_missing={r['max_missing_pct']}% "
            f"complete_before={r['complete_before']} complete_after={r['complete_after']}"
        )

    print("\n===== SUMMARY =====")
    print("Cases scanned:", len(case_dirs), "| errors:", errors)
    print("Clean cases:", clean_count, "| not clean:", len(case_dirs) - clean_count - errors)
    print("\ntime.1 states:")
    for state, n in time1_states.most_common():
        print(f"    {state}: {n}")
    print("\nCases per (rows, columns) shape (raw):")
    for shape, n in sorted(shapes.items()):
        print(f"    {shape}: {n}")
    print("\nNo files were created or modified.")


if __name__ == "__main__":
    main()