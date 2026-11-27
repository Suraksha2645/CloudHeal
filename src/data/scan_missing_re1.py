"""scan_missing_re1.py

Read-only scan of every RE1-OB case for missing values.
Loads ONE case at a time to keep RAM usage low.
Creates and modifies no files.
"""

from collections import Counter
from pathlib import Path

from preprocess_re1 import load_case, clean_case

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_OB_DIR = PROJECT_ROOT / "data" / "raw" / "rcaeval_re1" / "RE1" / "RE1" / "RE1-OB"


def list_case_dirs(raw_dir):
    """Return a sorted list of (case_name, folder) for every repetition.

    Input:  raw_dir (Path) - the RE1-OB folder.
    Output: list of ("adservice_cpu/1", Path) pairs.
    """
    case_dirs = []
    for group_dir in sorted(raw_dir.iterdir()):
        if not group_dir.is_dir():
            continue
        for rep_dir in sorted(group_dir.iterdir()):
            if rep_dir.is_dir():
                case_dirs.append((f"{group_dir.name}/{rep_dir.name}", rep_dir))
    return case_dirs


def scan_case(case_dir):
    """Describe the missing values in one case.

    Output: dict with rows, columns, injection_row and a 'missing_columns'
    dict (empty if the case has no missing values).
    """
    df_raw, inject_time = load_case(case_dir)
    df = clean_case(df_raw)  # also resets the index, so index == row number
    injection_row = int((df["time"] < inject_time).sum())

    info = {
        "rows": len(df),
        "columns": df.shape[1],
        "injection_row": injection_row,
        "missing_columns": {},
    }
    for col in df.columns[df.isna().any()]:
        missing_rows = df.index[df[col].isna()]
        valid = df[col].dropna()
        first_row = int(missing_rows.min())
        last_row = int(missing_rows.max())
        info["missing_columns"][col] = {
            "count": len(missing_rows),
            "first_row": first_row,
            "last_row": last_row,
            "one_block": (last_row - first_row + 1) == len(missing_rows),
            "ends_before_injection": last_row < injection_row,
            "rest_all_zero": len(valid) > 0 and bool((valid == 0).all()),
        }
    return info


def main():
    case_dirs = list_case_dirs(RAW_OB_DIR)
    print(f"Scanning {len(case_dirs)} cases (one at a time)...\n")

    shapes = Counter()
    missing_by_column = Counter()
    cases_with_missing = 0
    leading_zero_pattern = 0
    errors = 0

    for name, case_dir in case_dirs:
        try:
            info = scan_case(case_dir)
        except Exception as error:
            errors += 1
            print(f"ERROR {name}: {type(error).__name__}: {error}")
            continue

        shapes[(info["rows"], info["columns"])] += 1
        if not info["missing_columns"]:
            continue

        cases_with_missing += 1
        print(f"{name}  (rows={info['rows']}, injection_row={info['injection_row']})")
        for col, d in info["missing_columns"].items():
            missing_by_column[col] += 1
            print(
                f"    {col}: {d['count']} missing, rows {d['first_row']}-{d['last_row']}, "
                f"one_block={d['one_block']}, before_injection={d['ends_before_injection']}, "
                f"rest_all_zero={d['rest_all_zero']}"
            )
            if (d["first_row"] == 0 and d["one_block"]
                    and d["ends_before_injection"] and d["rest_all_zero"]):
                leading_zero_pattern += 1

    print("\n===== SUMMARY =====")
    print("Cases scanned:", len(case_dirs), "| errors:", errors)
    print("Cases with any missing values:", cases_with_missing)
    print("Case/column pairs matching 'leading block, before injection, rest all zero':",
          leading_zero_pattern)
    print("\nCases per (rows, columns) shape:")
    for shape, n in sorted(shapes.items()):
        print(f"    {shape}: {n}")
    print("\nColumns with missing values (number of cases affected):")
    for col, n in missing_by_column.most_common():
        print(f"    {col}: {n}")
    print("\nNo files were created or modified.")


if __name__ == "__main__":
    main()