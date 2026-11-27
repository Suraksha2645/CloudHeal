"""build_manifest_re1.py

Builds a manifest (one row per RE1-OB case) and saves it as a CSV.
Raw data is only read. The only file written is the manifest.
"""

import pandas as pd

from scan_missing_re1 import list_case_dirs, RAW_OB_DIR, PROJECT_ROOT
from scan_quality_re1 import scan_case

MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "processed" / "rcaeval_re1" / "manifest_re1_ob.csv"
)


def parse_case_name(name):
    """'adservice_cpu/1' -> ('adservice', 'cpu', 1)."""
    folder, repetition = name.split("/")
    service, fault = folder.rsplit("_", 1)
    return service, fault, int(repetition)


def group_for_fault(fault):
    """Group L (cpu, mem) and Group S (delay, disk, loss).

    Verified earlier: these fault types match the two column layouts exactly.
    """
    return "L" if fault in ("cpu", "mem") else "S"


def assign_split(group, is_clean, repetition):
    """Case-level split by repetition number (Group L only, clean cases only).

    Repetitions 1-3 -> train, 4 -> val, 5 -> test.
    Group S and problem cases stay 'unassigned' for now.
    """
    if group != "L" or not is_clean:
        return "unassigned"
    if repetition <= 3:
        return "train"
    if repetition == 4:
        return "val"
    return "test"


def main():
    rows = []
    for name, case_dir in list_case_dirs(RAW_OB_DIR):
        service, fault, repetition = parse_case_name(name)
        group = group_for_fault(fault)
        try:
            info = scan_case(case_dir)
        except Exception as error:
            print(f"ERROR {name}: {type(error).__name__}: {error}")
            continue
        rows.append({
            "case_id": name,
            "service": service,
            "fault": fault,
            "repetition": repetition,
            "group": group,
            "rows": info["rows"],
            "columns": info["columns"],
            "injection_row": info["injection_row"],
            "is_clean": info["is_clean"],
            "time1_state": info["time1_state"],
            "split": assign_split(group, info["is_clean"], repetition),
        })

    manifest = pd.DataFrame(rows)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST_PATH, index=False)

    print("Cases in manifest:", len(manifest))
    print("\nGroup x clean:")
    print(manifest.groupby(["group", "is_clean"]).size())
    print("\nGroup L split x fault:")
    print(manifest[manifest["group"] == "L"].groupby(["split", "fault"]).size())
    print("\nSaved to:", MANIFEST_PATH)


if __name__ == "__main__":
    main()