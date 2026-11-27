"""dataset_group_l.py

Loads Group L (cpu/mem fault) cases from RE1-OB with row labels.
Raw data is only read. Nothing is saved to disk.

Label rule (documented from our measurements):
    time <  injection time          -> 0 NORMAL
    0 <= time - injection < 30 s    -> 1 TRANSITION (excluded from scoring)
    time - injection >= 30 s        -> 2 ANOMALY
"""

from collections import Counter

import pandas as pd

from preprocess_re1 import load_case, clean_case
from scan_missing_re1 import RAW_OB_DIR, PROJECT_ROOT

MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "processed" / "rcaeval_re1" / "manifest_re1_ob.csv"
)
TRANSITION_SECONDS = 30
NORMAL, TRANSITION, ANOMALY = 0, 1, 2
LABEL_NAMES = {NORMAL: "normal", TRANSITION: "transition", ANOMALY: "anomaly"}


def case_ids_for_split(split, group="L"):
    """Return the clean case IDs of one split (train/val/test) from the manifest."""
    manifest = pd.read_csv(MANIFEST_PATH)
    chosen = manifest[
        (manifest["group"] == group)
        & (manifest["is_clean"])
        & (manifest["split"] == split)
    ]
    return chosen["case_id"].tolist()


def case_dir(case_id):
    """'adservice_cpu/1' -> raw folder path of that case."""
    folder, repetition = case_id.split("/")
    return RAW_OB_DIR / folder / repetition


def shared_columns(case_ids):
    """Columns present in every listed case (excluding the time columns), sorted."""
    common = None
    for case_id in case_ids:
        header = pd.read_csv(case_dir(case_id) / "data.csv", nrows=0).columns
        columns = {c for c in header if c not in ("time", "time.1")}
        common = columns if common is None else common & columns
    return sorted(common)


def label_rows(times, inject_time):
    """Return an int8 label per row: NORMAL, TRANSITION or ANOMALY."""
    elapsed = times - inject_time
    labels = pd.Series(ANOMALY, index=times.index, dtype="int8")
    labels[elapsed < 0] = NORMAL
    labels[(elapsed >= 0) & (elapsed < TRANSITION_SECONDS)] = TRANSITION
    return labels


def load_labelled_case(case_id, columns):
    """Load one clean case: 'time', the chosen columns, and 'label'.

    Input:  case_id like 'adservice_cpu/1'; columns = list of column names.
    Output: (DataFrame, inject_time)
    """
    df_raw, inject_time = load_case(case_dir(case_id))
    df = clean_case(df_raw)
    out = df[columns].copy()
    out.insert(0, "time", df["time"].astype("int64"))
    out["label"] = label_rows(df["time"], inject_time)
    return out, inject_time


def main():
    splits = {name: case_ids_for_split(name) for name in ("train", "val", "test")}
    all_ids = [cid for ids in splits.values() for cid in ids]
    columns = shared_columns(all_ids)

    print("Cases per split:", {k: len(v) for k, v in splits.items()})
    print("Shared columns:", len(columns))
    by_type = Counter(c.split("_")[-1] for c in columns)
    print("By metric type:", dict(by_type))
    for kind in ("latency", "error"):
        print(f"{kind} columns:", [c for c in columns if c.endswith("_" + kind)])

    for name, ids in splits.items():
        counts = Counter()
        shapes = set()
        for case_id in ids:
            df, inject_time = load_labelled_case(case_id, columns)
            counts.update(df["label"].value_counts().to_dict())
            shapes.add(df.shape)
        named = {LABEL_NAMES[k]: counts[k] for k in sorted(counts)}
        print(f"\n{name}: label counts {named} | case table shapes {sorted(shapes)}")


if __name__ == "__main__":
    main()