
"""dataset_pipeline.py

Shared data pipeline for CloudHeal model experiments.

- Uses processed Group L CSV files.
- Selects features using training data only.
- Removes constant training features.
- Applies the same feature schema to train, validation and test.
- Preserves the existing NORMAL / TRANSITION / ANOMALY labels.
"""

import pandas as pd

from preprocess_re1 import PROCESSED_OB_DIR
from dataset_group_l import (
    case_ids_for_split,
    case_dir,
    label_rows,
    NORMAL,
    TRANSITION,
    ANOMALY,
)


def processed_path(case_id):
    """Get the processed CSV path for a case."""
    filename = case_id.replace("/", "_") + ".csv"
    return PROCESSED_OB_DIR / filename


def get_injection_time(case_id):
    """Read the original injection timestamp."""
    return int(
        (case_dir(case_id) / "inject_time.txt")
        .read_text()
        .strip()
    )


def select_training_features(train_ids):
    """Select common numeric, non-constant features using training data only."""

    common_columns = None

    # Find columns shared by every training case.
    for case_id in train_ids:

        path = processed_path(case_id)

        if not path.exists():
            raise FileNotFoundError(
                f"Processed file missing: {path}"
            )

        columns = set(
            pd.read_csv(path, nrows=0).columns
        )

        columns.discard("time")
        columns.discard("time.1")

        common_columns = (
            columns
            if common_columns is None
            else common_columns.intersection(columns)
        )

    common_columns = sorted(common_columns)

    # Check numeric columns using training data.
    numeric_columns = None

    for case_id in train_ids:

        df = pd.read_csv(processed_path(case_id))

        current_numeric = {
            column
            for column in common_columns
            if pd.api.types.is_numeric_dtype(df[column])
        }

        numeric_columns = (
            current_numeric
            if numeric_columns is None
            else numeric_columns.intersection(current_numeric)
        )

        del df

    numeric_columns = sorted(numeric_columns)

    # Find features that vary across the training data.
    global_min = pd.Series(
        float("inf"),
        index=numeric_columns
    )

    global_max = pd.Series(
        float("-inf"),
        index=numeric_columns
    )

    for case_id in train_ids:

        df = pd.read_csv(
            processed_path(case_id),
            usecols=numeric_columns
        )

        global_min = pd.concat(
            [global_min, df.min()]
        ).groupby(level=0).min()

        global_max = pd.concat(
            [global_max, df.max()]
        ).groupby(level=0).max()

        del df

    selected_features = [
        column
        for column in numeric_columns
        if (
            pd.notna(global_min[column])
            and pd.notna(global_max[column])
            and global_min[column] != global_max[column]
        )
    ]

    if not selected_features:
        raise ValueError(
            "No usable training features were found."
        )

    return selected_features


def load_case_with_features(case_id, features):
    """Load a processed case with the frozen feature schema and labels."""

    path = processed_path(case_id)

    if not path.exists():
        raise FileNotFoundError(
            f"Processed file missing: {path}"
        )

    df = pd.read_csv(path)

    missing = [
        column
        for column in features
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{case_id} is missing features: {missing}"
        )

    inject_time = get_injection_time(case_id)

    result = df[["time"] + features].copy()

    result["time"] = result["time"].astype("int64")

    result["label"] = label_rows(
        result["time"],
        inject_time
    )

    return result


def load_dataset():
    """Build the common train, validation and test datasets."""

    splits = {
        name: case_ids_for_split(name, group="L")
        for name in ("train", "val", "test")
    }

    if (
        len(splits["train"]) != 29
        or len(splits["val"]) != 8
        or len(splits["test"]) != 9
    ):
        raise ValueError(
            f"Unexpected split sizes: "
            f"{ {name: len(ids) for name, ids in splits.items()} }"
        )

    # Feature selection uses training cases only.
    features = select_training_features(
        splits["train"]
    )

    datasets = {}

    for split_name, case_ids in splits.items():

        datasets[split_name] = {}

        for case_id in case_ids:

            datasets[split_name][case_id] = (
                load_case_with_features(
                    case_id,
                    features
                )
            )

    return datasets, features


def main():

    print("Building shared CloudHeal dataset...")

    datasets, features = load_dataset()

    print("\nFeature count:", len(features))
    print("Feature selection: training data only")

    print("\nSelected features:")
    for feature in features:
        print(" -", feature)

    print("\nDataset summary:")

    for split_name, cases in datasets.items():

        normal = 0
        transition = 0
        anomaly = 0

        for df in cases.values():

            normal += int((df["label"] == NORMAL).sum())
            transition += int((df["label"] == TRANSITION).sum())
            anomaly += int((df["label"] == ANOMALY).sum())

        print(f"\n{split_name.upper()}")
        print("Cases:", len(cases))
        print("Total rows:", sum(len(df) for df in cases.values()))
        print("Normal rows:", normal)
        print("Transition rows:", transition)
        print("Anomaly rows:", anomaly)

    print("\nShared dataset pipeline completed successfully.")


if __name__ == "__main__":
    main()
