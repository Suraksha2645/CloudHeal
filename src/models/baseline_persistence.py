"""
baseline_persistence.py

Persistence experiment for CloudHeal Group L.

This keeps the same preprocessing, labels, calibration,
feature selection, and scoring used by baseline_detector.py.

Only the prediction rule is changed:

Instead of triggering immediately when the anomaly score
crosses the threshold, the score must remain above the
threshold for N consecutive seconds.

Persistence values tested:
1, 5, 10, 20 seconds.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Allow imports from src/data
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
sys.path.insert(0, str(DATA_DIR))

from dataset_group_l import (  # noqa: E402
    ANOMALY,
    TRANSITION,
    case_ids_for_split,
    load_labelled_case,
    shared_columns,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

CALIBRATION_SECONDS = 10 * 60

MAD_MULTIPLIER = 1.4826
MIN_SCALE_FRACTION = 0.05

# Fixed threshold from the corrected baseline.
THRESHOLD = 20

# Persistence values to compare.
PERSISTENCE_VALUES = [1, 5, 10, 20]


# ---------------------------------------------------------
# Feature selection
# ---------------------------------------------------------

def get_metric_columns(all_columns):
    """
    Remove only the constant telemetry column.
    """

    excluded_columns = {
        "PassthroughCluster_error"
    }

    return [
        column
        for column in all_columns
        if column not in excluded_columns
    ]


# ---------------------------------------------------------
# Per-case calibration
# ---------------------------------------------------------

def fit_case_baseline(df, columns):
    """
    Use the first 10 minutes of the current case
    as the healthy calibration period.
    """

    time_from_start = (
        df["time"] - df["time"].iloc[0]
    )

    calibration = df[
        time_from_start < CALIBRATION_SECONDS
    ].copy()

    medians = calibration[columns].median()

    mads = (
        calibration[columns] - medians
    ).abs().median()

    scales = mads * MAD_MULTIPLIER

    minimum_scales = (
        medians.abs()
        * MIN_SCALE_FRACTION
    )

    scales = pd.concat(
        [scales, minimum_scales],
        axis=1,
    ).max(axis=1)

    scales = scales.replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(1e-9)

    scales = scales.clip(lower=1e-9)

    return medians, scales


# ---------------------------------------------------------
# Anomaly score
# ---------------------------------------------------------

def calculate_scores(
    df,
    columns,
    medians,
    scales,
):
    """
    Calculate the maximum absolute robust z-score
    across all telemetry columns.
    """

    values = df[columns]

    z = (
        values - medians
    ) / scales

    z = z.replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0.0)

    scores = z.abs().max(axis=1)

    return scores


# ---------------------------------------------------------
# Persistence prediction
# ---------------------------------------------------------

def make_predictions(
    scores,
    threshold,
    persistence_seconds,
):
    """
    Trigger an anomaly only when the anomaly score
    stays above the threshold for the required number
    of consecutive seconds.
    """

    above_threshold = (
        scores >= threshold
    )

    predictions = pd.Series(
        0,
        index=scores.index,
        dtype=int,
    )

    consecutive = 0

    for index, is_above in above_threshold.items():

        if is_above:
            consecutive += 1
        else:
            consecutive = 0

        if consecutive >= persistence_seconds:
            predictions.loc[index] = 1

    return predictions


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate_case(
    df,
    predictions,
):
    """
    Evaluate only rows after calibration.

    Transition rows are excluded.
    """

    time_from_start = (
        df["time"] - df["time"].iloc[0]
    )

    after_calibration = (
        time_from_start >= CALIBRATION_SECONDS
    )

    not_transition = (
        df["label"] != TRANSITION
    )

    mask = (
        after_calibration
        & not_transition
    )

    actual = (
        df.loc[mask, "label"] == ANOMALY
    ).astype(int)

    predicted = predictions.loc[mask]

    tp = int(
        ((actual == 1) & (predicted == 1)).sum()
    )

    tn = int(
        ((actual == 0) & (predicted == 0)).sum()
    )

    fp = int(
        ((actual == 0) & (predicted == 1)).sum()
    )

    fn = int(
        ((actual == 1) & (predicted == 0)).sum()
    )

    scored_rows = int(
        mask.sum()
    )

    return (
        tp,
        tn,
        fp,
        fn,
        scored_rows,
    )


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

def metrics_from_counts(
    tp,
    tn,
    fp,
    fn,
):

    total = (
        tp + tn + fp + fn
    )

    accuracy = (
        (tp + tn) / total
        if total
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp)
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn)
        else 0.0
    )

    if precision + recall:
        f1 = (
            2 * precision * recall
            / (precision + recall)
        )
    else:
        f1 = 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ---------------------------------------------------------
# Run one split
# ---------------------------------------------------------

def run_split(
    split,
    columns,
    persistence_seconds,
):

    case_ids = case_ids_for_split(
        split
    )

    total = {
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
        "scored_rows": 0,
    }

    for case_id in case_ids:

        df, _ = load_labelled_case(
            case_id,
            columns,
        )

        medians, scales = fit_case_baseline(
            df,
            columns,
        )

        scores = calculate_scores(
            df,
            columns,
            medians,
            scales,
        )

        predictions = make_predictions(
            scores,
            THRESHOLD,
            persistence_seconds,
        )

        (
            tp,
            tn,
            fp,
            fn,
            scored_rows,
        ) = evaluate_case(
            df,
            predictions,
        )

        total["tp"] += tp
        total["tn"] += tn
        total["fp"] += fp
        total["fn"] += fn
        total["scored_rows"] += scored_rows

    return total


# ---------------------------------------------------------
# Print metrics
# ---------------------------------------------------------

def print_metrics(
    label,
    counts,
):

    metrics = metrics_from_counts(
        counts["tp"],
        counts["tn"],
        counts["fp"],
        counts["fn"],
    )

    print(
        f"{label}: "
        f"accuracy={metrics['accuracy']:.4f} | "
        f"precision={metrics['precision']:.4f} | "
        f"recall={metrics['recall']:.4f} | "
        f"f1={metrics['f1']:.4f}"
    )

    print(
        f"  TP={counts['tp']} | "
        f"TN={counts['tn']} | "
        f"FP={counts['fp']} | "
        f"FN={counts['fn']} | "
        f"scored_rows={counts['scored_rows']}"
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    train_ids = case_ids_for_split(
        "train"
    )

    all_ids = (
        train_ids
        + case_ids_for_split("val")
        + case_ids_for_split("test")
    )

    all_columns = shared_columns(
        all_ids
    )

    columns = get_metric_columns(
        all_columns
    )

    print()
    print(
        "========== CloudHeal Persistence Experiment =========="
    )
    print()

    print(
        f"Training cases: "
        f"{len(train_ids)}"
    )

    print(
        f"Total telemetry columns: "
        f"{len(all_columns)}"
    )

    print(
        f"Columns used for scoring: "
        f"{len(columns)}"
    )

    print(
        "Excluded constant column: "
        "PassthroughCluster_error"
    )

    print(
        f"Threshold: {THRESHOLD}"
    )

    print()

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    print(
        "========== Validation =========="
    )

    validation_results = []

    for persistence in PERSISTENCE_VALUES:

        counts = run_split(
            "val",
            columns,
            persistence,
        )

        metrics = metrics_from_counts(
            counts["tp"],
            counts["tn"],
            counts["fp"],
            counts["fn"],
        )

        validation_results.append(
            {
                "persistence": persistence,
                "metrics": metrics,
                "counts": counts,
            }
        )

        print(
            f"persistence={persistence:>2} s | "
            f"precision={metrics['precision']:.4f} | "
            f"recall={metrics['recall']:.4f} | "
            f"f1={metrics['f1']:.4f} | "
            f"FP={counts['fp']} | "
            f"FN={counts['fn']}"
        )

    # -----------------------------------------------------
    # Select best persistence using validation F1
    # -----------------------------------------------------

    best = max(
        validation_results,
        key=lambda result:
        result["metrics"]["f1"],
    )

    best_persistence = (
        best["persistence"]
    )

    print()

    print(
        f"Best validation persistence: "
        f"{best_persistence} s "
        f"(F1="
        f"{best['metrics']['f1']:.4f})"
    )

    # -----------------------------------------------------
    # Test
    # -----------------------------------------------------

    print()
    print(
        "========== Test =========="
    )

    test_counts = run_split(
        "test",
        columns,
        best_persistence,
    )

    print_metrics(
        "test",
        test_counts,
    )


if __name__ == "__main__":
    main()