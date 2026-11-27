"""
baseline_detector.py

Corrected statistical baseline for CloudHeal Group L.

Design:
1. Use the first 10 minutes of each case for calibration.
2. Do not evaluate the calibration rows.
3. Use robust median and MAD scaling.
4. Remove only the constant telemetry column.
5. Calculate a multivariate anomaly score.
6. Select the threshold using validation data only.
7. Search a wider range of thresholds.
8. Keep the test set untouched for final reporting.

Transition rows (0-29 seconds after injection) are excluded.
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
    NORMAL,
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

# Wider threshold search.
THRESHOLDS = [5, 8, 10, 12, 15, 20, 25, 30, 40, 50]

# We will compare:
# 1 = no smoothing
# 30 = rolling mean, kept only as a diagnostic comparison
SMOOTHING_WINDOWS = [1, 30]


# ---------------------------------------------------------
# Feature selection
# ---------------------------------------------------------

def get_metric_columns(all_columns):
    """
    Select useful telemetry columns.

    PassthroughCluster_error is constant at zero across
    Group L, so it carries no information and is removed.
    Other error metrics are retained.
    """

    excluded_columns = {
        "PassthroughCluster_error"
    }

    columns = [
        column
        for column in all_columns
        if column not in excluded_columns
    ]

    return columns


# ---------------------------------------------------------
# Per-case calibration
# ---------------------------------------------------------

def fit_case_baseline(df, columns):
    """
    Calculate robust baseline statistics from only the
    first 10 minutes of the current case.
    """

    calibration = df[
        (df["time"] - df["time"].iloc[0]) < CALIBRATION_SECONDS
    ].copy()

    medians = calibration[columns].median()

    mads = (
        calibration[columns] - medians
    ).abs().median()

    scales = mads * MAD_MULTIPLIER

    # Prevent very small scales from creating
    # unrealistically large anomaly scores.
    minimum_scales = medians.abs() * MIN_SCALE_FRACTION

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

def calculate_scores(df, columns, medians, scales):
    """
    Calculate the largest absolute robust z-score
    across telemetry columns for every row.
    """

    values = df[columns]

    z = (values - medians) / scales

    z = z.replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0.0)

    scores = z.abs().max(axis=1)

    return scores


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

def make_predictions(scores, threshold, window_seconds):
    """
    Convert anomaly scores into binary predictions.

    window_seconds=1:
        Direct thresholding.

    window_seconds=30:
        Diagnostic rolling-average comparison.
    """

    if window_seconds == 1:
        smoothed = scores

    else:
        smoothed = scores.rolling(
            window=window_seconds,
            min_periods=window_seconds,
        ).mean()

    predictions = (
        smoothed >= threshold
    ).astype(int)

    return predictions


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate_case(df, predictions):
    """
    Evaluate only rows after the calibration period.

    Transition rows are also excluded.
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

    scored_rows = int(mask.sum())

    return tp, tn, fp, fn, scored_rows


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

def metrics_from_counts(tp, tn, fp, fn):
    total = tp + tn + fp + fn

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
    threshold,
    window_seconds,
):
    """
    Run the detector over every case in one split.
    """

    case_ids = case_ids_for_split(split)

    total = {
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
        "scored_rows": 0,
    }

    cases_with_unusual_steps = []

    for case_id in case_ids:

        df, _ = load_labelled_case(
            case_id,
            columns,
        )

        steps = df["time"].diff().dropna()

        if not np.allclose(
            steps.to_numpy(),
            1.0,
        ):
            cases_with_unusual_steps.append(
                case_id
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
            threshold,
            window_seconds,
        )

        tp, tn, fp, fn, scored_rows = evaluate_case(
            df,
            predictions,
        )

        total["tp"] += tp
        total["tn"] += tn
        total["fp"] += fp
        total["fn"] += fn
        total["scored_rows"] += scored_rows

    return total, cases_with_unusual_steps


# ---------------------------------------------------------
# Print metrics
# ---------------------------------------------------------

def print_metrics(label, counts):

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
# Threshold selection
# ---------------------------------------------------------

def choose_threshold(
    columns,
    window_seconds,
):
    """
    Select threshold using validation data only.

    F1 is used because both missed anomalies and
    false alarms matter for self-healing.
    """

    best_threshold = None
    best_f1 = -1.0
    best_counts = None

    print()
    print(
        f"Threshold selection | "
        f"smoothing window = {window_seconds} s"
    )

    for threshold in THRESHOLDS:

        counts, _ = run_split(
            "val",
            columns,
            threshold,
            window_seconds,
        )

        metrics = metrics_from_counts(
            counts["tp"],
            counts["tn"],
            counts["fp"],
            counts["fn"],
        )

        print(
            f"  threshold={threshold:>2} | "
            f"precision={metrics['precision']:.4f} | "
            f"recall={metrics['recall']:.4f} | "
            f"f1={metrics['f1']:.4f}"
        )

        if metrics["f1"] > best_f1:

            best_f1 = metrics["f1"]
            best_threshold = threshold
            best_counts = counts

    print(
        f"Best validation threshold: "
        f"{best_threshold} "
        f"(F1={best_f1:.4f})"
    )

    return best_threshold, best_counts


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    train_ids = case_ids_for_split("train")

    all_ids = (
        train_ids
        + case_ids_for_split("val")
        + case_ids_for_split("test")
    )

    all_columns = shared_columns(all_ids)

    columns = get_metric_columns(
        all_columns
    )

    print(
        f"Training cases: {len(train_ids)}"
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

    print()

    # Check time spacing in training cases.
    _, unusual_steps = run_split(
        "train",
        columns,
        threshold=999999,
        window_seconds=1,
    )

    print(
        "Training cases with time steps "
        "other than 1 second:"
    )

    if unusual_steps:
        print(
            " ",
            unusual_steps,
        )
    else:
        print("  none")

    # Compare smoothing approaches.
    for window in SMOOTHING_WINDOWS:

        print()
        print(
            "=" * 10,
            f"smoothing window = {window} s",
            "=" * 10,
        )

        threshold, _ = choose_threshold(
            columns,
            window,
        )

        print()
        print("Validation result:")

        val_counts, _ = run_split(
            "val",
            columns,
            threshold,
            window,
        )

        print_metrics(
            "  validation",
            val_counts,
        )

        print()
        print("Test result:")

        test_counts, _ = run_split(
            "test",
            columns,
            threshold,
            window,
        )

        print_metrics(
            "  test",
            test_counts,
        )


if __name__ == "__main__":
    main()