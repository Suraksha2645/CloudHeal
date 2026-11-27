"""
baseline_detector.py

Baseline anomaly detector for CloudHeal Group L.

Approach:
1. Use the first 10 minutes of each case as its normal calibration period.
2. Calculate a robust median and MAD for every telemetry column.
3. Convert each row into a multivariate anomaly score.
4. Compare 1-second and 30-second smoothing.
5. Use validation cases to choose the alert threshold.
6. Evaluate the final threshold on the untouched test cases.

Transition rows (0-29 seconds after injection) are excluded from scoring.
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

CALIBRATION_SECONDS = 10 * 60
MAD_MULTIPLIER = 1.4826
MIN_SCALE_FRACTION = 0.05

# Candidate thresholds.
THRESHOLDS = [3, 4, 5, 6, 8, 10, 12]

# Persistence windows to compare.
SMOOTHING_WINDOWS = [1, 30]


def fit_case_baseline(df, columns):
    """Fit robust normal statistics using only the first 10 minutes."""

    calibration = df[
        (df["time"] - df["time"].iloc[0]) < CALIBRATION_SECONDS
    ]

    medians = calibration[columns].median()
    mads = (calibration[columns] - medians).abs().median()

    scales = mads * MAD_MULTIPLIER

    # Prevent extremely small scales from producing huge scores.
    minimum_scales = medians.abs() * MIN_SCALE_FRACTION
    scales = pd.concat([scales, minimum_scales], axis=1).max(axis=1)

    scales = scales.replace([np.inf, -np.inf], np.nan).fillna(1e-9)
    scales = scales.clip(lower=1e-9)

    return medians, scales


def calculate_scores(df, columns, medians, scales):
    """Calculate the largest absolute robust z-score per row."""

    values = df[columns]

    z = (values - medians) / scales
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    scores = z.abs().max(axis=1)

    return scores


def make_predictions(scores, threshold, window_seconds):
    """
    Convert anomaly scores into binary predictions.

    A row becomes anomalous when the rolling mean score remains
    above the threshold for the requested smoothing window.
    """

    if window_seconds == 1:
        smoothed = scores
    else:
        smoothed = scores.rolling(
            window=window_seconds,
            min_periods=window_seconds,
        ).mean()

    predictions = (smoothed >= threshold).astype(int)

    return predictions


def evaluate_case(df, predictions):
    """Return confusion-matrix counts for scored rows only."""

    # NORMAL and ANOMALY are scored.
    # TRANSITION is deliberately excluded.
    mask = df["label"] != TRANSITION

    actual = (df.loc[mask, "label"] == ANOMALY).astype(int)
    predicted = predictions.loc[mask]

    tp = int(((actual == 1) & (predicted == 1)).sum())
    tn = int(((actual == 0) & (predicted == 0)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())

    return tp, tn, fp, fn


def metrics_from_counts(tp, tn, fp, fn):
    """Calculate standard classification metrics."""

    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total else 0.0

    precision = tp / (tp + fp) if (tp + fp) else 0.0

    recall = tp / (tp + fn) if (tp + fn) else 0.0

    if precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def run_split(split, columns, threshold, window_seconds):
    """Run the baseline over every case in one split."""

    case_ids = case_ids_for_split(split)

    total = {
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
    }

    cases_with_steps = []

    for case_id in case_ids:
        df, _ = load_labelled_case(case_id, columns)

        steps = df["time"].diff().dropna()

        if not np.allclose(steps.to_numpy(), 1.0):
            cases_with_steps.append(case_id)

        medians, scales = fit_case_baseline(df, columns)

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

        tp, tn, fp, fn = evaluate_case(
            df,
            predictions,
        )

        total["tp"] += tp
        total["tn"] += tn
        total["fp"] += fp
        total["fn"] += fn

    return total, cases_with_steps


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
        f"FN={counts['fn']}"
    )


def choose_threshold(columns, window_seconds):
    """
    Choose the threshold using validation data only.

    F1 is used as the selection metric because both missed
    anomalies and false alarms matter for a self-healing system.
    """

    best_threshold = None
    best_f1 = -1.0
    best_counts = None

    print()
    print(f"Threshold selection | smoothing window = {window_seconds} s")

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
        f"Best validation threshold: {best_threshold} "
        f"(F1={best_f1:.4f})"
    )

    return best_threshold, best_counts


def main():
    train_ids = case_ids_for_split("train")
    all_ids = (
        train_ids
        + case_ids_for_split("val")
        + case_ids_for_split("test")
    )

    columns = shared_columns(all_ids)

    print(
        f"training cases: {len(train_ids)} | "
        "cases with time steps other than 1 s:"
    )

    # Check the training cases' time steps.
    _, unusual_steps = run_split(
        "train",
        columns,
        threshold=999999,
        window_seconds=1,
    )

    if unusual_steps:
        print(" ", unusual_steps)
    else:
        print("  none")

    for window in SMOOTHING_WINDOWS:
        print()
        print(f"========== smoothing window = {window} s ==========")

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
        print_metrics("  validation", val_counts)

        print()
        print("Final test result:")
        test_counts, _ = run_split(
            "test",
            columns,
            threshold,
            window,
        )
        print_metrics("  test", test_counts)


if __name__ == "__main__":
    main()
    