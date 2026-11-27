"""
baseline_cv.py

Five-fold repetition-based evaluation of the locked
CloudHeal statistical baseline.

Locked detector:
- 10-minute per-case calibration
- 48 telemetry columns
- robust median/MAD scaling
- anomaly threshold = 20
- persistence = 5 seconds

Each repetition is evaluated as one fold.

No parameters are selected from these folds.
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
# Locked detector settings
# ---------------------------------------------------------

CALIBRATION_SECONDS = 10 * 60
MAD_MULTIPLIER = 1.4826
MIN_SCALE_FRACTION = 0.05

THRESHOLD = 20
PERSISTENCE_SECONDS = 5


# ---------------------------------------------------------
# Feature selection
# ---------------------------------------------------------

def get_metric_columns(all_columns):
    """Remove only the constant telemetry column."""

    excluded_columns = {
        "PassthroughCluster_error"
    }

    return [
        column
        for column in all_columns
        if column not in excluded_columns
    ]


# ---------------------------------------------------------
# Calibration
# ---------------------------------------------------------

def fit_case_baseline(df, columns):
    """Fit robust statistics using the first 10 minutes."""

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
    """Calculate the maximum absolute robust z-score."""

    values = df[columns]

    z = (
        values - medians
    ) / scales

    z = z.replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0.0)

    return z.abs().max(axis=1)


# ---------------------------------------------------------
# Persistence
# ---------------------------------------------------------

def make_predictions(scores):
    """
    Require the anomaly score to remain above the
    threshold for 5 consecutive seconds.
    """

    above_threshold = (
        scores >= THRESHOLD
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

        if consecutive >= PERSISTENCE_SECONDS:
            predictions.loc[index] = 1

    return predictions


# ---------------------------------------------------------
# Evaluate one case
# ---------------------------------------------------------

def evaluate_case(df, predictions):

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

    return tp, tn, fp, fn


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

def calculate_metrics(tp, tn, fp, fn):

    total = tp + tn + fp + fn

    accuracy = (
        (tp + tn) / total
        if total
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    return accuracy, precision, recall, f1


# ---------------------------------------------------------
# Get repetition number
# ---------------------------------------------------------

def get_repetition(case_id):
    """
    Extract repetition number from a case ID.

    Example:
        RE1-OB/adservice_cpu/3
        -> 3
    """

    return int(
        str(case_id).split("/")[-1]
    )


# ---------------------------------------------------------
# Evaluate one repetition
# ---------------------------------------------------------

def evaluate_repetition(
    case_ids,
    repetition,
    columns,
):

    selected_cases = [
        case_id
        for case_id in case_ids
        if get_repetition(case_id) == repetition
    ]

    total = {
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
        "cases": 0,
    }

    for case_id in selected_cases:

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
            scores
        )

        tp, tn, fp, fn = evaluate_case(
            df,
            predictions,
        )

        total["tp"] += tp
        total["tn"] += tn
        total["fp"] += fp
        total["fn"] += fn
        total["cases"] += 1

    return total


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    # Use all clean Group L cases from the existing
    # train/validation/test split definitions.
    all_case_ids = (
        case_ids_for_split("train")
        + case_ids_for_split("val")
        + case_ids_for_split("test")
    )

    all_columns = shared_columns(
        all_case_ids
    )

    columns = get_metric_columns(
        all_columns
    )

    print()
    print(
        "========== CloudHeal 5-Fold Repetition Evaluation =========="
    )
    print()

    print(
        f"Cases evaluated: {len(all_case_ids)}"
    )

    print(
        f"Telemetry columns: {len(columns)}"
    )

    print(
        "Excluded constant column: "
        "PassthroughCluster_error"
    )

    print(
        f"Threshold: {THRESHOLD}"
    )

    print(
        f"Persistence: {PERSISTENCE_SECONDS} seconds"
    )

    print()

    fold_results = []

    for repetition in range(1, 6):

        result = evaluate_repetition(
            all_case_ids,
            repetition,
            columns,
        )

        (
            accuracy,
            precision,
            recall,
            f1,
        ) = calculate_metrics(
            result["tp"],
            result["tn"],
            result["fp"],
            result["fn"],
        )

        fold_results.append(
            {
                "repetition": repetition,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": result["tp"],
                "tn": result["tn"],
                "fp": result["fp"],
                "fn": result["fn"],
                "cases": result["cases"],
            }
        )

        print(
            f"Fold {repetition} | "
            f"cases={result['cases']} | "
            f"accuracy={accuracy:.4f} | "
            f"precision={precision:.4f} | "
            f"recall={recall:.4f} | "
            f"f1={f1:.4f} | "
            f"FP={result['fp']} | "
            f"FN={result['fn']}"
        )

    # -----------------------------------------------------
    # Overall metrics
    # -----------------------------------------------------

    total_tp = sum(
        row["tp"] for row in fold_results
    )

    total_tn = sum(
        row["tn"] for row in fold_results
    )

    total_fp = sum(
        row["fp"] for row in fold_results
    )

    total_fn = sum(
        row["fn"] for row in fold_results
    )

    (
        accuracy,
        precision,
        recall,
        f1,
    ) = calculate_metrics(
        total_tp,
        total_tn,
        total_fp,
        total_fn,
    )

    print()
    print(
        "========== Overall =========="
    )

    print(
        f"accuracy={accuracy:.4f} | "
        f"precision={precision:.4f} | "
        f"recall={recall:.4f} | "
        f"f1={f1:.4f}"
    )

    print(
        f"TP={total_tp} | "
        f"TN={total_tn} | "
        f"FP={total_fp} | "
        f"FN={total_fn}"
    )

    # -----------------------------------------------------
    # Mean and standard deviation
    # -----------------------------------------------------

    fold_df = pd.DataFrame(
        fold_results
    )

    print()
    print(
        "========== Fold Mean ± Std =========="
    )

    for metric in [
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]:

        mean = fold_df[metric].mean()
        std = fold_df[metric].std(
            ddof=1
        )

        print(
            f"{metric}: "
            f"{mean:.4f} ± {std:.4f}"
        )


if __name__ == "__main__":
    main()