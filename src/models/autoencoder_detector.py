
from pathlib import Path
import sys

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import pandas as pd

from sklearn.neural_network import MLPRegressor

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from config import MANIFEST_FILE

from src.data.preprocess_re1 import (
    load_case,
    clean_case,
)


# ============================================================
# SETTINGS
# ============================================================

LOCKED_THRESHOLD = 2.948044

CALIBRATION_MINUTES = 10

HIDDEN_LAYERS = (32, 16, 32)

RANDOM_STATE = 42


# ============================================================
# DATASET LOCATION
# ============================================================

RE1_OB_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "rcaeval_re1"
    / "RE1"
    / "RE1"
    / "RE1-OB"
)


# ============================================================
# LOAD MANIFEST
# ============================================================

manifest = pd.read_csv(MANIFEST_FILE)

test_cases = manifest[
    manifest["split"].str.lower() == "test"
]["case_id"].tolist()

print("Test cases:", len(test_cases))
print("Cases:", test_cases)


# ============================================================
# FIND SHARED TELEMETRY COLUMNS
# ============================================================

def get_shared_columns(case_ids):

    column_sets = []

    for case_id in case_ids:

        case_dir = RE1_OB_DIR / case_id

        df_raw, _ = load_case(case_dir)

        df = clean_case(df_raw)

        column_sets.append(
            set(df.columns)
        )

    shared = set.intersection(
        *column_sets
    )

    excluded_columns = {
        "time",
        "time.1",
        "label",
        "PassthroughCluster_error",
    }

    telemetry_columns = sorted(
        shared - excluded_columns
    )

    return telemetry_columns


# ============================================================
# TEST TELEMETRY COLUMNS
# ============================================================

telemetry_columns = get_shared_columns(
    test_cases
)

print()
print(
    "Telemetry features:",
    len(telemetry_columns)
)

print(
    "Excluded columns: "
    "time, time.1, label, "
    "PassthroughCluster_error"
)


# ============================================================
# LOAD ONE LABELLED CASE
# ============================================================

def load_labelled_case(
    case_id,
    columns,
):

    case_dir = RE1_OB_DIR / case_id

    df_raw, inject_time = load_case(
        case_dir
    )

    df = clean_case(df_raw)

    # Keep only required telemetry columns
    X = df[columns].copy()

    # Convert values to numbers
    X = X.apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Remove infinite values
    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Fill missing values
    X = X.fillna(
        X.median()
    )

    X = X.fillna(0)

    return (
        df,
        X,
        inject_time,
    )


# ============================================================
# NORMALIZE ONE CASE
# ============================================================

def normalize_case(
    X,
    timestamps,
):

    calibration_end = (
        timestamps.iloc[0]
        + CALIBRATION_MINUTES * 60
    )

    calibration_mask = (
        timestamps <= calibration_end
    )

    calibration_data = X.loc[
        calibration_mask
    ]

    median = calibration_data.median()

    mad = (
        calibration_data - median
    ).abs().median()

    scale = np.maximum(
        mad * 1.4826,
        0.05 * median.abs(),
    )

    scale = scale.replace(
        0,
        1e-9,
    )

    X_normalized = (
        X - median
    ) / scale

    return X_normalized


# ============================================================
# TRAINING CASES
# ============================================================

train_cases = manifest[
    manifest["split"].str.lower() == "train"
]["case_id"].tolist()

print()
print(
    "Training cases:",
    len(train_cases)
)


# ============================================================
# FIND TRAINING TELEMETRY COLUMNS
# ============================================================

train_columns = get_shared_columns(
    train_cases
)


# ============================================================
# COLLECT NORMAL TRAINING DATA
# ============================================================

print()
print(
    "Training neural autoencoder "
    "on NORMAL training data..."
)

training_rows = []


for case_id in train_cases:

    case_dir = RE1_OB_DIR / case_id

    df_raw, inject_time = load_case(
        case_dir
    )

    df = clean_case(df_raw)

    X = df[
        train_columns
    ].copy()

    X = X.apply(
        pd.to_numeric,
        errors="coerce",
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    X = X.fillna(
        X.median()
    )

    X = X.fillna(0)

    timestamps = pd.to_numeric(
        df["time"]
    )

    X_normalized = normalize_case(
        X,
        timestamps,
    )

    # First 10 minutes are treated
    # as the normal calibration period.
    calibration_end = (
        timestamps.iloc[0]
        + CALIBRATION_MINUTES * 60
    )

    normal_mask = (
        timestamps <= calibration_end
    )

    normal_rows = X_normalized.loc[
        normal_mask
    ]

    training_rows.append(
        normal_rows
    )


# ============================================================
# COMBINE TRAINING DATA
# ============================================================

X_train = pd.concat(
    training_rows,
    ignore_index=True,
)

X_train = X_train.replace(
    [np.inf, -np.inf],
    np.nan,
)

X_train = X_train.fillna(0)


print(
    "Training rows:",
    len(X_train)
)

print(
    "Input features:",
    X_train.shape[1]
)


# ============================================================
# CREATE AUTOENCODER
# ============================================================

print()
print(
    "Creating neural autoencoder..."
)

print(
    "Hidden layers:",
    HIDDEN_LAYERS
)


autoencoder = MLPRegressor(
    hidden_layer_sizes=HIDDEN_LAYERS,
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=100,
    random_state=RANDOM_STATE,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
)


# ============================================================
# TRAIN AUTOENCODER
# ============================================================

print(
    "Training..."
)

autoencoder.fit(
    X_train,
    X_train,
)

print(
    "Training completed."
)

print(
    "Iterations:",
    autoencoder.n_iter_
)


# ============================================================
# TEST EVALUATION
# ============================================================

print()
print("=" * 60)
print("TEST EVALUATION")
print("=" * 60)


all_true = []

all_pred = []

total_test_rows = 0

total_scored_rows = 0


# ============================================================
# EVALUATE EACH TEST CASE
# ============================================================

for case_id in test_cases:

    print()
    print(
        "Testing case:",
        case_id
    )

    df, X, inject_time = (
        load_labelled_case(
            case_id,
            telemetry_columns,
        )
    )

    timestamps = pd.to_numeric(
        df["time"]
    )

    X_normalized = normalize_case(
        X,
        timestamps,
    )


    # --------------------------------------------------------
    # RECONSTRUCTION
    # --------------------------------------------------------

    reconstructed = (
        autoencoder.predict(
            X_normalized
        )
    )

    reconstruction_error = np.mean(
        (
            X_normalized.values
            - reconstructed
        ) ** 2,
        axis=1,
    )


    # --------------------------------------------------------
    # CREATE LABELS
    #
    # 0 = NORMAL
    # 1 = TRANSITION
    # 2 = ANOMALY
    # --------------------------------------------------------

    seconds_after_injection = (
        timestamps - inject_time
    )

    labels = np.zeros(
        len(df),
        dtype=int,
    )


    # Transition period:
    # 0 to 29 seconds after injection

    labels[
        (seconds_after_injection >= 0)
        & (seconds_after_injection < 30)
    ] = 1


    # Anomaly period:
    # 30 seconds or more after injection

    labels[
        seconds_after_injection >= 30
    ] = 2


    # --------------------------------------------------------
    # EXCLUDE TRANSITION ROWS
    # --------------------------------------------------------

    scored_mask = (
        labels != 1
    )


    y_true = (
        labels[scored_mask] == 2
    ).astype(int)


    # Locked threshold from validation
    y_pred = (
        reconstruction_error[
            scored_mask
        ] >= LOCKED_THRESHOLD
    ).astype(int)


    all_true.extend(
        y_true
    )

    all_pred.extend(
        y_pred
    )


    total_test_rows += len(df)

    total_scored_rows += len(
        y_true
    )


    print(
        "Rows:",
        len(df)
    )

    print(
        "Scored rows:",
        len(y_true)
    )


# ============================================================
# CONVERT RESULTS TO NUMPY ARRAYS
# ============================================================

all_true = np.array(
    all_true
)

all_pred = np.array(
    all_pred
)


# ============================================================
# CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    all_true,
    all_pred,
)

precision = precision_score(
    all_true,
    all_pred,
    zero_division=0,
)

recall = recall_score(
    all_true,
    all_pred,
    zero_division=0,
)

f1 = f1_score(
    all_true,
    all_pred,
    zero_division=0,
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

tn, fp, fn, tp = confusion_matrix(
    all_true,
    all_pred,
    labels=[0, 1],
).ravel()


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("=" * 60)
print("AUTOENCODER TEST RESULTS")
print("=" * 60)

print()

print(
    "Locked threshold:",
    LOCKED_THRESHOLD
)

print()

print(
    "Accuracy :",
    f"{accuracy:.4f}"
)

print(
    "Precision:",
    f"{precision:.4f}"
)

print(
    "Recall   :",
    f"{recall:.4f}"
)

print(
    "F1 Score :",
    f"{f1:.4f}"
)

print()

print(
    "TP:",
    tp
)

print(
    "TN:",
    tn
)

print(
    "FP:",
    fp
)

print(
    "FN:",
    fn
)

print()

print(
    "Total test rows  :",
    total_test_rows
)

print(
    "Scored test rows :",
    total_scored_rows
)

print()

print("=" * 60)
print("TEST EVALUATION COMPLETE")
print("=" * 60)
