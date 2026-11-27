
import numpy as np
import pandas as pd

SEQUENCE_LENGTH = 30
LABEL_COLUMN = "label"
TIME_COLUMN = "time"


def prepare_case_sequences(
    df,
    feature_columns,
    sequence_length=SEQUENCE_LENGTH
):
    df = df.sort_values(TIME_COLUMN).reset_index(drop=True)

    values = df[feature_columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    values = values.replace([np.inf, -np.inf], np.nan)

    if values.isna().any().any():
        raise ValueError(
            "Missing or infinite feature values found. "
            "Handle them before sequence preparation."
        )

    data = values.to_numpy(dtype=np.float32)
    labels = df[LABEL_COLUMN].to_numpy(dtype=np.int64)

    sequences = []
    sequence_labels = []
    all_normal = []

    for end in range(sequence_length, len(data) + 1):
        start = end - sequence_length

        sequences.append(data[start:end])
        sequence_labels.append(labels[end - 1])

        # True only when every row in the sequence is NORMAL.
        all_normal.append(
            np.all(labels[start:end] == 0)
        )

    if not sequences:
        return (
            np.empty((0, sequence_length, len(feature_columns)),
                     dtype=np.float32),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=bool)
        )

    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(sequence_labels, dtype=np.int64),
        np.asarray(all_normal, dtype=bool)
    )


def prepare_split(case_data, feature_columns):
    all_sequences = []
    all_labels = []
    all_normal = []

    for case_id, df in case_data.items():
        sequences, labels, normal_mask = prepare_case_sequences(
            df,
            feature_columns
        )

        if len(sequences) == 0:
            continue

        all_sequences.append(sequences)
        all_labels.append(labels)
        all_normal.append(normal_mask)

    if not all_sequences:
        raise ValueError("No sequences were created.")

    return (
        np.concatenate(all_sequences),
        np.concatenate(all_labels),
        np.concatenate(all_normal)
    )
