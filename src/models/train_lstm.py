
import sys
import json
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "src" / "data"
MODEL_DIR = PROJECT_ROOT / "models" / "lstm"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(DATA_DIR))

from dataset_pipeline import load_dataset

SEQUENCE_LENGTH = 30
HIDDEN_SIZE = 32
LATENT_SIZE = 16

BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.001
PATIENCE = 5

RANDOM_SEED = 42

# --------------------------------------------------
# 2. Reproducibility and device
# --------------------------------------------------

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

torch.set_num_threads(2)

device = torch.device("cpu")

print("Using device:", device)


# --------------------------------------------------
# 3. Dataset for lazy sequence loading
# --------------------------------------------------

class WindowDataset(Dataset):

    def __init__(
        self,
        case_data,
        features,
        means,
        stds,
        normal_only=False
    ):

        self.cases = []
        self.indices = []

        for case_id, df in case_data.items():

            df = df.sort_values("time").reset_index(drop=True)

            values = df[features].to_numpy(dtype=np.float32)
            labels = df["label"].to_numpy(dtype=np.int64)

            if not np.isfinite(values).all():
                raise ValueError(
                    f"Non-finite values found in {case_id}"
                )

            values = ((values - means) / stds).astype(np.float32)

            case_index = len(self.cases)

            self.cases.append((values, labels))

            for end in range(SEQUENCE_LENGTH, len(values) + 1):

                start = end - SEQUENCE_LENGTH

                window_labels = labels[start:end]
                final_label = int(labels[end - 1])

                if normal_only:

                    # Every row in a training window must be NORMAL.
                    if not np.all(window_labels == 0):
                        continue

                self.indices.append(
                    (case_index, start, end, final_label)
                )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):

        case_index, start, end, label = self.indices[index]

        values, _ = self.cases[case_index]

        sequence = values[start:end]

        return (
            torch.from_numpy(sequence.copy()),
            torch.tensor(label, dtype=torch.long)
        )


# --------------------------------------------------
# 4. LSTM Autoencoder
# --------------------------------------------------

class LSTMAutoencoder(nn.Module):

    def __init__(
        self,
        input_features,
        hidden_size=HIDDEN_SIZE,
        latent_size=LATENT_SIZE
    ):

        super().__init__()

        self.encoder = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.to_latent = nn.Linear(
            hidden_size,
            latent_size
        )

        self.from_latent = nn.Linear(
            latent_size,
            hidden_size
        )

        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.output_layer = nn.Linear(
            hidden_size,
            input_features
        )

    def forward(self, x):

        _, (hidden, _) = self.encoder(x)

        latent = self.to_latent(hidden[-1])

        expanded = self.from_latent(latent)

        repeated = expanded.unsqueeze(1).repeat(
            1,
            x.size(1),
            1
        )

        decoded, _ = self.decoder(repeated)

        return self.output_layer(decoded)


# --------------------------------------------------
# 5. Model evaluation helper
# --------------------------------------------------

def calculate_scores(model, loader):

    model.eval()

    scores = []
    labels = []

    with torch.no_grad():

        for sequences, batch_labels in loader:

            sequences = sequences.to(device)

            reconstructed = model(sequences)

            errors = torch.mean(
                (sequences - reconstructed) ** 2,
                dim=(1, 2)
            )

            scores.extend(errors.cpu().numpy())
            labels.extend(batch_labels.numpy())

    return np.asarray(scores), np.asarray(labels)


def normal_validation_loss(scores, labels):

    normal_scores = scores[labels == 0]

    if len(normal_scores) == 0:
        raise ValueError("No normal validation sequences found.")

    return float(np.mean(normal_scores))


# --------------------------------------------------
# 6. Select threshold using validation data
# --------------------------------------------------

def select_threshold(scores, labels):

    # Exclude TRANSITION rows from binary evaluation.
    valid = labels != 1

    scores = scores[valid]
    labels = labels[valid]

    binary_labels = (labels == 2).astype(int)

    if len(np.unique(binary_labels)) != 2:
        raise ValueError(
            "Validation must contain both NORMAL and ANOMALY sequences."
        )

    candidates = np.unique(
        np.quantile(scores, np.linspace(0.01, 0.99, 200))
    )

    best_threshold = None
    best_f1 = -1.0

    for threshold in candidates:

        predictions = (scores > threshold).astype(int)

        current_f1 = f1_score(
            binary_labels,
            predictions,
            zero_division=0
        )

        if current_f1 > best_f1:

            best_f1 = current_f1
            best_threshold = float(threshold)

    return best_threshold, best_f1


# --------------------------------------------------
# 7. Load CloudHeal dataset
# --------------------------------------------------

print("\nLoading CloudHeal dataset...")

datasets, features = load_dataset()

print("Selected features:", len(features))
print("Training cases:", len(datasets["train"]))
print("Validation cases:", len(datasets["val"]))
print("Test cases:", len(datasets["test"]))

if len(features) != 47:
    raise ValueError(
        f"Expected 47 features, received {len(features)}."
    )


# --------------------------------------------------
# 8. Fit normalization using training-normal rows
# --------------------------------------------------

print("\nFitting normalization parameters...")

normal_training_rows = []

for case_id, df in datasets["train"].items():

    values = df.loc[
        df["label"] == 0,
        features
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(values).all():
        raise ValueError(
            f"Invalid training values in {case_id}"
        )

    normal_training_rows.append(values)

normal_training_rows = np.concatenate(
    normal_training_rows,
    axis=0
)

means = normal_training_rows.mean(axis=0)

stds = normal_training_rows.std(axis=0)

# Avoid division by zero for constant or nearly constant features.
stds = np.maximum(stds, 1e-6)

print("Normalization fitted using training-normal rows.")
print("Training-normal rows:", len(normal_training_rows))


# --------------------------------------------------
# 9. Prepare datasets and data loaders
# --------------------------------------------------

print("\nPreparing sequence datasets...")

train_data = WindowDataset(
    datasets["train"],
    features,
    means,
    stds,
    normal_only=True
)

val_normal_data = WindowDataset(
    datasets["val"],
    features,
    means,
    stds,
    normal_only=True
)

val_all_data = WindowDataset(
    datasets["val"],
    features,
    means,
    stds,
    normal_only=False
)

test_data = WindowDataset(
    datasets["test"],
    features,
    means,
    stds,
    normal_only=False
)

print("Training sequences:", len(train_data))
print("Validation normal sequences:", len(val_normal_data))
print("Validation total sequences:", len(val_all_data))
print("Test total sequences:", len(test_data))

if len(train_data) == 0:
    raise ValueError("No normal training sequences found.")

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_normal_loader = DataLoader(
    val_normal_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

val_all_loader = DataLoader(
    val_all_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# --------------------------------------------------
# 10. Initialize model and optimizer
# --------------------------------------------------

model = LSTMAutoencoder(
    input_features=len(features)
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

criterion = nn.MSELoss()

best_val_loss = float("inf")
best_state = None
epochs_without_improvement = 0


# --------------------------------------------------
# 11. Training with validation-based early stopping
# --------------------------------------------------

print("\nStarting LSTM Autoencoder training...")

history = []

for epoch in range(EPOCHS):

    model.train()

    training_losses = []

    for sequences, _ in train_loader:

        sequences = sequences.to(device)

        optimizer.zero_grad()

        reconstructed = model(sequences)

        loss = criterion(reconstructed, sequences)

        loss.backward()

        optimizer.step()

        training_losses.append(loss.item())

    training_loss = float(np.mean(training_losses))

    # Evaluate reconstruction loss on fully normal
    # validation sequences only.
    val_scores, val_labels = calculate_scores(
        model,
        val_normal_loader
    )

    val_loss = normal_validation_loss(
        val_scores,
        val_labels
    )

    history.append({
        "epoch": epoch + 1,
        "training_loss": training_loss,
        "validation_loss": val_loss
    })

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {training_loss:.6f} | "
        f"Validation Loss: {val_loss:.6f}"
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_state = copy.deepcopy(
            model.state_dict()
        )

        epochs_without_improvement = 0

    else:

        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:

        print("Early stopping triggered.")
        break


if best_state is None:
    raise RuntimeError("Training did not produce a model.")

model.load_state_dict(best_state)

print("\nBest validation loss:", best_val_loss)


# --------------------------------------------------
# 12. Select anomaly threshold on validation data
# --------------------------------------------------

print("\nSelecting anomaly threshold...")

val_scores, val_labels = calculate_scores(
    model,
    val_all_loader
)

threshold, validation_f1 = select_threshold(
    val_scores,
    val_labels
)

print("Selected threshold:", threshold)
print("Validation F1:", validation_f1)


# --------------------------------------------------
# 13. Final test evaluation
# --------------------------------------------------

print("\nEvaluating on test data...")

test_scores, test_labels = calculate_scores(
    model,
    test_loader
)

# Exclude TRANSITION from binary classification.
valid_test = test_labels != 1

binary_test_labels = (
    test_labels[valid_test] == 2
).astype(int)

test_predictions = (
    test_scores[valid_test] > threshold
).astype(int)

test_accuracy = accuracy_score(
    binary_test_labels,
    test_predictions
)

test_precision = precision_score(
    binary_test_labels,
    test_predictions,
    zero_division=0
)

test_recall = recall_score(
    binary_test_labels,
    test_predictions,
    zero_division=0
)

test_f1 = f1_score(
    binary_test_labels,
    test_predictions,
    zero_division=0
)

cm = confusion_matrix(
    binary_test_labels,
    test_predictions,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

print("\nLSTM Test Results")
print("------------------------------")
print(f"Accuracy : {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall   : {test_recall:.4f}")
print(f"F1-score : {test_f1:.4f}")
print(f"TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}")


# --------------------------------------------------
# 14. Save model and experiment results
# --------------------------------------------------

model_path = MODEL_DIR / "lstm_autoencoder.pth"
config_path = MODEL_DIR / "lstm_config.json"
history_path = MODEL_DIR / "lstm_training_history.csv"
results_path = MODEL_DIR / "lstm_test_results.json"

torch.save(
    model.state_dict(),
    model_path
)

config = {
    "features": features,
    "input_features": len(features),
    "sequence_length": SEQUENCE_LENGTH,
    "hidden_size": HIDDEN_SIZE,
    "latent_size": LATENT_SIZE,
    "threshold": threshold,
    "normalization_mean": means.tolist(),
    "normalization_std": stds.tolist(),
    "best_validation_loss": best_val_loss,
    "validation_f1": validation_f1,
    "epochs_completed": len(history),
    "random_seed": RANDOM_SEED
}

with open(config_path, "w") as file:
    json.dump(config, file, indent=4)

pd.DataFrame(history).to_csv(
    history_path,
    index=False
)

results = {
    "accuracy": float(test_accuracy),
    "precision": float(test_precision),
    "recall": float(test_recall),
    "f1_score": float(test_f1),
    "true_negative": int(tn),
    "false_positive": int(fp),
    "false_negative": int(fn),
    "true_positive": int(tp),
    "threshold": float(threshold)
}

with open(results_path, "w") as file:
    json.dump(results, file, indent=4)

print("\nSaved files:")
print(model_path)
print(config_path)
print(history_path)
print(results_path)

print("\nLSTM training and evaluation completed.")


