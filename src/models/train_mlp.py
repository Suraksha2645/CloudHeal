
import sys
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
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
MODEL_DIR = PROJECT_ROOT / "models" / "mlp"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(DATA_DIR))
from dataset_pipeline import load_dataset

INPUT_SIZE = 47
HIDDEN_SIZE = 32
LATENT_SIZE = 16

BATCH_SIZE = 256
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
# 3. Load the shared dataset
# --------------------------------------------------

datasets, features = load_dataset()

INPUT_SIZE = len(features)

print("Selected features:", INPUT_SIZE)
print("Training cases:", len(datasets["train"]))
print("Validation cases:", len(datasets["val"]))
print("Test cases:", len(datasets["test"]))


def collect_rows(case_data, normal_only=False):
    values_list = []
    labels_list = []

    for case_id, df in case_data.items():

        df = df.sort_values("time").reset_index(drop=True)

        values = df[features].to_numpy(dtype=np.float32)
        labels = df["label"].to_numpy(dtype=np.int64)

        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite values found in {case_id}")

        if normal_only:
            mask = labels == 0
            values = values[mask]
            labels = labels[mask]

        values_list.append(values)
        labels_list.append(labels)

    return (
        np.concatenate(values_list, axis=0),
        np.concatenate(labels_list, axis=0)
    )


# --------------------------------------------------
# 4. Training-only normalization
# --------------------------------------------------

train_values, train_labels = collect_rows(
    datasets["train"],
    normal_only=True
)

means = train_values.mean(axis=0)
stds = train_values.std(axis=0)
stds = np.maximum(stds, 1e-6)

train_values = ((train_values - means) / stds).astype(np.float32)

print("Normal training rows:", len(train_values))


def prepare_split(case_data):
    values, labels = collect_rows(case_data)

    values = ((values - means) / stds).astype(np.float32)

    return values, labels


val_values, val_labels = prepare_split(datasets["val"])
test_values, test_labels = prepare_split(datasets["test"])

# --------------------------------------------------
# 5. Convert data into PyTorch loaders
# --------------------------------------------------

train_dataset = TensorDataset(
    torch.from_numpy(train_values)
)

val_normal_mask = val_labels == 0

val_normal_dataset = TensorDataset(
    torch.from_numpy(val_values[val_normal_mask])
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_normal_loader = DataLoader(
    val_normal_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# --------------------------------------------------
# 6. MLP Autoencoder
# --------------------------------------------------

class MLPAutoencoder(nn.Module):

    def __init__(self, input_size=INPUT_SIZE):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, LATENT_SIZE),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(LATENT_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, input_size)
        )

    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)


model = MLPAutoencoder().to(device)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# --------------------------------------------------
# 7. Training and early stopping
# --------------------------------------------------

best_val_loss = float("inf")
best_weights = None
patience_counter = 0
history = []

print("\nStarting MLP training...\n")

for epoch in range(EPOCHS):

    model.train()
    train_losses = []

    for (batch,) in train_loader:

        batch = batch.to(device)

        optimizer.zero_grad()

        reconstructed = model(batch)

        loss = criterion(reconstructed, batch)

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    model.eval()
    val_losses = []

    with torch.no_grad():

        for (batch,) in val_normal_loader:

            batch = batch.to(device)

            reconstructed = model(batch)

            loss = criterion(reconstructed, batch)

            val_losses.append(loss.item())

    train_loss = float(np.mean(train_losses))
    val_loss = float(np.mean(val_losses))

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "validation_loss": val_loss
    })

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} "
        f"- Train Loss: {train_loss:.6f} "
        f"- Validation Loss: {val_loss:.6f}"
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_weights = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

        patience_counter = 0

    else:

        patience_counter += 1

    if patience_counter >= PATIENCE:
        print("\nEarly stopping triggered.")
        break


if best_weights is None:
    raise RuntimeError("Training did not produce a valid model.")

model.load_state_dict(best_weights)

print("\nBest validation loss:", best_val_loss)

# --------------------------------------------------
# 8. Reconstruction error calculation
# --------------------------------------------------

def calculate_scores(values):

    model.eval()

    dataset = TensorDataset(torch.from_numpy(values))

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    scores = []

    with torch.no_grad():

        for (batch,) in loader:

            batch = batch.to(device)

            reconstructed = model(batch)

            errors = torch.mean(
                (batch - reconstructed) ** 2,
                dim=1
            )

            scores.extend(errors.cpu().numpy())

    return np.asarray(scores)


val_scores = calculate_scores(val_values)
test_scores = calculate_scores(test_values)

# --------------------------------------------------
# 9. Select threshold using validation data
# --------------------------------------------------

# Ignore transition rows (label 1).
val_mask = val_labels != 1

valid_val_labels = val_labels[val_mask]
valid_val_scores = val_scores[val_mask]

best_threshold = None
best_f1 = -1

for threshold in np.quantile(
    valid_val_scores,
    np.linspace(0, 1, 501)
):

    predictions = (valid_val_scores > threshold).astype(int)
    actual = (valid_val_labels == 2).astype(int)

    score = f1_score(
        actual,
        predictions,
        zero_division=0
    )

    if score > best_f1:
        best_f1 = score
        best_threshold = float(threshold)

print("Selected threshold:", best_threshold)
print("Validation F1:", best_f1)

# --------------------------------------------------
# 10. Final test evaluation
# --------------------------------------------------

test_mask = test_labels != 1

actual = (test_labels[test_mask] == 2).astype(int)

predictions = (
    test_scores[test_mask] > best_threshold
).astype(int)

accuracy = accuracy_score(actual, predictions)

precision = precision_score(
    actual,
    predictions,
    zero_division=0
)

recall = recall_score(
    actual,
    predictions,
    zero_division=0
)

f1 = f1_score(
    actual,
    predictions,
    zero_division=0
)

tn, fp, fn, tp = confusion_matrix(
    actual,
    predictions,
    labels=[0, 1]
).ravel()

print("\n========== MLP TEST RESULTS ==========")
print(f"Accuracy:  {accuracy:.6f}")
print(f"Precision: {precision:.6f}")
print(f"Recall:    {recall:.6f}")
print(f"F1 Score:  {f1:.6f}")

print("\nConfusion Matrix:")
print("TN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)

# --------------------------------------------------
# 11. Save model and experiment results
# --------------------------------------------------

torch.save(
    model.state_dict(),
    MODEL_DIR / "mlp_autoencoder.pth"
)

config = {
    "model": "MLP Autoencoder",
    "input_features": INPUT_SIZE,
    "features": features,
    "hidden_size": HIDDEN_SIZE,
    "latent_size": LATENT_SIZE,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "patience": PATIENCE,
    "random_seed": RANDOM_SEED,
    "normalization_means": means.tolist(),
    "normalization_stds": stds.tolist(),
    "threshold": best_threshold
}

with open(MODEL_DIR / "mlp_config.json", "w") as file:
    json.dump(config, file, indent=4)

pd.DataFrame(history).to_csv(
    MODEL_DIR / "mlp_training_history.csv",
    index=False
)

results = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "true_negative": int(tn),
    "false_positive": int(fp),
    "false_negative": int(fn),
    "true_positive": int(tp),
    "threshold": best_threshold,
    "validation_f1": float(best_f1),
    "best_validation_loss": float(best_val_loss)
}

with open(MODEL_DIR / "mlp_test_results.json", "w") as file:
    json.dump(results, file, indent=4)

print("\nMLP experiment completed.")
print("Saved files to:", MODEL_DIR)
