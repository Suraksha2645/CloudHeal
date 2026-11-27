"""Read-only access to the saved offline LSTM evaluation artifacts."""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LSTM_DIR = PROJECT_ROOT / "model" / "lstm"
CONFIG_PATH = LSTM_DIR / "lstm_config.json"
RESULTS_PATH = LSTM_DIR / "lstm_test_results.json"
ARTIFACT_PATH = LSTM_DIR / "lstm_autoencoder.pth"


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Offline LSTM file is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Offline LSTM file must contain an object: {path}")
    return value


def get_offline_evaluation() -> dict:
    """Return verified saved model metadata and test metrics.

    This function never reads live telemetry and never performs inference.
    """
    config = _read_json(CONFIG_PATH)
    results = _read_json(RESULTS_PATH)

    required_config = (
        "input_features",
        "sequence_length",
        "hidden_size",
        "latent_size",
        "threshold",
    )
    required_results = (
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
        "threshold",
    )
    missing_config = [key for key in required_config if key not in config]
    missing_results = [key for key in required_results if key not in results]
    if missing_config or missing_results:
        raise ValueError(
            "Offline LSTM artifacts are incomplete: "
            f"config={missing_config}, results={missing_results}"
        )

    return {
        "status": "available",
        "mode": "offline_evaluation",
        "live_inference": False,
        "inference_reason": (
            "Live telemetry provides CPU and memory only; "
            "the trained model requires 47 features."
        ),
        "artifact": str(ARTIFACT_PATH.relative_to(PROJECT_ROOT)),
        "artifact_available": ARTIFACT_PATH.is_file(),
        "features": config["input_features"],
        "sequence_length": config["sequence_length"],
        "hidden_size": config["hidden_size"],
        "latent_size": config["latent_size"],
        "threshold": results["threshold"],
        "metrics": {
            "accuracy": results["accuracy"],
            "precision": results["precision"],
            "recall": results["recall"],
            "f1_score": results["f1_score"],
            "true_negative": results["true_negative"],
            "false_positive": results["false_positive"],
            "false_negative": results["false_negative"],
            "true_positive": results["true_positive"],
        },
    }
