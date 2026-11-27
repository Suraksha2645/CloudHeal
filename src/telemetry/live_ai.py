import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.telemetry.live_preprocessing import read_and_validate_telemetry


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TELEMETRY_FILE = PROJECT_ROOT / "data" / "live" / "telemetry.csv"
MODEL_CONFIG_FILE = PROJECT_ROOT / "model" / "lstm" / "lstm_config.json"
MODEL_FILE = PROJECT_ROOT / "model" / "lstm" / "lstm_autoencoder.pth"
MAX_TELEMETRY_AGE_SECONDS = 30


def _diagnosis_context(
    *,
    status,
    diagnosis_state,
    service=None,
    pod=None,
    suspected_fault=None,
    suspected_resource=None,
    severity="unavailable",
    relative_deviation=0.0,
    evidence=None,
    recommended_action,
):
    return {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "diagnosis_state": diagnosis_state,
        "affected_service": service,
        "affected_pod": pod,
        "suspected_fault": suspected_fault,
        "suspected_resource": suspected_resource,
        "severity": severity,
        "relative_deviation": round(relative_deviation, 3),
        "confidence": round(min(relative_deviation, 1.0), 3),
        "evidence": evidence or [],
        "recommended_action": recommended_action,
    }


def _severity_for_deviation(score):
    if score < 0.5:
        return "normal"
    if score < 1.0:
        return "suspected"
    if score < 2.0:
        return "high"
    return "critical"


def _load_model_config():
    try:
        with MODEL_CONFIG_FILE.open(encoding="utf-8-sig") as file:
            return json.load(file), None
    except (OSError, json.JSONDecodeError) as error:
        return {}, f"Could not load LSTM configuration: {error}"


def _read_live_rows():
    rows, summary = read_and_validate_telemetry(TELEMETRY_FILE)
    if summary["status"] == "missing":
        return [], "Live telemetry file does not exist."
    if summary["status"] == "empty":
        return [], "Live telemetry file is empty."
    if summary["status"] != "ok":
        return [], (
            "Live telemetry is unavailable: "
            f"{summary['status']}."
        )
    if not rows:
        return [], "No valid live telemetry records are available."
    return rows, None


def _service_name(service):
    aliases = {
        "redis-cart": "redis",
    }
    return aliases.get(service, service)


def _available_features(rows, feature_names):
    available = set()

    for row in rows:
        if row.get("namespace") != "default":
            continue

        service = _service_name(row.get("service", ""))
        if not service:
            continue

        for suffix, column in (
            ("cpu", "cpu_millicores"),
            ("mem", "memory_mib"),
        ):
            feature = f"{service}_{suffix}"
            if feature in feature_names and row.get(column):
                try:
                    value = float(row[column])
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(value):
                    continue
                available.add(feature)

    return sorted(available)


def _latest_service_rows(rows):
    latest = {}

    for row in rows:
        if row.get("namespace") != "default":
            continue

        service = row.get("service")
        if not service or service.startswith("cloudheal"):
            continue

        timestamp = _parse_timestamp(row.get("timestamp"))
        if timestamp is None:
            continue

        key = (
            row.get("namespace"),
            service,
            row.get("pod"),
            row.get("container"),
        )
        current = latest.get(key)
        if current is None or timestamp > current["_parsed_timestamp"]:
            record = dict(row)
            record["_parsed_timestamp"] = timestamp
            latest[key] = record

    return list(latest.values())


def _parse_timestamp(value):
    try:
        parsed = datetime.fromisoformat(
            str(value).strip().replace("Z", "+00:00")
        )
    except (TypeError, ValueError, OverflowError):
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _is_fresh(timestamp, now):
    age = (now - timestamp).total_seconds()
    return 0 <= age <= MAX_TELEMETRY_AGE_SECONDS


def _diagnose(rows, *, include_evidence=False):
    now = datetime.now(timezone.utc)
    fresh_rows = []
    for row in rows:
        timestamp = _parse_timestamp(row.get("timestamp"))
        if timestamp is not None and _is_fresh(timestamp, now):
            record = dict(row)
            record["_parsed_timestamp"] = timestamp
            fresh_rows.append(record)

    latest = _latest_service_rows(fresh_rows)
    if not latest:
        return _diagnosis_context(
            status="unavailable",
            diagnosis_state="insufficient_evidence",
            evidence=[],
            recommended_action="Collect live application telemetry first.",
        )

    candidates = []
    for current in latest:
        service = current["service"]
        try:
            current_cpu = float(current["cpu_millicores"])
            current_memory = float(current["memory_mib"])
        except (KeyError, TypeError, ValueError):
            continue

        history = [
            row
            for row in fresh_rows
            if row.get("namespace") == "default"
            and row.get("service") == service
            and row.get("pod") == current.get("pod")
            and row.get("container") == current.get("container")
            and row["_parsed_timestamp"] < current["_parsed_timestamp"]
        ][-6:-1]

        if len(history) < 3:
            continue

        try:
            baseline_cpu = sum(
                float(row["cpu_millicores"]) for row in history
            ) / len(history)
            baseline_memory = sum(
                float(row["memory_mib"]) for row in history
            ) / len(history)
        except (KeyError, TypeError, ValueError):
            continue
        if not (
            math.isfinite(current_cpu)
            and math.isfinite(current_memory)
            and math.isfinite(baseline_cpu)
            and math.isfinite(baseline_memory)
        ):
            continue

        cpu_delta = current_cpu - baseline_cpu
        memory_delta = current_memory - baseline_memory
        cpu_relative = max(
            cpu_delta / max(abs(baseline_cpu), 1.0),
            0.0,
        )
        memory_relative = max(
            memory_delta / max(abs(baseline_memory), 1.0),
            0.0,
        )
        score = max(cpu_relative, memory_relative)
        candidates.append(
            (
                score,
                service,
                cpu_delta,
                memory_delta,
                cpu_relative,
                memory_relative,
                current["_parsed_timestamp"],
                current,
            )
        )

    if not candidates:
        return _diagnosis_context(
            status="insufficient_evidence",
            diagnosis_state="insufficient_evidence",
            evidence=[
                "At least three previous samples for the same pod are "
                "required for a live CPU/memory comparison."
            ],
            recommended_action="Continue collecting telemetry.",
        )

    (
        score,
        service,
        cpu_delta,
        memory_delta,
        cpu_relative,
        memory_relative,
        _current_timestamp,
        current,
    ) = max(
        candidates,
        key=lambda candidate: (
            candidate[0],
            candidate[1],
            candidate[6],
            candidate[7].get("pod", ""),
            candidate[7].get("container", ""),
        ),
    )
    evidence = [
        (
            f"CPU is {current_cpu:.2f} millicores versus a "
            f"{baseline_cpu:.2f} millicore baseline "
            f"({cpu_relative * 100:.1f}% relative deviation)."
        ),
        (
            f"Memory is {current_memory:.2f} MiB versus a "
            f"{baseline_memory:.2f} MiB baseline "
            f"({memory_relative * 100:.1f}% relative deviation)."
        ),
    ]

    if score < 0.5:
        return _diagnosis_context(
            status="normal",
            diagnosis_state="healthy",
            service=service,
            pod=current.get("pod"),
            severity=_severity_for_deviation(score),
            relative_deviation=score,
            evidence=evidence,
            recommended_action="Continue monitoring; do not heal.",
        )

    fault = "CPU" if cpu_relative >= memory_relative else "MEM"
    result = _diagnosis_context(
        status="anomalous",
        diagnosis_state="suspected_anomaly",
        service=service,
        pod=current.get("pod"),
        suspected_fault=fault,
        suspected_resource="CPU" if fault == "CPU" else "memory",
        severity=_severity_for_deviation(score),
        relative_deviation=score,
        evidence=evidence,
        recommended_action=(
            "Inspect the affected workload before any guarded recovery action."
        ),
    )
    if include_evidence:
        result["_evidence_record"] = dict(current)
        result["_evidence_fault"] = fault
        result["_evidence_score"] = round(min(score, 1.0), 3)
    return result


def get_live_ai_status():
    config, config_error = _load_model_config()
    rows, telemetry_error = _read_live_rows()
    feature_names = config.get("features", [])
    available = _available_features(rows, feature_names)
    missing = [
        feature for feature in feature_names if feature not in available
    ]

    status = "blocked"
    status_message = (
        "The saved LSTM requires all configured features; live telemetry "
        "currently provides CPU and memory only."
    )
    if config_error:
        status = "error"
        status_message = config_error
    elif telemetry_error:
        status = "unavailable"
        status_message = telemetry_error
    elif not MODEL_FILE.exists():
        status = "error"
        status_message = f"Model artifact does not exist: {MODEL_FILE}"
    elif not missing:
        status = "ready"
        status_message = "All model features are available for live scoring."

    timestamp = datetime.now(timezone.utc).isoformat()
    return {
        "model": {
            "name": "LSTM Autoencoder",
            "status": status,
            "message": status_message,
            "artifact": str(MODEL_FILE.relative_to(PROJECT_ROOT)),
            "feature_count": config.get("input_features"),
            "sequence_length": config.get("sequence_length"),
            "threshold": config.get("threshold"),
            "available_feature_count": len(available),
            "missing_features": missing,
        },
        "prediction": {
            "status": "not_scored" if status != "ready" else "ready",
            "score": None,
            "is_anomaly": None,
            "timestamp": timestamp,
        },
        "diagnosis": _diagnose(rows),
        "telemetry": {
            "source": str(TELEMETRY_FILE.relative_to(PROJECT_ROOT)),
            "sample_count": len(rows),
            "timestamp": timestamp,
        },
    }
