from datetime import datetime, timedelta, timezone

from src.telemetry import live_ai
from src.api import main


def record(
    timestamp,
    cpu,
    memory=10.0,
    pod="api-1",
    container="server",
    service="frontend",
):
    return {
        "timestamp": timestamp.isoformat(),
        "namespace": "default",
        "service": service,
        "pod": pod,
        "container": container,
        "cpu_millicores": cpu,
        "memory_mib": memory,
    }


def test_diagnosis_uses_same_pod_and_container_history(monkeypatch):
    now = datetime.now(timezone.utc)
    rows = [
        record(now - timedelta(seconds=25 - index * 5), 10.0)
        for index in range(5)
    ]
    rows.append(record(now, 100.0))
    rows.append(
        record(now - timedelta(seconds=10), 1000.0, pod="api-old")
    )

    result = live_ai._diagnose(rows)

    assert result["status"] == "anomalous"
    assert result["affected_service"] == "frontend"
    assert result["affected_pod"] == "api-1"
    assert result["suspected_fault"] == "CPU"
    assert result["suspected_resource"] == "CPU"
    assert result["diagnosis_state"] == "suspected_anomaly"
    assert result["severity"] in {"suspected", "high", "critical"}
    assert result["relative_deviation"] > 0
    assert len(result["evidence"]) == 2
    assert result["updated_at"]


def test_replacement_pod_does_not_use_old_pod_baseline():
    now = datetime.now(timezone.utc)
    rows = [
        record(now - timedelta(seconds=20), 10.0, pod="api-old"),
        record(now - timedelta(seconds=15), 10.0, pod="api-old"),
        record(now - timedelta(seconds=10), 10.0, pod="api-old"),
        record(now - timedelta(seconds=5), 10.0, pod="api-old"),
        record(now, 100.0, pod="api-new"),
    ]

    result = live_ai._diagnose(rows)

    assert result["status"] == "insufficient_evidence"
    assert result["diagnosis_state"] == "insufficient_evidence"
    assert result["affected_pod"] is None


def test_stale_records_are_not_diagnosed():
    stale = datetime.now(timezone.utc) - timedelta(seconds=31)
    rows = [
        record(stale - timedelta(seconds=5 * index), 10.0)
        for index in range(5)
    ]

    result = live_ai._diagnose(rows)

    assert result["status"] == "unavailable"


def test_future_records_are_not_diagnosed():
    future = datetime.now(timezone.utc) + timedelta(seconds=31)
    rows = [record(future, 100.0)]

    result = live_ai._diagnose(rows)

    assert result["status"] == "unavailable"


def test_slightly_future_records_are_not_diagnosed():
    future = datetime.now(timezone.utc) + timedelta(seconds=1)
    rows = [record(future, 100.0)]

    result = live_ai._diagnose(rows)

    assert result["status"] == "unavailable"


def test_mixed_timezone_offsets_are_compared_chronologically():
    now = datetime.now(timezone.utc)
    rows = [
        record(
            (now - timedelta(seconds=20)).astimezone(
                timezone(timedelta(hours=5, minutes=30))
            ),
            10.0,
        ),
        record(
            (now - timedelta(seconds=15)).astimezone(timezone.utc),
            10.0,
        ),
        record(
            (now - timedelta(seconds=10)).astimezone(
                timezone(timedelta(hours=-4))
            ),
            10.0,
        ),
        record(
            (now - timedelta(seconds=5)).astimezone(timezone.utc),
            10.0,
        ),
        record(now, 100.0),
    ]

    result = live_ai._diagnose(rows)

    assert result["status"] == "anomalous"


def test_duplicate_rows_do_not_change_diagnosis():
    now = datetime.now(timezone.utc)
    rows = [
        record(now - timedelta(seconds=20), 10.0),
        record(now - timedelta(seconds=15), 10.0),
        record(now - timedelta(seconds=10), 10.0),
        record(now - timedelta(seconds=5), 10.0),
        record(now, 100.0),
    ]

    result = live_ai._diagnose(rows + [rows[-1]])

    assert result["status"] == "anomalous"


def test_direct_non_finite_input_is_ignored():
    now = datetime.now(timezone.utc)
    rows = [
        record(now - timedelta(seconds=20), 10.0),
        record(now - timedelta(seconds=15), 10.0),
        record(now - timedelta(seconds=10), 10.0),
        record(now - timedelta(seconds=5), 10.0),
        record(now, float("nan")),
    ]

    result = live_ai._diagnose(rows)

    assert result["status"] == "insufficient_evidence"


def test_read_live_rows_uses_validated_preprocessing(monkeypatch):
    sentinel = [{"timestamp": "valid"}]
    monkeypatch.setattr(
        live_ai,
        "read_and_validate_telemetry",
        lambda path: (
            sentinel,
            {"status": "ok"},
        ),
    )

    rows, error = live_ai._read_live_rows()

    assert rows is sentinel
    assert error is None


def test_invalid_preprocessing_status_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        live_ai,
        "read_and_validate_telemetry",
        lambda path: ([], {"status": "read_error"}),
    )

    rows, error = live_ai._read_live_rows()

    assert rows == []
    assert "read_error" in error


def test_empty_csv_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        live_ai,
        "read_and_validate_telemetry",
        lambda path: ([], {"status": "empty"}),
    )

    rows, error = live_ai._read_live_rows()

    assert rows == []
    assert error == "Live telemetry file is empty."


def test_unreadable_csv_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        live_ai,
        "read_and_validate_telemetry",
        lambda path: ([], {"status": "read_error"}),
    )

    rows, error = live_ai._read_live_rows()

    assert rows == []
    assert "read_error" in error


def test_memory_dominant_anomaly():
    now = datetime.now(timezone.utc)
    rows = [
        record(now - timedelta(seconds=20), 10.0, memory=10.0),
        record(now - timedelta(seconds=15), 10.0, memory=10.0),
        record(now - timedelta(seconds=10), 10.0, memory=10.0),
        record(now - timedelta(seconds=5), 10.0, memory=10.0),
        record(now, 10.0, memory=100.0),
    ]

    result = live_ai._diagnose(rows)

    assert result["status"] == "anomalous"
    assert result["suspected_fault"] == "MEM"
    assert result["suspected_resource"] == "memory"
    assert result["affected_pod"] == "api-1"
    assert result["severity"] in {"suspected", "high", "critical"}


def test_healthy_diagnosis_reports_pod_and_normal_severity():
    now = datetime.now(timezone.utc)
    rows = [
        record(now - timedelta(seconds=20), 10.0, memory=10.0),
        record(now - timedelta(seconds=15), 10.0, memory=10.0),
        record(now - timedelta(seconds=10), 10.0, memory=10.0),
        record(now - timedelta(seconds=5), 10.0, memory=10.0),
        record(now, 11.0, memory=10.5),
    ]

    result = live_ai._diagnose(rows)

    assert result["status"] == "normal"
    assert result["diagnosis_state"] == "healthy"
    assert result["affected_pod"] == "api-1"
    assert result["severity"] == "normal"
    assert result["relative_deviation"] < 0.5
    assert result["updated_at"]


def test_interleaved_containers_use_their_own_history():
    now = datetime.now(timezone.utc)
    rows = [
        record(now - timedelta(seconds=20), 10.0, container="server"),
        record(now - timedelta(seconds=19), 10.0, container="worker"),
        record(now - timedelta(seconds=15), 10.0, container="server"),
        record(now - timedelta(seconds=14), 10.0, container="worker"),
        record(now - timedelta(seconds=10), 10.0, container="server"),
        record(now - timedelta(seconds=9), 10.0, container="worker"),
        record(now - timedelta(seconds=5), 10.0, container="server"),
        record(now - timedelta(seconds=4), 10.0, container="worker"),
        record(now, 100.0, container="server"),
        record(now, 10.0, container="worker"),
    ]

    result = live_ai._diagnose(rows)

    assert result["status"] == "anomalous"
    assert result["suspected_fault"] == "CPU"


def test_equal_score_candidates_are_selected_deterministically():
    now = datetime.now(timezone.utc)
    rows = []
    for pod in ("api-a", "api-b"):
        for seconds in (20, 15, 10, 5):
            rows.append(record(
                now - timedelta(seconds=seconds),
                cpu=10.0,
                pod=pod,
            ))
        rows.append(record(now - timedelta(seconds=1), cpu=100.0, pod=pod))

    first = live_ai._diagnose(rows, include_evidence=True)
    second = live_ai._diagnose(rows, include_evidence=True)

    assert first["status"] == "anomalous"
    assert first["_evidence_record"]["pod"] == "api-b"
    assert second["_evidence_record"]["pod"] == "api-b"


def test_api_ai_status_response_compatibility(monkeypatch):
    monkeypatch.setattr(
        main,
        "get_live_ai_status",
        lambda: {
            "model": {
                "name": "LSTM Autoencoder",
                "status": "blocked",
                "message": "offline",
                "artifact": "model/lstm/lstm_autoencoder.pth",
                "feature_count": 47,
                "sequence_length": 30,
                "threshold": 1.0,
                "available_feature_count": 2,
                "missing_features": ["latency"],
            },
            "prediction": {
                "status": "not_scored",
                "score": None,
                "is_anomaly": None,
                "timestamp": "2026-11-05T10:00:00+00:00",
            },
            "diagnosis": {
                "status": "insufficient_evidence",
                "affected_service": None,
                "suspected_fault": None,
                "confidence": 0.0,
                "evidence": [],
                "recommended_action": "Continue collecting telemetry.",
            },
            "telemetry": {
                "source": "data/live/telemetry.csv",
                "sample_count": 0,
                "timestamp": "2026-11-05T10:00:00+00:00",
            },
        },
    )

    response = main.ai_status()

    assert set(response) == {"model", "prediction", "diagnosis", "telemetry"}
    assert response["model"]["status"] == "blocked"
    assert response["prediction"]["score"] is None
    assert response["diagnosis"]["status"] == "insufficient_evidence"
