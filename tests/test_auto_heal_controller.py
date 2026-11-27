from datetime import datetime, timedelta, timezone
import json
import math

import pytest

from src.telemetry import auto_heal_controller as controller
from src.remediation import remediation


@pytest.fixture(autouse=True)
def reset_controller_state(monkeypatch, tmp_path):
    controller.baselines.clear()
    controller.current_pods.clear()
    controller.last_processed.clear()
    controller.consecutive_anomalies.clear()
    controller.last_restart.clear()
    controller.persistence_trusted = True
    controller.persistence_error = None
    controller.persistence_initialized = False
    monkeypatch.setattr(controller, "STATE_FILE", tmp_path / "state.json")


def telemetry(cpu, timestamp=None, pod="api-1", service="frontend"):
    timestamp = timestamp or datetime.now(timezone.utc)
    return {
        "timestamp": timestamp.isoformat(),
        "_parsed_timestamp": timestamp,
        "namespace": "default",
        "service": service,
        "pod": pod,
        "container": "server",
        "cpu_millicores": cpu,
        "memory_mib": 10.0,
    }


def add_baseline(service="frontend", pod="api-1"):
    now = datetime.now(timezone.utc)
    for index in range(controller.BASELINE_SAMPLES):
        controller.check_service(
            service,
            telemetry(
                10.0,
                now + timedelta(seconds=-(5 - index)),
                pod,
                service,
            ),
        )


def test_normal_telemetry_does_not_trigger_recovery(monkeypatch):
    monkeypatch.setattr(
        controller,
        "restart_service",
        lambda service: pytest.fail(),
    )
    add_baseline()

    result = controller.check_service("frontend", telemetry(10.0))

    assert result["status"] == "NORMAL"


def test_insufficient_baseline_samples_do_not_trigger_recovery():
    result = controller.check_service("frontend", telemetry(100.0))

    assert result["status"] == "BASELINE"
    assert controller.consecutive_anomalies["frontend"] == 0


def test_three_consecutive_anomalies_trigger_eligible_recovery(monkeypatch):
    calls = []
    monkeypatch.setattr(
        controller,
        "restart_service",
        lambda service, persistence_verified=False: calls.append(service)
        or {"service": service, "status": "SUCCESS", "message": "ok", "error": None},
    )
    add_baseline()

    for _ in range(controller.REQUIRED_ANOMALIES):
        result = controller.check_service("frontend", telemetry(100.0))

    assert calls == ["frontend"]
    assert result["status"] == "SUCCESS"


def test_unapproved_services_are_rejected():
    result = controller.handle_remediation("unknown")

    assert result["status"] == "REJECTED"
    assert result["error"] == "unapproved_service"


def test_stale_telemetry_is_rejected():
    result = controller.check_service(
        "frontend",
        telemetry(
            10.0,
            datetime.now(timezone.utc)
            - timedelta(seconds=controller.MAX_SAMPLE_AGE_SECONDS + 1),
        ),
    )

    assert result["status"] == "REJECTED"
    assert result["error"] == "stale_telemetry"


def test_duplicate_records_are_ignored():
    record = telemetry(10.0)
    controller.check_service("frontend", record)

    result = controller.check_service("frontend", record)

    assert result["status"] == "IGNORED"
    assert result["error"] == "duplicate_record"


def test_pod_change_resets_baseline():
    add_baseline()
    result = controller.check_service("frontend", telemetry(100.0, pod="api-2"))

    assert result["status"] == "BASELINE"
    assert controller.baselines["frontend"] == [100.0]
    assert controller.consecutive_anomalies["frontend"] == 0


def test_cooldown_prevents_repeated_recovery(monkeypatch):
    monkeypatch.setattr(
        controller,
        "restart_service",
        lambda service: {"service": service, "status": "FAILED", "message": "down", "error": "down"},
    )
    add_baseline()
    controller.last_restart["frontend"] = datetime.now(timezone.utc).timestamp()

    result = controller.handle_remediation("frontend")

    assert result["status"] == "REJECTED"
    assert result["error"] == "cooldown_active"


def test_restart_history_is_saved_and_loaded_for_cooldown(monkeypatch):
    controller.last_restart["frontend"] = datetime.now(timezone.utc).timestamp()
    controller.save_state()
    controller.last_restart.clear()

    controller.load_state()
    monkeypatch.setattr(
        controller,
        "restart_service",
        lambda service: pytest.fail("cooldown should prevent remediation"),
    )

    result = controller.handle_remediation("frontend")

    assert result["status"] == "REJECTED"
    assert result["error"] == "cooldown_active"


def test_failed_save_blocks_remediation_and_preserves_history(monkeypatch):
    original = {"frontend": 123.0}
    controller.last_restart.update(original)
    monkeypatch.setattr(controller, "save_state", lambda history: False)
    monkeypatch.setattr(
        controller,
        "restart_service",
        lambda service, persistence_verified=False: pytest.fail(),
    )

    result = controller.handle_remediation("frontend")

    assert result["status"] == "FAILED"
    assert result["error"] == "persistence_failed"
    assert controller.last_restart == original

def test_save_failure_marks_persistence_untrusted(monkeypatch):
    monkeypatch.setattr(
        controller.json,
        "dump",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            PermissionError("permission denied")
        ),
    )

    assert controller.save_state({"frontend": 123.0}) is False
    assert controller.persistence_trusted is False


def test_missing_state_after_initial_load_is_untrusted():
    controller.load_state()
    controller.last_restart["frontend"] = 123.0
    controller.STATE_FILE.unlink(missing_ok=True)

    controller.load_state()

    assert controller.persistence_trusted is False
    assert controller.last_restart == {"frontend": 123.0}


@pytest.mark.parametrize(
    "state",
    [
        "[]",
        "null",
        '"state"',
        "123",
        json.dumps({"last_restart": "invalid"}),
        json.dumps(
            {
                "last_restart": {
                    "frontend": "not-a-number",
                    "cartservice": math.nan,
                    "checkoutservice": math.inf,
                    "currencyservice": -1,
                }
            }
        ),
    ],
)
def test_structurally_invalid_restart_history_is_untrusted(state):
    controller.STATE_FILE.write_text(state, encoding="utf-8")

    controller.load_state()

    assert controller.persistence_trusted is False
    assert controller.persistence_error


def test_corrupted_state_preserves_existing_in_memory_history():
    controller.last_restart["frontend"] = datetime.now(timezone.utc).timestamp()
    controller.STATE_FILE.write_text("{not-json", encoding="utf-8")

    controller.load_state()

    assert "frontend" in controller.last_restart
    assert controller.persistence_trusted is False


def test_untrusted_persistence_blocks_remediation(monkeypatch):
    controller.STATE_FILE.write_text("[]", encoding="utf-8")
    controller.load_state()
    monkeypatch.setattr(
        controller,
        "restart_service",
        lambda service: pytest.fail("untrusted persistence must block remediation"),
    )

    result = controller.handle_remediation("frontend")

    assert result["status"] == "REJECTED"
    assert result["error"] == "persistence_untrusted"


def test_missing_state_file_allows_safe_initialisation():
    controller.load_state()

    assert controller.persistence_trusted is True
    assert controller.persistence_error is None
    assert controller.last_restart == {}


def test_missing_last_restart_field_is_untrusted():
    controller.STATE_FILE.write_text("{}", encoding="utf-8")

    controller.load_state()

    assert controller.persistence_trusted is False
    assert controller.persistence_error


def test_invalid_service_key_is_untrusted():
    controller.STATE_FILE.write_text(
        json.dumps({"last_restart": {"not-a-service": 123.0}}),
        encoding="utf-8",
    )

    controller.load_state()

    assert controller.persistence_trusted is False


def test_malformed_telemetry_does_not_crash_controller():
    result = controller.check_service(
        "frontend",
        {
            "pod": "api-1",
            "_parsed_timestamp": datetime.now(timezone.utc),
            "cpu_millicores": "not-a-number",
        },
    )

    assert result["status"] == "REJECTED"
    assert result["error"] == "invalid_telemetry"


@pytest.mark.parametrize(
    "field,value",
    [
        ("cpu_millicores", ""),
        ("cpu_millicores", "-1"),
        ("cpu_millicores", "nan"),
        ("cpu_millicores", "inf"),
        ("memory_mib", ""),
        ("memory_mib", "-1"),
        ("memory_mib", "nan"),
        ("memory_mib", "inf"),
    ],
)
def test_direct_invalid_measurements_are_rejected(field, value):
    record = telemetry(10.0)
    record[field] = value

    result = controller.check_service("frontend", record)

    assert result["status"] == "REJECTED"
    assert result["error"] == "invalid_telemetry"
    assert controller.consecutive_anomalies.get("frontend", 0) == 0


@pytest.mark.parametrize("timestamp", ["not-a-timestamp", float("nan"), float("inf")])
def test_direct_invalid_timestamps_are_rejected(timestamp):
    record = telemetry(10.0)
    record["timestamp"] = timestamp

    result = controller.check_service("frontend", record)

    assert result["status"] == "REJECTED"
    assert result["error"] == "invalid_telemetry"


def test_failed_remediation_is_reported(monkeypatch):
    monkeypatch.setattr(
        controller,
        "restart_service",
        lambda service, persistence_verified=False: {
            "service": service,
            "status": "FAILED",
            "message": "Remediation command failed.",
            "error": "kubectl unavailable",
        },
    )

    result = controller.handle_remediation("frontend")

    assert result["status"] == "FAILED"
    assert result["error"] == "kubectl unavailable"


def test_remediation_returns_structured_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(
        remediation,
        "LOG_FILE",
        tmp_path / "remediation_log.jsonl",
    )
    monkeypatch.setattr(
        remediation,
        "run_command",
        lambda command: (_ for _ in ()).throw(RuntimeError("kubectl failed")),
    )

    result = remediation.restart_service(
        "frontend",
        persistence_verified=True,
    )

    assert result["status"] == "FAILED"
    assert result["error"] == "kubectl failed"


def test_direct_remediation_requires_persistence_approval(monkeypatch):
    monkeypatch.setattr(
        remediation,
        "run_command",
        lambda command: pytest.fail("kubectl must not run"),
    )

    result = remediation.restart_service("frontend")

    assert result["status"] == "REJECTED"
    assert result["error"] == "persistence_required"


def test_remediation_log_path_is_project_relative():
    assert remediation.LOG_FILE.name == "remediation_log.jsonl"
    assert remediation.LOG_FILE.parent.name == "live"
    assert remediation.LOG_FILE.parent.parent.name == "data"
    assert remediation.LOG_FILE.is_absolute()
