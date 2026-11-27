from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
import asyncio
import json
import multiprocessing

import pytest

from src.api import main
from src.telemetry import auto_heal_controller as controller
from src.telemetry import live_remediation


def record(
    timestamp,
    *,
    service="frontend",
    namespace="default",
    pod="frontend-1",
    container="server",
):
    return {
        "timestamp": timestamp.isoformat(),
        "namespace": namespace,
        "service": service,
        "pod": pod,
        "container": container,
        "cpu_millicores": 100.0,
        "memory_mib": 10.0,
    }


def _cross_process_worker(state_path, timestamp, result_queue):
    from src.telemetry import auto_heal_controller as child_controller
    from src.telemetry import live_ai as child_ai
    from src.telemetry import live_remediation as child_remediation

    row = record(
        datetime.fromisoformat(timestamp.replace("Z", "+00:00")),
    )
    child_remediation.DRY_RUN_STATE_FILE = state_path
    child_controller.persistence_trusted = True
    child_controller.persistence_error = None
    child_controller.load_state = lambda: None
    child_ai._read_live_rows = lambda: ([row], None)
    child_ai._diagnose = lambda rows, **kwargs: {
        **diagnosis(),
        "_evidence_record": dict(row),
        "_evidence_fault": "CPU",
        "_evidence_score": 0.8,
    }
    result = child_remediation.evaluate_dry_run()
    result_queue.put({
        "decision": result["dry_run_decision"],
        "confirmation_count": result["confirmation_count"],
    })


def _stale_loaded_state_worker(
    state_path,
    initial_timestamp,
    next_timestamp,
    barrier,
    proceed_event,
    first_process,
    result_queue,
):
    from src.telemetry import auto_heal_controller as child_controller
    from src.telemetry import live_ai as child_ai
    from src.telemetry import live_remediation as child_remediation

    child_remediation.DRY_RUN_STATE_FILE = state_path
    child_controller.persistence_trusted = True
    child_controller.persistence_error = None
    child_controller.load_state = lambda: None
    initial_row = record(
        datetime.fromisoformat(initial_timestamp.replace("Z", "+00:00")),
    )
    next_row = record(
        datetime.fromisoformat(next_timestamp.replace("Z", "+00:00")),
    )
    child_ai._read_live_rows = lambda: ([initial_row], None)
    child_ai._diagnose = lambda rows, **kwargs: {
        **diagnosis(),
        "_evidence_record": dict(initial_row),
        "_evidence_fault": "CPU",
        "_evidence_score": 0.8,
    }
    child_remediation.evaluate_dry_run()
    barrier.wait(timeout=10)
    if not first_process:
        proceed_event.wait(timeout=10)
    child_ai._read_live_rows = lambda: ([next_row], None)
    child_ai._diagnose = lambda rows, **kwargs: {
        **diagnosis(),
        "_evidence_record": dict(next_row),
        "_evidence_fault": "CPU",
        "_evidence_score": 0.8,
    }
    result = child_remediation.evaluate_dry_run()
    if first_process:
        proceed_event.set()
    result_queue.put(result["confirmation_count"])


def diagnosis(status="anomalous", service="frontend", fault="CPU"):
    return {
        "status": status,
        "affected_service": service,
        "suspected_fault": fault if status == "anomalous" else None,
        "confidence": 0.8 if status == "anomalous" else 0.0,
        "evidence": ["CPU changed by 90 millicores."] if status == "anomalous" else [],
    }


@pytest.fixture(autouse=True)
def reset_state(monkeypatch, tmp_path):
    live_remediation.reset_dry_run_state()
    monkeypatch.setattr(
        live_remediation,
        "DRY_RUN_STATE_FILE",
        tmp_path / "dry_run_state.json",
    )
    controller.last_restart.clear()
    controller.persistence_trusted = True
    controller.persistence_error = None
    monkeypatch.setattr(
        controller,
        "load_state",
        lambda: setattr(controller, "persistence_trusted", True),
    )


def configure(monkeypatch, rows, result):
    evidence = result.get("_evidence_record") or max(
        rows,
        key=lambda row: row["timestamp"],
    )
    monkeypatch.setattr(
        live_remediation.live_ai,
        "_read_live_rows",
        lambda: (rows, None),
    )
    monkeypatch.setattr(
        live_remediation.live_ai,
        "_diagnose",
        lambda current_rows, **kwargs: {
            **result,
            **(
                {
                    "_evidence_record": evidence,
                    "_evidence_fault": result.get("suspected_fault"),
                    "_evidence_score": result.get("confidence"),
                }
                if result.get("status") == "anomalous"
                else {}
            ),
        },
    )


def test_normal_telemetry_is_not_eligible(monkeypatch):
    row = record(datetime.now(timezone.utc))
    configure(monkeypatch, [row], diagnosis(status="normal"))

    result = live_remediation.evaluate_dry_run()

    assert result["dry_run_decision"] == "not_eligible"
    assert result["eligible_for_remediation"] is False


@pytest.mark.parametrize(
    ("offset", "reason"),
    [
        (31, "stale"),
        (-1, "future"),
    ],
)
def test_stale_and_future_telemetry_are_blocked(monkeypatch, offset, reason):
    row = record(datetime.now(timezone.utc) - timedelta(seconds=offset))
    configure(monkeypatch, [row], diagnosis())

    result = live_remediation.evaluate_dry_run()

    assert result["dry_run_decision"] == "blocked"
    assert reason in result["reason"].lower()


def test_insufficient_evidence_is_blocked(monkeypatch):
    row = record(datetime.now(timezone.utc))
    configure(
        monkeypatch,
        [row],
        {
            **diagnosis(status="insufficient_evidence"),
            "affected_service": None,
        },
    )

    result = live_remediation.evaluate_dry_run()

    assert result["dry_run_decision"] == "not_eligible"
    assert "not anomalous" in result["reason"]


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("service", "not-approved", "allowlist"),
        ("namespace", "other", "default namespace"),
        ("pod", "", "pod and container"),
        ("container", "", "pod and container"),
    ],
)
def test_identity_and_allowlist_safety_checks(
    monkeypatch,
    field,
    value,
    expected,
):
    row = record(datetime.now(timezone.utc))
    row[field] = value
    configure(monkeypatch, [row], diagnosis(service=row["service"]))

    result = live_remediation.evaluate_dry_run()

    assert result["dry_run_decision"] == "blocked"
    assert expected in result["reason"].lower()


def test_first_anomaly_sample_is_confirmation_pending(monkeypatch):
    row = record(datetime.now(timezone.utc))
    configure(monkeypatch, [row], diagnosis())

    result = live_remediation.evaluate_dry_run()

    assert result["dry_run_decision"] == "confirmation_pending"
    assert result["confirmation_count"] == 1
    assert result["required_confirmations"] == controller.REQUIRED_ANOMALIES


def test_repeated_polling_does_not_increase_confirmation_count(monkeypatch):
    row = record(datetime.now(timezone.utc))
    configure(monkeypatch, [row], diagnosis())

    first = live_remediation.evaluate_dry_run()
    second = live_remediation.evaluate_dry_run()

    assert first["confirmation_count"] == 1
    assert second["confirmation_count"] == 1
    assert second["dry_run_decision"] == "duplicate"


def test_distinct_anomaly_samples_progress_confirmation(monkeypatch):
    now = datetime.now(timezone.utc)
    rows = [record(now - timedelta(seconds=2), pod="frontend-1")]
    configure(monkeypatch, rows, diagnosis())

    first = live_remediation.evaluate_dry_run()
    rows[0] = record(now - timedelta(seconds=1), pod="frontend-1")
    configure(monkeypatch, rows, diagnosis())
    second = live_remediation.evaluate_dry_run()
    rows[0] = record(now, pod="frontend-1")
    configure(monkeypatch, rows, diagnosis())
    third = live_remediation.evaluate_dry_run()

    assert [first["confirmation_count"], second["confirmation_count"]] == [1, 2]
    assert third["confirmation_count"] == controller.REQUIRED_ANOMALIES
    assert third["dry_run_decision"] == "eligible"


def test_replacement_pod_has_separate_confirmation_identity(monkeypatch):
    now = datetime.now(timezone.utc)
    rows = [record(now - timedelta(seconds=2), pod="frontend-old")]
    configure(monkeypatch, rows, diagnosis())
    old_result = live_remediation.evaluate_dry_run()

    rows[0] = record(now - timedelta(seconds=1), pod="frontend-new")
    configure(monkeypatch, rows, diagnosis())
    new_result = live_remediation.evaluate_dry_run()

    assert old_result["incident_id"] != new_result["incident_id"]
    assert new_result["confirmation_count"] == 1


def test_diagnosis_evidence_remains_bound_to_diagnosed_pod(monkeypatch):
    now = datetime.now(timezone.utc)
    pod_a = record(now - timedelta(seconds=1), pod="frontend-a")
    pod_b = record(now, pod="frontend-b")
    result = diagnosis()
    result["_evidence_record"] = pod_a
    configure(monkeypatch, [pod_a, pod_b], result)

    dry_run = live_remediation.evaluate_dry_run()

    assert dry_run["pod"] == "frontend-a"
    assert dry_run["telemetry_timestamp"] == pod_a["timestamp"]


def test_real_diagnosis_binds_orchestrator_to_pod_a(monkeypatch):
    now = datetime.now(timezone.utc)
    rows = [
        record(now - timedelta(seconds=20), pod="frontend-a"),
        record(now - timedelta(seconds=15), pod="frontend-a"),
        record(now - timedelta(seconds=10), pod="frontend-a"),
        record(now - timedelta(seconds=5), pod="frontend-a"),
        record(now - timedelta(seconds=1), pod="frontend-a"),
        record(now, pod="frontend-b"),
    ]
    for row in rows[:4]:
        row["cpu_millicores"] = 10.0
    rows[4]["cpu_millicores"] = 100.0
    rows[5]["cpu_millicores"] = 10.0
    monkeypatch.setattr(
        live_remediation.live_ai,
        "_read_live_rows",
        lambda: (rows, None),
    )

    result = live_remediation.evaluate_dry_run()

    assert result["pod"] == "frontend-a"
    assert result["container"] == "server"


def test_timestamp_equivalent_replay_is_duplicate(monkeypatch):
    instant = datetime.now(timezone.utc) - timedelta(seconds=1)
    first = record(instant)
    equivalent = record(
        instant.astimezone(timezone(timedelta(hours=5, minutes=30)))
    )
    configure(monkeypatch, [first], diagnosis())
    live_remediation.evaluate_dry_run()
    configure(monkeypatch, [equivalent], diagnosis())

    result = live_remediation.evaluate_dry_run()

    assert result["dry_run_decision"] == "duplicate"
    assert result["confirmation_count"] == 1


def test_cpu_anomaly_is_labelled_without_execution(monkeypatch):
    row = record(datetime.now(timezone.utc))
    configure(monkeypatch, [row], diagnosis(fault="CPU"))

    result = live_remediation.evaluate_dry_run()

    assert result["anomaly_type"] == "CPU"
    assert result["execution_performed"] is False


def test_memory_anomaly_is_labelled_and_stays_dry_run(monkeypatch):
    row = record(datetime.now(timezone.utc))
    configure(monkeypatch, [row], diagnosis(fault="MEM"))

    result = live_remediation.evaluate_dry_run()

    assert result["anomaly_type"] == "MEM"
    assert result["dry_run_decision"] == "confirmation_pending"
    assert result["execution_performed"] is False


def test_active_cooldown_blocks_after_confirmation(monkeypatch):
    row = record(datetime.now(timezone.utc))
    configure(monkeypatch, [row], diagnosis())
    controller.last_restart["frontend"] = datetime.now(timezone.utc).timestamp()

    result = live_remediation.evaluate_dry_run()

    assert result["dry_run_decision"] == "blocked"
    assert result["cooldown_active"] is True


def test_cooldown_samples_do_not_count_after_expiry(monkeypatch):
    now = datetime.now(timezone.utc)
    controller.last_restart["frontend"] = now.timestamp()
    rows = [record(now - timedelta(seconds=2))]
    configure(monkeypatch, rows, diagnosis())

    during = live_remediation.evaluate_dry_run()
    controller.last_restart["frontend"] = (
        now - timedelta(seconds=controller.COOLDOWN_SECONDS + 1)
    ).timestamp()
    after_same = live_remediation.evaluate_dry_run()

    assert during["dry_run_decision"] == "blocked"
    assert during["confirmation_count"] == 0
    assert after_same["dry_run_decision"] == "duplicate"
    assert after_same["confirmation_count"] == 0


def test_new_confirmations_after_cooldown_expire(monkeypatch):
    now = datetime.now(timezone.utc)
    controller.last_restart["frontend"] = now.timestamp()
    rows = [record(now - timedelta(seconds=3))]
    configure(monkeypatch, rows, diagnosis())
    live_remediation.evaluate_dry_run()
    controller.last_restart["frontend"] = (
        now - timedelta(seconds=controller.COOLDOWN_SECONDS + 1)
    ).timestamp()

    results = []
    for seconds in (2, 1, 0):
        rows[0] = record(now - timedelta(seconds=seconds))
        configure(monkeypatch, rows, diagnosis())
        results.append(live_remediation.evaluate_dry_run())

    assert [result["confirmation_count"] for result in results] == [1, 2, 3]
    assert results[-1]["dry_run_decision"] == "eligible"


def test_untrusted_persistence_blocks(monkeypatch):
    row = record(datetime.now(timezone.utc))
    configure(monkeypatch, [row], diagnosis())

    def fail_load():
        controller.persistence_trusted = False
        controller.persistence_error = "state is corrupted"

    monkeypatch.setattr(controller, "load_state", fail_load)
    result = live_remediation.evaluate_dry_run()

    assert result["dry_run_decision"] == "blocked"
    assert result["persistence_trusted"] is False
    assert "corrupted" in result["reason"]


def test_unexpected_controller_load_exception_blocks(monkeypatch):
    row = record(datetime.now(timezone.utc))
    configure(monkeypatch, [row], diagnosis())
    monkeypatch.setattr(
        controller,
        "load_state",
        lambda: (_ for _ in ()).throw(RuntimeError("load failed")),
    )

    result = live_remediation.evaluate_dry_run()

    assert result["dry_run_decision"] == "blocked"
    assert "load failed" in result["reason"]


def test_missing_state_after_initialization_blocks(monkeypatch):
    row = record(datetime.now(timezone.utc))
    configure(monkeypatch, [row], diagnosis())
    first = live_remediation.evaluate_dry_run()
    live_remediation.DRY_RUN_STATE_FILE.unlink()
    rows = [record(datetime.now(timezone.utc) - timedelta(seconds=1))]
    configure(monkeypatch, rows, diagnosis())

    result = live_remediation.evaluate_dry_run()

    assert first["confirmation_count"] == 1
    assert result["dry_run_decision"] == "blocked"
    assert "disappeared" in result["reason"]
    assert live_remediation._confirmation_counts[first["incident_id"]] == 1


def test_malformed_dry_run_state_blocks(monkeypatch):
    live_remediation.DRY_RUN_STATE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    live_remediation.DRY_RUN_STATE_FILE.write_text(
        json.dumps({"version": 1, "incidents": []}),
        encoding="utf-8",
    )
    row = record(datetime.now(timezone.utc))
    configure(monkeypatch, [row], diagnosis())

    result = live_remediation.evaluate_dry_run()

    assert result["dry_run_decision"] == "blocked"
    assert "untrusted" in result["reason"]


def test_persistence_failure_blocks(monkeypatch):
    row = record(datetime.now(timezone.utc))
    configure(monkeypatch, [row], diagnosis())
    monkeypatch.setattr(
        live_remediation,
        "_save_dry_run_state",
        lambda: (
            setattr(live_remediation, "_state_error", "write failed")
            or setattr(live_remediation, "_state_trusted", False)
            or False
        ),
    )

    result = live_remediation.evaluate_dry_run()

    assert result["dry_run_decision"] == "blocked"
    assert "write failed" in result["reason"]


def test_json_replacement_failure_cleans_temporary_file(monkeypatch):
    row = record(datetime.now(timezone.utc))
    configure(monkeypatch, [row], diagnosis())
    original_replace = type(live_remediation.DRY_RUN_STATE_FILE).replace

    def fail_replace(path, target):
        raise OSError("replace failed")

    monkeypatch.setattr(
        type(live_remediation.DRY_RUN_STATE_FILE),
        "replace",
        fail_replace,
    )
    result = live_remediation.evaluate_dry_run()

    assert result["dry_run_decision"] == "blocked"
    assert "replace failed" in result["reason"]
    assert not live_remediation.DRY_RUN_STATE_FILE.with_suffix(".tmp").exists()
    monkeypatch.setattr(
        type(live_remediation.DRY_RUN_STATE_FILE),
        "replace",
        original_replace,
    )


def test_sqlite_lock_failure_blocks_without_counting(monkeypatch):
    row = record(datetime.now(timezone.utc))
    configure(monkeypatch, [row], diagnosis())

    def fail_connect(*args, **kwargs):
        raise live_remediation.sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(live_remediation.sqlite3, "connect", fail_connect)
    result = live_remediation.evaluate_dry_run()

    assert result["dry_run_decision"] == "blocked"
    assert "database is locked" in result["reason"]
    assert result["confirmation_count"] == 0


def test_failed_save_does_not_restore_trust_on_next_evaluation(monkeypatch):
    row = record(datetime.now(timezone.utc))
    configure(monkeypatch, [row], diagnosis())
    monkeypatch.setattr(
        live_remediation,
        "_save_dry_run_state",
        lambda: (
            setattr(live_remediation, "_state_error", "write failed")
            or setattr(live_remediation, "_state_trusted", False)
            or False
        ),
    )
    first = live_remediation.evaluate_dry_run()
    second = live_remediation.evaluate_dry_run()

    assert first["dry_run_decision"] == "blocked"
    assert second["dry_run_decision"] == "blocked"
    assert "write failed" in second["reason"]


def test_semantically_invalid_state_is_rejected(monkeypatch):
    identity = {
        "namespace": "default",
        "service": "frontend",
        "pod": "frontend-1",
        "container": "server",
        "fault": "CPU",
    }
    incident_id = live_remediation._incident_id(**identity)
    valid_sample = "2026-11-05T15:00:00Z"
    invalid_states = [
        {
            "version": 1,
            "incidents": {
                incident_id: {
                    "identity": identity,
                    "samples": [valid_sample],
                    "confirmation_samples": [],
                    "confirmation_count": 1,
                    "last_sample_timestamp": valid_sample,
                }
            },
        },
        {
            "version": 1,
            "incidents": {
                "invalid": {
                    "identity": identity,
                    "samples": [],
                    "confirmation_samples": [],
                    "confirmation_count": 0,
                    "last_sample_timestamp": None,
                }
            },
        },
    ]
    for state in invalid_states:
        with pytest.raises(ValueError):
            live_remediation._validate_state(state)


def test_state_validation_rejects_non_chronological_samples():
    identity = {
        "namespace": "default",
        "service": "frontend",
        "pod": "frontend-1",
        "container": "server",
        "fault": "CPU",
    }
    incident_id = live_remediation._incident_id(**identity)
    with pytest.raises(ValueError, match="chronological"):
        live_remediation._validate_state({
            "version": 1,
            "incidents": {
                incident_id: {
                    "identity": identity,
                    "samples": [
                        "2026-11-05T15:00:01Z",
                        "2026-11-05T15:00:00Z",
                    ],
                    "confirmation_samples": [],
                    "confirmation_count": 0,
                    "last_sample_timestamp": "2026-11-05T15:00:01Z",
                },
            },
        })


def test_state_validation_requires_last_timestamp_for_samples():
    identity = {
        "namespace": "default",
        "service": "frontend",
        "pod": "frontend-1",
        "container": "server",
        "fault": "CPU",
    }
    incident_id = live_remediation._incident_id(**identity)
    with pytest.raises(ValueError, match="last sample"):
        live_remediation._validate_state({
            "version": 1,
            "incidents": {
                incident_id: {
                    "identity": identity,
                    "samples": ["2026-11-05T15:00:00Z"],
                    "confirmation_samples": [],
                    "confirmation_count": 0,
                    "last_sample_timestamp": None,
                },
            },
        })


def test_explicit_recovery_requires_confirmation(monkeypatch):
    assert live_remediation.recover_dry_run_state()[0] is False
    assert "confirmation" in live_remediation.recover_dry_run_state()[1]


def test_explicit_recovery_reloads_valid_state(monkeypatch):
    row = record(datetime.now(timezone.utc))
    configure(monkeypatch, [row], diagnosis())
    live_remediation.evaluate_dry_run()
    live_remediation._mark_state_untrusted("simulated failure")

    recovered, error = live_remediation.recover_dry_run_state(confirm=True)

    assert recovered is True
    assert error is None
    assert live_remediation._state_trusted is True


def test_backend_state_reload_preserves_duplicate_protection(monkeypatch):
    row = record(datetime.now(timezone.utc))
    configure(monkeypatch, [row], diagnosis())
    first = live_remediation.evaluate_dry_run()
    live_remediation.reset_dry_run_state()
    configure(monkeypatch, [row], diagnosis())

    result = live_remediation.evaluate_dry_run()

    assert first["confirmation_count"] == 1
    assert result["dry_run_decision"] == "duplicate"
    assert result["confirmation_count"] == 1


def test_concurrent_requests_count_same_sample_once(monkeypatch):
    row = record(datetime.now(timezone.utc))
    configure(monkeypatch, [row], diagnosis())

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(
            lambda _index: live_remediation.evaluate_dry_run(),
            range(8),
        ))

    assert sum(
        result["dry_run_decision"] == "confirmation_pending"
        for result in results
    ) == 1
    assert sum(
        result["dry_run_decision"] == "duplicate"
        for result in results
    ) == 7
    assert all(result["confirmation_count"] == 1 for result in results)


def test_two_processes_coordinate_same_sample(tmp_path):
    context = multiprocessing.get_context("spawn")
    state_path = tmp_path / "dry_run_state.json"
    timestamp = (
        datetime.now(timezone.utc) - timedelta(seconds=1)
    ).isoformat().replace("+00:00", "Z")
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_cross_process_worker,
            args=(state_path, timestamp, result_queue),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)

    assert all(process.exitcode == 0 for process in processes)
    results = [result_queue.get(timeout=2) for _ in processes]
    assert sorted(result["decision"] for result in results) == [
        "confirmation_pending",
        "duplicate",
    ]
    assert {result["confirmation_count"] for result in results} == {1}


def test_two_processes_reload_stale_loaded_state_without_lost_updates(
    monkeypatch,
    tmp_path,
):
    context = multiprocessing.get_context("spawn")
    state_path = tmp_path / "dry_run_state.json"
    initial = datetime.now(timezone.utc) - timedelta(seconds=3)
    first = record(initial)
    live_remediation.DRY_RUN_STATE_FILE = state_path
    monkeypatch.setattr(
        live_remediation.live_ai,
        "_read_live_rows",
        lambda: ([first], None),
    )
    monkeypatch.setattr(
        live_remediation.live_ai,
        "_diagnose",
        lambda rows, **kwargs: {
            **diagnosis(),
            "_evidence_record": dict(first),
            "_evidence_fault": "CPU",
            "_evidence_score": 0.8,
        },
    )
    assert live_remediation.evaluate_dry_run()["confirmation_count"] == 1

    barrier = context.Barrier(2)
    proceed_event = context.Event()
    result_queue = context.Queue()
    timestamps = [
        (initial + timedelta(seconds=1)).isoformat().replace("+00:00", "Z"),
        (initial + timedelta(seconds=2)).isoformat().replace("+00:00", "Z"),
    ]
    processes = [
        context.Process(
            target=_stale_loaded_state_worker,
            args=(state_path, initial.isoformat().replace("+00:00", "Z"),
                  timestamp, barrier, proceed_event, index == 0,
                  result_queue),
        )
        for index, timestamp in enumerate(timestamps)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)

    assert all(process.exitcode == 0 for process in processes)
    assert sorted(result_queue.get(timeout=2) for _ in processes) == [2, 3]
    state = json.loads(state_path.read_text(encoding="utf-8"))
    incident = next(iter(state["incidents"].values()))
    assert incident["confirmation_count"] == 3
    assert len(incident["confirmation_samples"]) == 3


def test_json_save_then_sqlite_commit_failure_fails_closed(
    monkeypatch,
):
    row = record(datetime.now(timezone.utc))
    configure(monkeypatch, [row], diagnosis())
    real_connect = live_remediation.sqlite3.connect

    class CommitFailureConnection:
        def __init__(self, connection):
            self._connection = connection

        def __getattr__(self, name):
            return getattr(self._connection, name)

        def commit(self):
            raise live_remediation.sqlite3.OperationalError(
                "commit failed"
            )

    monkeypatch.setattr(
        live_remediation.sqlite3,
        "connect",
        lambda *args, **kwargs: CommitFailureConnection(
            real_connect(*args, **kwargs)
        ),
    )
    first = live_remediation.evaluate_dry_run()
    assert first["dry_run_decision"] == "blocked"
    assert first["persistence_trusted"] is False
    assert live_remediation.DRY_RUN_STATE_FILE.exists()

    second = live_remediation.evaluate_dry_run()
    assert second["dry_run_decision"] == "blocked"
    assert second["persistence_trusted"] is False
def test_duplicate_incident_is_reported_without_reprocessing(monkeypatch):
    row = record(datetime.now(timezone.utc))
    configure(monkeypatch, [row], diagnosis())
    first = live_remediation.evaluate_dry_run()
    duplicate = live_remediation.evaluate_dry_run()

    assert duplicate["incident_id"] == first["incident_id"]
    assert duplicate["dry_run_decision"] == "duplicate"


def test_unavailable_telemetry_is_safe(monkeypatch):
    monkeypatch.setattr(
        live_remediation.live_ai,
        "_read_live_rows",
        lambda: ([], "Live telemetry file is empty."),
    )

    result = live_remediation.evaluate_dry_run()

    assert result["dry_run_decision"] == "unavailable"
    assert result["execution_performed"] is False


def test_api_response_has_dry_run_contract(monkeypatch):
    expected = {
        "dry_run": True,
        "execution_performed": False,
        "dry_run_decision": "blocked",
    }
    monkeypatch.setattr(
        main,
        "get_dry_run_status",
        lambda: expected,
    )

    result = main.remediation_dry_run()

    assert result == expected


def test_dry_run_http_route_serializes_read_only_response(monkeypatch):
    monkeypatch.setattr(main, "get_dry_run_status", lambda: {
        "dry_run": True,
        "execution_performed": False,
        "dry_run_decision": "blocked",
    })

    messages = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    async def request():
        await main.app(
            {
                "type": "http",
                "http_version": "1.1",
                "method": "GET",
                "path": "/api/remediation/dry-run",
                "raw_path": b"/api/remediation/dry-run",
                "query_string": b"",
                "headers": [],
                "scheme": "http",
                "server": ("testserver", 80),
                "client": ("testclient", 123),
            },
            receive,
            send,
        )

    asyncio.run(request())
    status = next(message for message in messages if message["type"] == "http.response.start")
    body = b"".join(
        message["body"]
        for message in messages
        if message["type"] == "http.response.body"
    )

    assert status["status"] == 200
    assert json.loads(body)["dry_run"] is True
    assert json.loads(body)["execution_performed"] is False


def test_dry_run_never_calls_remediation(monkeypatch):
    row = record(datetime.now(timezone.utc))
    configure(monkeypatch, [row], diagnosis())
    monkeypatch.setattr(
        live_remediation.controller,
        "handle_remediation",
        lambda service: pytest.fail("controller remediation must not run"),
    )
    monkeypatch.setattr(
        live_remediation.controller,
        "restart_service",
        lambda *args, **kwargs: pytest.fail("restart must not run"),
    )

    result = live_remediation.evaluate_dry_run()

    assert result["execution_performed"] is False
