"""Dry-run orchestration for live anomaly remediation decisions.

This module deliberately contains no remediation execution path. It evaluates
validated live diagnosis evidence and reports what a guarded remediation step
would require. Dry-run state is persisted only to prevent replay; it is never
an execution authorization. Processes using the same state path coordinate
through SQLite, while JSON replacement and SQLite commit remain separate and
do not provide a full crash-durability guarantee.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any
from contextlib import contextmanager

from src.telemetry import auto_heal_controller as controller
from src.telemetry import live_ai


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DRY_RUN_STATE_FILE = PROJECT_ROOT / "data" / "live" / "dry_run_state.json"
STATE_VERSION = 1
REQUIRED_CONFIRMATIONS = controller.REQUIRED_ANOMALIES
APPROVED_SERVICES = controller.APPROVED_SERVICES


@dataclass(frozen=True)
class IncidentEvidence:
    incident_id: str | None
    detection_timestamp: str | None
    namespace: str | None
    service: str | None
    pod: str | None
    container: str | None
    anomaly_type: str | None
    diagnosis_status: str
    evidence: list[str]
    relative_deviation: float | None
    telemetry_timestamp: str | None
    telemetry_age_seconds: float | None
    confirmation_count: int
    cooldown_active: bool | None
    persistence_trusted: bool
    dry_run_decision: str
    reason: str


# The SQLite transaction below coordinates processes that use the same state
# path. This lock only coordinates threads within one process.
_STATE_LOCK = threading.RLock()
_seen_samples: dict[str, set[str]] = {}
_confirmation_counts: dict[str, int] = {}
_last_sample_timestamps: dict[str, datetime] = {}
_incident_identities: dict[str, dict[str, str]] = {}
_confirmation_samples: dict[str, set[str]] = {}
_state_loaded = False
_state_trusted = False
_state_error: str | None = None


def reset_dry_run_state() -> None:
    """Reset process-local tracking for tests or deliberate local reset."""

    global _state_loaded, _state_trusted, _state_error
    with _STATE_LOCK:
        _seen_samples.clear()
        _confirmation_counts.clear()
        _last_sample_timestamps.clear()
        _incident_identities.clear()
        _confirmation_samples.clear()
        _state_loaded = False
        _state_trusted = False
        _state_error = None


@contextmanager
def _process_coordination():
    """Serialize state read-modify-write operations across backend processes.

    The JSON state is reloaded only after this transaction has acquired its
    write lock. JSON replacement and SQLite commit are separate operations, so
    this prevents lost updates but cannot provide full crash atomicity.
    """

    DRY_RUN_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(
        str(DRY_RUN_STATE_FILE.with_suffix(".coordination.sqlite3")),
        timeout=2,
    )
    try:
        connection.execute("PRAGMA busy_timeout = 2000")
        connection.execute("BEGIN IMMEDIATE")
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _incident_identity(
    namespace: Any,
    service: Any,
    pod: Any,
    container: Any,
    fault: Any,
) -> dict[str, str]:
    return {
        "namespace": namespace,
        "service": service,
        "pod": pod,
        "container": container,
        "fault": fault,
    }


def _incident_id(
    namespace: str,
    service: str,
    pod: str,
    container: str,
    fault: str,
) -> str:
    identity = "|".join((namespace, service, pod, container, fault))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def _canonical_timestamp(value: Any) -> tuple[str, datetime] | None:
    timestamp = live_ai._parse_timestamp(value)
    if timestamp is None:
        return None
    utc_timestamp = timestamp.astimezone(timezone.utc)
    return utc_timestamp.isoformat().replace("+00:00", "Z"), utc_timestamp


def _state_snapshot() -> dict[str, Any]:
    incidents = {}
    for incident_id, samples in _seen_samples.items():
        incidents[incident_id] = {
            "identity": _incident_identities.get(incident_id),
            "samples": sorted(samples),
            "confirmation_samples": sorted(
                _confirmation_samples.get(incident_id, set())
            ),
            "confirmation_count": _confirmation_counts.get(incident_id, 0),
            "last_sample_timestamp": (
                _last_sample_timestamps[incident_id].isoformat().replace(
                    "+00:00", "Z"
                )
                if incident_id in _last_sample_timestamps
                else None
            ),
        }
    return {"version": STATE_VERSION, "incidents": incidents}


def _validate_state(
    data: Any,
) -> tuple[
    dict[str, set[str]],
    dict[str, int],
    dict[str, datetime],
    dict[str, dict[str, str]],
    dict[str, set[str]],
]:
    if not isinstance(data, dict) or data.get("version") != STATE_VERSION:
        raise ValueError("dry-run state has an invalid structure")
    incidents = data.get("incidents")
    if not isinstance(incidents, dict):
        raise ValueError("dry-run incidents must be an object")

    seen: dict[str, set[str]] = {}
    counts: dict[str, int] = {}
    last_timestamps: dict[str, datetime] = {}
    identities: dict[str, dict[str, str]] = {}
    confirmation_samples: dict[str, set[str]] = {}
    for incident_id, entry in incidents.items():
        if not isinstance(incident_id, str) or not isinstance(entry, dict):
            raise ValueError("dry-run incident entry is invalid")
        identity = entry.get("identity")
        if (
            not isinstance(identity, dict)
            or set(identity) != {"namespace", "service", "pod", "container", "fault"}
            or not all(
                isinstance(value, str) and value
                for value in identity.values()
            )
            or _incident_id(**identity) != incident_id
        ):
            raise ValueError("dry-run incident identity is invalid")
        samples = entry.get("samples")
        counted = entry.get("confirmation_samples")
        count = entry.get("confirmation_count")
        if (
            not isinstance(samples, list)
            or not isinstance(counted, list)
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
        ):
            raise ValueError("dry-run incident samples are invalid")

        normalized_samples = set()
        previous_sample: datetime | None = None
        for sample in samples:
            normalized = _canonical_timestamp(sample)
            if normalized is None:
                raise ValueError("dry-run sample timestamp is invalid")
            if normalized[0] in normalized_samples:
                raise ValueError("dry-run samples are not unique")
            if previous_sample is not None and normalized[1] <= previous_sample:
                raise ValueError("dry-run samples are not chronological")
            previous_sample = normalized[1]
            normalized_samples.add(normalized[0])

        normalized_counted = set()
        for sample in counted:
            normalized = _canonical_timestamp(sample)
            if normalized is None or normalized[0] in normalized_counted:
                raise ValueError("dry-run confirmation samples are invalid")
            normalized_counted.add(normalized[0])
        if not normalized_counted.issubset(normalized_samples):
            raise ValueError("confirmation samples must be observed samples")
        if count != len(normalized_counted):
            raise ValueError("confirmation count does not match samples")

        last_sample = entry.get("last_sample_timestamp")
        if last_sample is not None:
            normalized_last = _canonical_timestamp(last_sample)
            if normalized_last is None:
                raise ValueError("dry-run last sample timestamp is invalid")
            if not normalized_samples or normalized_last[0] != max(normalized_samples):
                raise ValueError("last sample timestamp is inconsistent")
            last_timestamps[incident_id] = normalized_last[1]
        elif normalized_samples:
            raise ValueError(
                "non-empty samples require a last sample timestamp"
            )

        seen[incident_id] = normalized_samples
        counts[incident_id] = count
        identities[incident_id] = identity
        confirmation_samples[incident_id] = normalized_counted
    return seen, counts, last_timestamps, identities, confirmation_samples


def _mark_state_untrusted(message: str) -> None:
    global _state_trusted, _state_error
    _state_trusted = False
    _state_error = message


def _save_dry_run_state() -> bool:
    global _state_loaded, _state_trusted, _state_error
    temporary_file = DRY_RUN_STATE_FILE.with_suffix(".tmp")
    snapshot = _state_snapshot()
    try:
        DRY_RUN_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with temporary_file.open("w", encoding="utf-8") as file:
            json.dump(snapshot, file, indent=2, sort_keys=True)
        temporary_file.replace(DRY_RUN_STATE_FILE)
        with DRY_RUN_STATE_FILE.open(encoding="utf-8") as file:
            written = json.load(file)
        if written != snapshot:
            raise ValueError("dry-run state could not be verified")
    except Exception as error:
        try:
            temporary_file.unlink(missing_ok=True)
        except OSError:
            pass
        _mark_state_untrusted(f"Dry-run state could not be persisted: {error}")
        return False

    _state_loaded = True
    _state_trusted = True
    _state_error = None
    return True


def _load_dry_run_state(
    *,
    force: bool = False,
    explicit_recovery: bool = False,
) -> tuple[bool, str | None]:
    global _state_loaded, _state_trusted, _state_error
    if (
        _state_loaded
        and not _state_trusted
        and not explicit_recovery
    ):
        return False, _state_error or "Dry-run state remains untrusted."
    if not DRY_RUN_STATE_FILE.exists():
        if _state_loaded:
            _mark_state_untrusted(
                "Dry-run state file disappeared after initialization."
            )
            return False, _state_error
        _state_loaded = True
        _state_trusted = True
        _state_error = None
        if not _save_dry_run_state():
            return False, _state_error
        return True, None

    try:
        with DRY_RUN_STATE_FILE.open(encoding="utf-8") as file:
            data = json.load(file)
        (
            seen,
            counts,
            last_timestamps,
            identities,
            confirmation_samples,
        ) = _validate_state(data)
    except Exception as error:
        _mark_state_untrusted(f"Dry-run state is untrusted: {error}")
        return False, _state_error

    _seen_samples.clear()
    _seen_samples.update(seen)
    _confirmation_counts.clear()
    _confirmation_counts.update(counts)
    _last_sample_timestamps.clear()
    _last_sample_timestamps.update(last_timestamps)
    _incident_identities.clear()
    _incident_identities.update(identities)
    _confirmation_samples.clear()
    _confirmation_samples.update(confirmation_samples)
    _state_loaded = True
    _state_trusted = True
    _state_error = None
    return True, None


def recover_dry_run_state(*, confirm: bool = False) -> tuple[bool, str | None]:
    """Explicitly revalidate dry-run state; never grants execution authority."""

    if not confirm:
        return False, "Explicit dry-run state recovery confirmation is required."
    with _STATE_LOCK:
        global _state_loaded
        if not DRY_RUN_STATE_FILE.exists():
            return False, "Cannot recover dry-run state because the file is missing."
        _state_loaded = False
        return _load_dry_run_state(force=True, explicit_recovery=True)


def _validate_evidence(
    rows: list[dict[str, Any]],
    diagnosis: dict[str, Any],
    evidence_record: Any,
) -> tuple[bool, str | None]:
    if not isinstance(evidence_record, dict):
        return False, "Anomalous diagnosis did not include exact telemetry evidence."
    normalized = _canonical_timestamp(evidence_record.get("timestamp"))
    if normalized is None:
        return False, "Evidence timestamp is invalid."
    if not live_ai._is_fresh(normalized[1], datetime.now(timezone.utc)):
        return False, "Evidence telemetry is stale or future-dated."

    identity_fields = ("namespace", "service", "pod", "container")
    for row in rows:
        row_timestamp = _canonical_timestamp(row.get("timestamp"))
        if row_timestamp is None:
            continue
        if (
            row_timestamp[0] == normalized[0]
            and all(row.get(field) == evidence_record.get(field)
                    for field in identity_fields)
            and row.get("cpu_millicores") == evidence_record.get("cpu_millicores")
            and row.get("memory_mib") == evidence_record.get("memory_mib")
        ):
            break
    else:
        return False, "Diagnosis evidence is not present in validated telemetry."

    if diagnosis.get("_evidence_fault") != diagnosis.get("suspected_fault"):
        return False, "Evidence anomaly type does not match the diagnosis."
    if diagnosis.get("_evidence_score") != diagnosis.get("confidence"):
        return False, "Evidence score does not match the diagnosis."
    if evidence_record.get("service") != diagnosis.get("affected_service"):
        return False, "Diagnosis and telemetry evidence identify different services."
    return True, None


def _load_persistence_state(*, force_dry_run_reload: bool = False) -> tuple[bool, str | None]:
    try:
        controller.load_state()
    except Exception as error:
        return False, f"Cooldown persistence could not be loaded: {error}"
    if not controller.persistence_trusted:
        return False, controller.persistence_error or (
            "Cooldown persistence is untrusted."
        )
    return _load_dry_run_state(force=force_dry_run_reload)


def _base_result(
    *,
    diagnosis: dict[str, Any],
    reason: str,
    decision: str,
    record: dict[str, Any] | None = None,
    incident_id: str | None = None,
    confirmation_count: int = 0,
    cooldown_active: bool | None = None,
    persistence_trusted: bool = False,
    telemetry_age: float | None = None,
) -> dict[str, Any]:
    confidence = diagnosis.get("confidence")
    relative_deviation = (
        float(confidence)
        if isinstance(confidence, (int, float))
        else None
    )
    evidence = diagnosis.get("evidence")
    if not isinstance(evidence, list):
        evidence = []

    incident = IncidentEvidence(
        incident_id=incident_id,
        detection_timestamp=datetime.now(timezone.utc).isoformat(),
        namespace=record.get("namespace") if record else None,
        service=record.get("service") if record else diagnosis.get("affected_service"),
        pod=record.get("pod") if record else None,
        container=record.get("container") if record else None,
        anomaly_type=diagnosis.get("suspected_fault"),
        diagnosis_status=str(diagnosis.get("status", "unavailable")),
        evidence=evidence,
        relative_deviation=relative_deviation,
        telemetry_timestamp=record.get("timestamp") if record else None,
        telemetry_age_seconds=telemetry_age,
        confirmation_count=confirmation_count,
        cooldown_active=cooldown_active,
        persistence_trusted=persistence_trusted,
        dry_run_decision=decision,
        reason=reason,
    )
    result = asdict(incident)
    result.update({
        "dry_run": True,
        "execution_performed": False,
        "eligible_for_remediation": decision == "eligible",
        "required_confirmations": REQUIRED_CONFIRMATIONS,
    })
    return result


def _evaluate_dry_run_locked() -> dict[str, Any]:
    rows, telemetry_error = live_ai._read_live_rows()
    if telemetry_error:
        return _base_result(
            diagnosis={"status": "unavailable", "evidence": []},
            reason=telemetry_error,
            decision="unavailable",
        )

    diagnosis = live_ai._diagnose(rows, include_evidence=True)
    evidence_record = diagnosis.get("_evidence_record")
    if diagnosis.get("status") != "anomalous":
        return _base_result(
            diagnosis=diagnosis,
            reason="Diagnosis is not anomalous; no remediation is eligible.",
            decision="not_eligible",
        )
    if not isinstance(evidence_record, dict):
        return _base_result(
            diagnosis=diagnosis,
            reason="Anomalous diagnosis did not include exact telemetry evidence.",
            decision="blocked",
        )
    evidence_valid, evidence_error = _validate_evidence(
        rows,
        diagnosis,
        evidence_record,
    )
    if not evidence_valid:
        return _base_result(
            diagnosis=diagnosis,
            record=evidence_record,
            reason=evidence_error or "Evidence validation failed.",
            decision="blocked",
        )

    timestamp_data = _canonical_timestamp(evidence_record.get("timestamp"))
    if timestamp_data is None:
        return _base_result(
            diagnosis=diagnosis,
            record=evidence_record,
            reason="Telemetry timestamp is invalid.",
            decision="blocked",
        )
    canonical_timestamp, timestamp = timestamp_data
    age = (datetime.now(timezone.utc) - timestamp).total_seconds()
    if age < 0:
        return _base_result(
            diagnosis=diagnosis,
            record=evidence_record,
            reason="Future telemetry cannot be considered for remediation.",
            decision="blocked",
            telemetry_age=age,
        )
    if age > live_ai.MAX_TELEMETRY_AGE_SECONDS:
        return _base_result(
            diagnosis=diagnosis,
            record=evidence_record,
            reason="Telemetry is stale and cannot be considered for remediation.",
            decision="blocked",
            telemetry_age=age,
        )

    namespace = evidence_record.get("namespace")
    service = evidence_record.get("service")
    pod = evidence_record.get("pod")
    container = evidence_record.get("container")
    fault = diagnosis.get("suspected_fault")
    if namespace != "default":
        reason = "Only the default namespace is eligible for remediation consideration."
    elif service not in APPROVED_SERVICES:
        reason = "Service is not on the approved remediation allowlist."
    elif not pod or not container:
        reason = "Pod and container identity are required for safe confirmation."
    elif fault not in {"CPU", "MEM"}:
        reason = "Diagnosis did not identify a supported CPU or memory anomaly."
    else:
        reason = None

    incident_id = (
        _incident_id(namespace, service, pod, container, fault)
        if all(isinstance(value, str) and value for value in
               (namespace, service, pod, container, fault))
        else None
    )
    if reason is not None:
        return _base_result(
            diagnosis=diagnosis,
            record=evidence_record,
            incident_id=incident_id,
            reason=reason,
            decision="blocked",
            telemetry_age=age,
        )

    trusted, persistence_error = _load_persistence_state(
        force_dry_run_reload=True,
    )
    if not trusted:
        return _base_result(
            diagnosis=diagnosis,
            record=evidence_record,
            incident_id=incident_id,
            reason=persistence_error or "Dry-run persistence is untrusted.",
            decision="blocked",
            persistence_trusted=False,
            telemetry_age=age,
        )

    samples = _seen_samples.setdefault(incident_id, set())
    if canonical_timestamp in samples:
        return _base_result(
            diagnosis=diagnosis,
            record=evidence_record,
            incident_id=incident_id,
            reason="This validated telemetry sample was already processed.",
            decision="duplicate",
            confirmation_count=_confirmation_counts.get(incident_id, 0),
            persistence_trusted=True,
            telemetry_age=age,
        )

    previous_restart = float(controller.last_restart.get(service, 0))
    cooldown_active = time.time() - previous_restart < controller.COOLDOWN_SECONDS
    _incident_identities[incident_id] = _incident_identity(
        namespace, service, pod, container, fault
    )
    samples.add(canonical_timestamp)
    if cooldown_active:
        _last_sample_timestamps[incident_id] = timestamp
        if not _save_dry_run_state():
            return _base_result(
                diagnosis=diagnosis,
                record=evidence_record,
                incident_id=incident_id,
                reason=_state_error or "Dry-run persistence failed.",
                decision="blocked",
                cooldown_active=True,
                persistence_trusted=False,
                telemetry_age=age,
            )
        return _base_result(
            diagnosis=diagnosis,
            record=evidence_record,
            incident_id=incident_id,
            reason="Service restart cooldown is active.",
            decision="blocked",
            confirmation_count=_confirmation_counts.get(incident_id, 0),
            cooldown_active=True,
            persistence_trusted=True,
            telemetry_age=age,
        )

    previous_timestamp = _last_sample_timestamps.get(incident_id)
    if previous_timestamp is not None and timestamp <= previous_timestamp:
        _save_dry_run_state()
        return _base_result(
            diagnosis=diagnosis,
            record=evidence_record,
            incident_id=incident_id,
            reason="Telemetry samples must advance chronologically.",
            decision="duplicate",
            confirmation_count=_confirmation_counts.get(incident_id, 0),
            persistence_trusted=_state_trusted,
            telemetry_age=age,
        )

    _last_sample_timestamps[incident_id] = timestamp
    count = _confirmation_counts.get(incident_id, 0) + 1
    _confirmation_counts[incident_id] = count
    _confirmation_samples.setdefault(incident_id, set()).add(
        canonical_timestamp
    )
    if not _save_dry_run_state():
        return _base_result(
            diagnosis=diagnosis,
            record=evidence_record,
            incident_id=incident_id,
            reason=_state_error or "Dry-run persistence failed.",
            decision="blocked",
            confirmation_count=count,
            cooldown_active=False,
            persistence_trusted=False,
            telemetry_age=age,
        )

    if count < REQUIRED_CONFIRMATIONS:
        decision = "confirmation_pending"
        reason = (
            f"Confirmation pending: {count}/{REQUIRED_CONFIRMATIONS} "
            "distinct anomaly samples."
        )
    else:
        decision = "eligible"
        reason = (
            "Dry-run eligible: required distinct anomaly confirmations, "
            "allowlist, freshness and persistence checks passed."
        )
    return _base_result(
        diagnosis=diagnosis,
        record=evidence_record,
        incident_id=incident_id,
        reason=reason,
        decision=decision,
        confirmation_count=count,
        cooldown_active=False,
        persistence_trusted=True,
        telemetry_age=age,
    )


def evaluate_dry_run() -> dict[str, Any]:
    """Evaluate the current live diagnosis without executing remediation."""

    with _STATE_LOCK:
        try:
            with _process_coordination():
                return _evaluate_dry_run_locked()
        except Exception as error:
            return _base_result(
                diagnosis={"status": "unavailable", "evidence": []},
                reason=f"Dry-run coordination failed: {error}",
                decision="blocked",
            )


def get_dry_run_status() -> dict[str, Any]:
    """Return the read-only API representation of the dry-run decision."""

    result = evaluate_dry_run()
    result["mode"] = "DRY RUN — NO ACTION EXECUTED"
    return result


__all__ = [
    "DRY_RUN_STATE_FILE",
    "REQUIRED_CONFIRMATIONS",
    "evaluate_dry_run",
    "get_dry_run_status",
    "reset_dry_run_state",
]
