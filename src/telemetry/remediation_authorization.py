"""Non-executing authorization capabilities for a future remediation stage."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import secrets
import sqlite3
import threading
from contextlib import contextmanager
from typing import Any

from src.telemetry import live_remediation


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUTHORIZATION_STATE_FILE = (
    PROJECT_ROOT / "data" / "live" / "remediation_authorizations.json"
)
AUTHORIZATION_DB_FILE = (
    PROJECT_ROOT / "data" / "live" / "remediation_authorizations.sqlite3"
)
AUTHORIZATION_VERSION = 1
MAX_AUTHORIZATION_SECONDS = 300
SUPPORTED_ACTION = "restart_service"
_AUTHORIZATION_LOCK = threading.RLock()


def _utc_timestamp(value: Any) -> tuple[str, datetime] | None:
    parsed = live_remediation._canonical_timestamp(value)
    if parsed is None:
        return None
    return parsed


def _token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _authorization_id(
    incident_id: str,
    service: str,
    pod: str,
    container: str,
    anomaly_type: str,
    proposed_action: str,
) -> str:
    value = "|".join(
        (incident_id, service, pod, container, anomaly_type, proposed_action)
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _identity_matches(
    incident_id: str,
    service: str,
    pod: str,
    container: str,
    anomaly_type: str,
) -> bool:
    expected = live_remediation._incident_id(
        "default", service, pod, container, anomaly_type
    )
    return incident_id == expected


@contextmanager
def _coordination():
    AUTHORIZATION_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(AUTHORIZATION_DB_FILE), timeout=2)
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


def _load_state() -> dict[str, Any]:
    if not AUTHORIZATION_STATE_FILE.exists():
        return {"version": AUTHORIZATION_VERSION, "authorizations": {}}
    with AUTHORIZATION_STATE_FILE.open(encoding="utf-8") as file:
        state = json.load(file)
    if (
        not isinstance(state, dict)
        or state.get("version") != AUTHORIZATION_VERSION
        or not isinstance(state.get("authorizations"), dict)
    ):
        raise ValueError("authorization state is invalid")
    return state


def _save_state(state: dict[str, Any]) -> None:
    temporary = AUTHORIZATION_STATE_FILE.with_suffix(".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(state, file, indent=2, sort_keys=True)
        temporary.replace(AUTHORIZATION_STATE_FILE)
        with AUTHORIZATION_STATE_FILE.open(encoding="utf-8") as file:
            if json.load(file) != state:
                raise ValueError("authorization state verification failed")
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def approve_authorization(
    *,
    incident_id: str,
    service: str,
    pod: str,
    container: str,
    anomaly_type: str,
    proposed_action: str,
    expires_at: str,
    operator_id: str,
) -> dict[str, Any]:
    """Create a single-use capability without executing an action."""

    now = datetime.now(timezone.utc)
    expiry = _utc_timestamp(expires_at)
    if not all(
        isinstance(value, str) and value
        for value in (
            incident_id,
            service,
            pod,
            container,
            anomaly_type,
            proposed_action,
            operator_id,
        )
    ):
        raise ValueError("authorization fields are required")
    if service not in live_remediation.APPROVED_SERVICES:
        raise ValueError("service is not approved")
    if anomaly_type not in {"CPU", "MEM"}:
        raise ValueError("anomaly type is unsupported")
    if proposed_action != SUPPORTED_ACTION:
        raise ValueError("proposed action is unsupported")
    if not _identity_matches(
        incident_id, service, pod, container, anomaly_type
    ):
        raise ValueError("incident identity does not match")
    if expiry is None:
        raise ValueError("expiry timestamp is invalid")
    expiry_timestamp = expiry[1]
    if expiry_timestamp <= now:
        raise ValueError("authorization is already expired")
    if (expiry_timestamp - now).total_seconds() > MAX_AUTHORIZATION_SECONDS:
        raise ValueError("authorization expiry is too far in the future")

    authorization_id = _authorization_id(
        incident_id,
        service,
        pod,
        container,
        anomaly_type,
        proposed_action,
    )
    token = secrets.token_urlsafe(32)
    state_entry = {
        "authorization_id": authorization_id,
        "token_digest": _token_digest(token),
        "incident_id": incident_id,
        "service": service,
        "pod": pod,
        "container": container,
        "anomaly_type": anomaly_type,
        "proposed_action": proposed_action,
        "expires_at": expiry[0],
        "operator_id": operator_id,
        "used": False,
    }
    with _AUTHORIZATION_LOCK:
        try:
            with _coordination():
                state = _load_state()
                authorizations = state["authorizations"]
                if authorization_id in authorizations:
                    raise ValueError("authorization already exists")
                authorizations[authorization_id] = state_entry
                _save_state(state)
        except ValueError:
            raise
        except (OSError, sqlite3.Error) as error:
            raise RuntimeError(
                f"authorization persistence failed: {error}"
            ) from error
    return {
        "authorization_id": authorization_id,
        "authorization_token": token,
        "expires_at": expiry[0],
        "dry_run": True,
        "execution_performed": False,
    }


def consume_authorization(
    *,
    authorization_token: str,
    authorization_id: str,
    incident_id: str,
    service: str,
    pod: str,
    container: str,
    anomaly_type: str,
    proposed_action: str,
) -> dict[str, Any]:
    """Atomically consume a capability; this function never executes it."""

    if not isinstance(authorization_token, str) or not authorization_token:
        raise ValueError("authorization token is required")
    with _AUTHORIZATION_LOCK:
        try:
            with _coordination():
                state = _load_state()
                entry = state["authorizations"].get(authorization_id)
                if not isinstance(entry, dict):
                    raise ValueError("authorization was not found")
                if entry.get("token_digest") != _token_digest(
                    authorization_token
                ):
                    raise ValueError("authorization token is invalid")
                expected = {
                    "incident_id": incident_id,
                    "service": service,
                    "pod": pod,
                    "container": container,
                    "anomaly_type": anomaly_type,
                    "proposed_action": proposed_action,
                }
                if any(entry.get(key) != value for key, value in expected.items()):
                    raise ValueError("authorization binding does not match")
                if entry.get("used") is True:
                    raise ValueError("authorization has already been used")
                expiry = _utc_timestamp(entry.get("expires_at"))
                if expiry is None or expiry[1] <= datetime.now(timezone.utc):
                    raise ValueError("authorization has expired")
                entry["used"] = True
                _save_state(state)
        except ValueError:
            raise
        except (OSError, sqlite3.Error) as error:
            raise RuntimeError(
                f"authorization consumption failed: {error}"
            ) from error
    return {
        "authorization_id": authorization_id,
        "consumed": True,
        "dry_run": True,
        "execution_performed": False,
        "action_executed": False,
    }


def reset_authorization_state() -> None:
    with _AUTHORIZATION_LOCK:
        for path in (AUTHORIZATION_STATE_FILE, AUTHORIZATION_DB_FILE):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
