from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import asyncio
import json

import pytest

from src.api import main
from src.telemetry import live_remediation
from src.telemetry import remediation_authorization as authorization


IDENTITY = {
    "service": "frontend",
    "pod": "frontend-1",
    "container": "server",
    "anomaly_type": "CPU",
    "proposed_action": "restart_service",
}


def expiry(seconds=60):
    return (
        datetime.now(timezone.utc) + timedelta(seconds=seconds)
    ).isoformat().replace("+00:00", "Z")


@pytest.fixture(autouse=True)
def isolated_state(monkeypatch, tmp_path):
    monkeypatch.setattr(
        authorization,
        "AUTHORIZATION_STATE_FILE",
        tmp_path / "authorizations.json",
    )
    monkeypatch.setattr(
        authorization,
        "AUTHORIZATION_DB_FILE",
        tmp_path / "authorizations.sqlite3",
    )
    authorization.reset_authorization_state()


def approve(**overrides):
    values = {
        **IDENTITY,
        "incident_id": live_remediation._incident_id(
            "default",
            IDENTITY["service"],
            IDENTITY["pod"],
            IDENTITY["container"],
            IDENTITY["anomaly_type"],
        ),
        "expires_at": expiry(),
        "operator_id": "operator-1",
    }
    values.update(overrides)
    return authorization.approve_authorization(**values)


def consume(approved, **overrides):
    values = {
        **IDENTITY,
        "authorization_token": approved["authorization_token"],
        "authorization_id": approved["authorization_id"],
        "incident_id": live_remediation._incident_id(
            "default",
            IDENTITY["service"],
            IDENTITY["pod"],
            IDENTITY["container"],
            IDENTITY["anomaly_type"],
        ),
    }
    values.update(overrides)
    return authorization.consume_authorization(**values)


def test_approval_returns_server_generated_non_executable_capability():
    result = approve()

    assert result["authorization_token"]
    assert result["dry_run"] is True
    assert result["execution_performed"] is False
    state = json.loads(
        authorization.AUTHORIZATION_STATE_FILE.read_text(encoding="utf-8")
    )
    entry = state["authorizations"][result["authorization_id"]]
    assert entry["token_digest"] != result["authorization_token"]


def test_forged_token_and_mismatched_binding_are_rejected():
    approved = approve()

    with pytest.raises(ValueError, match="token"):
        consume(approved, authorization_token="forged")
    with pytest.raises(ValueError, match="binding"):
        consume(approved, pod="other-pod")


def test_expired_authorization_is_rejected():
    with pytest.raises(ValueError, match="expired"):
        approve(expires_at=expiry(-1))


def test_authorization_is_single_use():
    approved = approve()
    result = consume(approved)

    assert result["consumed"] is True
    assert result["action_executed"] is False
    with pytest.raises(ValueError, match="already been used"):
        consume(approved)


def test_persistence_failure_fails_closed(monkeypatch):
    monkeypatch.setattr(
        authorization,
        "_save_state",
        lambda state: (_ for _ in ()).throw(OSError("write failed")),
    )

    with pytest.raises(RuntimeError, match="persistence failed"):
        approve()
    assert not authorization.AUTHORIZATION_STATE_FILE.exists()


def test_concurrent_consumption_is_single_use():
    approved = approve()

    def attempt():
        try:
            return consume(approved)["consumed"]
        except (ValueError, RuntimeError):
            return False

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: attempt(), range(2)))

    assert results.count(True) == 1


def test_api_authorization_and_consumption_routes_are_non_executing():
    incident_id = live_remediation._incident_id(
        "default",
        IDENTITY["service"],
        IDENTITY["pod"],
        IDENTITY["container"],
        IDENTITY["anomaly_type"],
    )
    request = main.RemediationAuthorizationRequest(
        **{
            **IDENTITY,
            "incident_id": incident_id,
            "expires_at": expiry(),
            "operator_id": "operator-1",
        }
    )
    approved = main.remediation_authorize(request)
    consumed = main.remediation_consume(
        main.RemediationConsumptionRequest(
            **{
                **IDENTITY,
                "authorization_token": approved["authorization_token"],
                "authorization_id": approved["authorization_id"],
                "incident_id": incident_id,
            }
        )
    )

    assert consumed["dry_run"] is True
    assert consumed["execution_performed"] is False
    assert consumed["action_executed"] is False


def test_api_route_registration():
    paths = {
        route.path
        for route in main.app.routes
        if hasattr(route, "path")
    }
    assert "/api/remediation/authorize" in paths
    assert "/api/remediation/consume-authorization" in paths
