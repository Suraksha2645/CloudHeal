from pathlib import Path

from src.api import main


def reset_cache():
    main.PREPROCESSING_CACHE = None
    main.PREPROCESSING_CACHE_TIME = 0.0
    main.PREPROCESSING_UPDATED_AT = None


def test_preprocessing_endpoint_returns_summary_and_caches(monkeypatch):
    reset_cache()
    calls = []

    def fake_reader(*args):
        calls.append(str(args[0]))
        return [], {
            "status": "ok",
            "total_rows": 2,
            "valid_records": 1,
            "rejected_records": 1,
            "exact_duplicate_records": 1,
            "missing_field_counts": {"service": 1},
            "invalid_field_counts": {"cpu_millicores": 0},
            "rejection_reasons": {"exact_duplicate": 1},
            "timestamp_range": {"start": "a", "end": "b"},
            "groups": [{"service": "api"}],
            "pod_changes": {"api": {"pod_change_count": 2}},
        }

    monkeypatch.setattr(main, "read_and_validate_telemetry", fake_reader)
    monkeypatch.setattr(main, "TELEMETRY_FILE", Path("test.csv"))

    first = main.telemetry_preprocessing()
    second = main.telemetry_preprocessing()

    assert first["total_rows"] == 2
    assert first["group_count"] == 1
    assert first["service_count"] == 1
    assert first["pod_changes_by_service"] == {"api": 2}
    assert first["last_preprocessing_update"] == second["last_preprocessing_update"]
    assert len(calls) == 1


def test_preprocessing_endpoint_reports_unavailable_file(monkeypatch):
    reset_cache()

    def missing_reader(*args):
        assert args
        return (
            [],
            {
                "status": "missing",
                "total_rows": 0,
                "valid_records": 0,
                "rejected_records": 0,
                "exact_duplicate_records": 0,
                "missing_field_counts": {},
                "invalid_field_counts": {},
                "rejection_reasons": {},
                "timestamp_range": {"start": None, "end": None},
                "groups": [],
                "pod_changes": {},
            },
        )

    monkeypatch.setattr(main, "read_and_validate_telemetry", missing_reader)

    response = main.telemetry_preprocessing()

    assert response["status"] == "missing"
    assert response["total_rows"] == 0
    assert response["group_count"] == 0
