import csv
import math

from src.telemetry.live_preprocessing import read_and_validate_telemetry


FIELDS = [
    "timestamp",
    "namespace",
    "service",
    "pod",
    "container",
    "cpu_millicores",
    "memory_mib",
]


def write_rows(path, rows, fields=FIELDS):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [{field: current.get(field, "") for field in fields} for current in rows]
        )


def row(timestamp="2026-11-05T10:00:00+00:00", pod="api-1", **values):
    result = {
        "timestamp": timestamp,
        "namespace": "default",
        "service": "api",
        "pod": pod,
        "container": "server",
        "cpu_millicores": "2.5",
        "memory_mib": "10.0",
    }
    result.update(values)
    return result


def test_valid_record_and_group_summary(tmp_path):
    path = tmp_path / "telemetry.csv"
    write_rows(path, [row()])

    records, summary = read_and_validate_telemetry(path)

    assert len(records) == 1
    assert records[0]["cpu_millicores"] == 2.5
    assert records[0]["memory_mib"] == 10.0
    assert summary["valid_records"] == 1
    assert summary["rejected_records"] == 0
    assert summary["groups"][0]["record_count"] == 1


def test_missing_required_fields_are_reported(tmp_path):
    path = tmp_path / "telemetry.csv"
    missing = row()
    missing["service"] = ""
    missing["memory_mib"] = ""
    write_rows(path, [missing])

    records, summary = read_and_validate_telemetry(path)

    assert records == []
    assert summary["missing_field_counts"]["service"] == 1
    assert summary["missing_field_counts"]["memory_mib"] == 1
    assert summary["invalid_field_counts"]["memory_mib"] == 0
    assert summary["rejection_reasons"]["missing_required_field"] == 1


def test_invalid_timestamp_is_rejected(tmp_path):
    path = tmp_path / "telemetry.csv"
    write_rows(path, [row(timestamp="not-a-timestamp")])

    records, summary = read_and_validate_telemetry(path)

    assert records == []
    assert summary["invalid_field_counts"]["timestamp"] == 1


def test_non_numeric_measurements_are_rejected(tmp_path):
    path = tmp_path / "telemetry.csv"
    write_rows(path, [row(cpu_millicores="bad", memory_mib="also-bad")])

    records, summary = read_and_validate_telemetry(path)

    assert records == []
    assert summary["invalid_field_counts"]["cpu_millicores"] == 1
    assert summary["invalid_field_counts"]["memory_mib"] == 1


def test_negative_and_non_finite_measurements_are_rejected(tmp_path):
    path = tmp_path / "telemetry.csv"
    write_rows(
        path,
        [
            row(cpu_millicores="-1"),
            row(
                timestamp="2026-11-05T10:00:01+00:00",
                memory_mib=str(math.inf),
            ),
        ],
    )

    records, summary = read_and_validate_telemetry(path)

    assert records == []
    assert summary["invalid_field_counts"]["cpu_millicores"] == 1
    assert summary["invalid_field_counts"]["memory_mib"] == 1


def test_exact_duplicate_is_rejected(tmp_path):
    path = tmp_path / "telemetry.csv"
    duplicate = row()
    write_rows(path, [duplicate, duplicate])

    records, summary = read_and_validate_telemetry(path)

    assert len(records) == 1
    assert summary["exact_duplicate_records"] == 1
    assert summary["rejected_records"] == 1


def test_valid_records_are_sorted_chronologically(tmp_path):
    path = tmp_path / "telemetry.csv"
    write_rows(
        path,
        [
            row(timestamp="2026-11-05T10:00:02+00:00"),
            row(timestamp="2026-11-05T10:00:01+00:00"),
        ],
    )

    records, summary = read_and_validate_telemetry(path)

    assert [record["timestamp"] for record in records] == [
        "2026-11-05T10:00:01+00:00",
        "2026-11-05T10:00:02+00:00",
    ]
    assert summary["timestamp_range"]["start"].startswith(
        "2026-11-05T10:00:01"
    )


def test_pod_identity_changes_are_reported_per_service(tmp_path):
    path = tmp_path / "telemetry.csv"
    write_rows(
        path,
        [
            row(timestamp="2026-11-05T10:00:00+00:00", pod="api-old"),
            row(timestamp="2026-11-05T10:00:10+00:00", pod="api-new"),
        ],
    )

    _, summary = read_and_validate_telemetry(path)

    changes = summary["pod_changes"]["api"]
    assert changes["distinct_pods"] == ["api-old", "api-new"]
    assert changes["pod_change_count"] == 1
    assert changes["changes"][0]["from_pod"] == "api-old"
    assert changes["changes"][0]["to_pod"] == "api-new"


def test_missing_and_empty_csv_are_safe(tmp_path):
    missing_records, missing_summary = read_and_validate_telemetry(
        tmp_path / "missing.csv"
    )
    empty_path = tmp_path / "empty.csv"
    empty_path.touch()
    empty_records, empty_summary = read_and_validate_telemetry(empty_path)

    assert missing_records == []
    assert missing_summary["status"] == "missing"
    assert empty_records == []
    assert empty_summary["status"] == "empty"


def test_missing_csv_columns_are_reported(tmp_path):
    path = tmp_path / "telemetry.csv"
    write_rows(path, [row()], fields=["timestamp", "service"])

    _, summary = read_and_validate_telemetry(path)

    assert summary["status"] == "invalid_schema"
    assert "cpu_millicores" in summary["missing_columns"]
