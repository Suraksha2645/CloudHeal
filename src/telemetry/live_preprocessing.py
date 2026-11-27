"""Read-only validation and quality reporting for live telemetry CSV data."""

import csv
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = (
    "timestamp",
    "namespace",
    "service",
    "pod",
    "container",
    "cpu_millicores",
    "memory_mib",
)

IDENTITY_COLUMNS = (
    "namespace",
    "service",
    "pod",
    "container",
)

MEASUREMENT_COLUMNS = (
    "cpu_millicores",
    "memory_mib",
)


def _empty_summary(csv_path: Path, status: str) -> dict[str, Any]:
    return {
        "source": str(csv_path),
        "status": status,
        "columns_present": [],
        "missing_columns": [],
        "total_rows": 0,
        "valid_records": 0,
        "rejected_records": 0,
        "exact_duplicate_records": 0,
        "missing_field_counts": {column: 0 for column in REQUIRED_COLUMNS},
        "invalid_field_counts": {column: 0 for column in REQUIRED_COLUMNS},
        "rejection_reasons": {},
        "timestamp_range": {"start": None, "end": None},
        "groups": [],
        "pod_changes": {},
    }


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _is_missing(value: Any) -> bool:
    return value is None or not str(value).strip()


def _format_timestamp(value: datetime) -> str:
    return value.isoformat()


def _group_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for record in records:
        key = tuple(record[column] for column in IDENTITY_COLUMNS)
        grouped[key].append(record)

    return [
        {
            "namespace": key[0],
            "service": key[1],
            "pod": key[2],
            "container": key[3],
            "record_count": len(group_records),
        }
        for key, group_records in sorted(grouped.items())
    ]


def _pod_change_summary(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_service: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_service[record["service"]].append(record)

    result: dict[str, dict[str, Any]] = {}
    for service, service_records in sorted(by_service.items()):
        distinct_pods: list[str] = []
        changes: list[dict[str, str]] = []
        last_pod = None

        for record in service_records:
            pod = record["pod"]
            if pod not in distinct_pods:
                distinct_pods.append(pod)
            if last_pod is not None and pod != last_pod:
                changes.append(
                    {
                        "timestamp": record["timestamp"],
                        "from_pod": last_pod,
                        "to_pod": pod,
                    }
                )
            last_pod = pod

        result[service] = {
            "distinct_pods": distinct_pods,
            "pod_change_count": len(changes),
            "changes": changes,
        }

    return result


def read_and_validate_telemetry(
    csv_path: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read, validate, sort, and summarize a live telemetry CSV.

    The source file is opened only for reading. Invalid rows are excluded from
    the returned records and described in the summary; no values are filled or
    converted beyond validating CPU and memory as floats.
    """

    path = Path(csv_path)
    if not path.is_file():
        return [], _empty_summary(path, "missing")

    records: list[dict[str, Any]] = []
    parsed_timestamps: dict[int, datetime] = {}
    seen_rows: set[tuple[Any, ...]] = set()
    summary = _empty_summary(path, "empty")

    try:
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            fieldnames = reader.fieldnames or []
            summary["columns_present"] = list(fieldnames)
            if not fieldnames:
                return [], summary

            summary["missing_columns"] = [
                column for column in REQUIRED_COLUMNS if column not in fieldnames
            ]

            if summary["missing_columns"]:
                summary["status"] = "invalid_schema"
                return [], summary

            summary["status"] = "ok"

            for row in reader:
                summary["total_rows"] += 1
                raw_values = tuple(row.get(column) for column in fieldnames)
                missing_fields = [
                    column
                    for column in REQUIRED_COLUMNS
                    if _is_missing(row.get(column))
                ]
                invalid_fields: list[str] = []
                rejection_reasons: list[str] = []

                if raw_values in seen_rows:
                    summary["exact_duplicate_records"] += 1
                    rejection_reasons.append("exact_duplicate")
                else:
                    seen_rows.add(raw_values)

                for column in missing_fields:
                    summary["missing_field_counts"][column] += 1

                timestamp = None
                if "timestamp" not in missing_fields:
                    try:
                        timestamp = _parse_timestamp(row["timestamp"])
                    except (TypeError, ValueError, OverflowError):
                        invalid_fields.append("timestamp")

                for column in MEASUREMENT_COLUMNS:
                    if column in missing_fields:
                        continue
                    try:
                        value = float(row[column])
                    except (TypeError, ValueError):
                        invalid_fields.append(column)
                        continue
                    if not math.isfinite(value) or value < 0:
                        invalid_fields.append(column)

                for column in invalid_fields:
                    summary["invalid_field_counts"][column] += 1

                if missing_fields:
                    rejection_reasons.append("missing_required_field")
                if invalid_fields:
                    rejection_reasons.append("invalid_field")

                if rejection_reasons:
                    summary["rejected_records"] += 1
                    for reason in rejection_reasons:
                        summary["rejection_reasons"][reason] = (
                            summary["rejection_reasons"].get(reason, 0) + 1
                        )
                    continue

                record = dict(row)
                record["cpu_millicores"] = float(record["cpu_millicores"])
                record["memory_mib"] = float(record["memory_mib"])
                records.append(record)
                parsed_timestamps[len(records) - 1] = timestamp

    except (OSError, csv.Error) as error:
        summary["status"] = "read_error"
        summary["rejection_reasons"]["read_error"] = str(error)
        return [], summary

    order = sorted(
        range(len(records)),
        key=lambda index: parsed_timestamps[index],
    )
    records = [records[index] for index in order]

    summary["valid_records"] = len(records)
    if records:
        timestamps = [_parse_timestamp(record["timestamp"]) for record in records]
        summary["timestamp_range"] = {
            "start": _format_timestamp(min(timestamps)),
            "end": _format_timestamp(max(timestamps)),
        }
        summary["groups"] = _group_summary(records)
        summary["pod_changes"] = _pod_change_summary(records)

    return records, summary


__all__ = ["REQUIRED_COLUMNS", "read_and_validate_telemetry"]
