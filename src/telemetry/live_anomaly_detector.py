
import csv
import time
import statistics
import json

from pathlib import Path
from collections import defaultdict
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


PROJECT = Path("/mnt/c/CloudHeal")
CSV_FILE = PROJECT / "data/live/telemetry.csv"

BASELINE_SAMPLES = 5
CHECK_INTERVAL = 10

# Dedicated threshold for the injected CPU test workload.
NEW_WORKLOAD_CPU_THRESHOLD = 100.0

# More conservative threshold for normal application services.
NORMAL_SERVICE_MIN_CPU_INCREASE = 15.0
NORMAL_SERVICE_STD_MULTIPLIER = 5

API_URL = "http://localhost:8000"


def is_cpu_test_running():
    """Check whether the injected CPU test pod is running."""

    try:
        with urlopen(
            f"{API_URL}/api/anomalies/cpu/status",
            timeout=10
        ) as response:
            status = json.loads(response.read().decode())

        return status.get("running", False)

    except Exception as error:
        print("[STATUS CHECK] Could not verify CPU test pod:", error)
        return False


def trigger_auto_healing():
    """Trigger recovery for the injected CPU test workload."""

    print("\n[AUTO-HEALING] CPU anomaly detected.")
    print("[AUTO-HEALING] Checking test workload status...")

    try:
        if not is_cpu_test_running():
            print(
                "[AUTO-HEALING] Test workload is no longer running. "
                "Skipping duplicate healing."
            )
            return

        print("[AUTO-HEALING] Test workload is running.")
        print("[AUTO-HEALING] Starting recovery...")

        request = Request(
            f"{API_URL}/api/healing/execute",
            data=b"",
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urlopen(request, timeout=20) as response:
            result = json.loads(response.read().decode())

        print("\n[AUTO-HEALING RESULT]")
        print("Healing:", result.get("healing"))
        print("Action:", result.get("action"))
        print("Application healthy:", result.get("application_healthy"))
        print("Recovery verified:", result.get("recovery_verified"))

        if result.get("recovery_verified"):
            print("[AUTO-HEALING] Recovery successful!")
        else:
            print("[AUTO-HEALING] Recovery could not be verified.")

    except (URLError, HTTPError, TimeoutError) as error:
        print("[AUTO-HEALING] API connection error:", error)

    except Exception as error:
        print("[AUTO-HEALING] Error:", error)


def detect_anomalies():

    if not CSV_FILE.exists():
        print("Telemetry file not found.")
        return

    with CSV_FILE.open(newline="") as file:
        rows = list(csv.DictReader(file))

    services = defaultdict(list)

    for row in rows:
        if row["namespace"] == "default":
            services[row["service"]].append(row)

    print("\nCLOUDHEAL LIVE ANOMALY DETECTION")
    print("-" * 45)

    for service, records in sorted(services.items()):

        if len(records) <= BASELINE_SAMPLES:
            print(f"{service}: Collecting baseline...")
            continue

        current = records[-1]
        baseline = records[:BASELINE_SAMPLES]

        # Ignore old records from a deleted CPU test workload.
        if current["pod"] == "cloudheal-cpu-test":
            if not is_cpu_test_running():
                print(
                    "STALE TELEMETRY: CPU test pod is no longer running."
                )
                continue

        baseline_cpu = [
            float(record["cpu_millicores"])
            for record in baseline
        ]

        baseline_memory = [
            float(record["memory_mib"])
            for record in baseline
        ]

        cpu_mean = statistics.mean(baseline_cpu)
        memory_mean = statistics.mean(baseline_memory)

        cpu_std = statistics.pstdev(baseline_cpu)
        memory_std = statistics.pstdev(baseline_memory)

        current_cpu = float(current["cpu_millicores"])
        current_memory = float(current["memory_mib"])

        # Use a more conservative threshold for normal services.
        cpu_limit = cpu_mean + max(
            NORMAL_SERVICE_STD_MULTIPLIER * cpu_std,
            NORMAL_SERVICE_MIN_CPU_INCREASE
        )

        memory_limit = memory_mean + max(3 * memory_std, 5)

        alerts = []

        # Keep dedicated detection for the injected CPU workload.
        if current["pod"] == "cloudheal-cpu-test":
            if current_cpu > NEW_WORKLOAD_CPU_THRESHOLD:
                alerts.append("HIGH CPU")

        elif current_cpu > cpu_limit:
            alerts.append("HIGH CPU")

        if current_memory > memory_limit:
            alerts.append("HIGH MEMORY")

        status = ", ".join(alerts) if alerts else "NORMAL"

        print(
            f"{'ANOMALY' if alerts else 'NORMAL'}: "
            f"{service} - {status}"
        )

        print(f"  Pod: {current['pod']}")
        print(
            f"  CPU: {current_cpu:.2f}m "
            f"(baseline {cpu_mean:.2f}m)"
        )
        print(
            f"  Memory: {current_memory:.2f}MiB "
            f"(baseline {memory_mean:.2f}MiB)"
        )

        # Automatically heal only the injected CPU test workload.
        if (
            current["pod"] == "cloudheal-cpu-test"
            and current_cpu > NEW_WORKLOAD_CPU_THRESHOLD
        ):
            trigger_auto_healing()
            return


if __name__ == "__main__":

    print("CloudHeal live anomaly detector started.")
    print("Statistical anomaly detection enabled.")
    print("Automatic healing enabled.")
    print("Stale telemetry protection enabled.")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            detect_anomalies()
            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("\nCloudHeal anomaly detector stopped.")
v