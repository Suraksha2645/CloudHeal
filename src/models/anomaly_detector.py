
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

API_URL = "http://localhost:8000"

BASELINE_SAMPLES = 5
CHECK_INTERVAL = 10
NEW_WORKLOAD_CPU_THRESHOLD = 100.0


def trigger_auto_healing():
    """Automatically trigger recovery through the CloudHeal API."""

    try:
        with urlopen(
            f"{API_URL}/api/anomalies/cpu/status",
            timeout=5
        ) as response:
            status = json.loads(response.read().decode())

        if not status.get("running", False):
            print("AUTO-HEALING: Test pod is already stopped.")
            return

        print("\nAUTO-HEALING: CPU anomaly confirmed.")
        print("Starting recovery automatically...")

        request = Request(
            f"{API_URL}/api/healing/execute",
            data=b"",
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urlopen(request, timeout=20) as response:
            result = json.loads(response.read().decode())

        print("\nAUTO-HEALING RESULT")
        print("-" * 45)
        print("Healing:", result.get("healing"))
        print("Pod removed:", result.get("pod_removed"))
        print(
            "Healthy Online Boutique pods:",
            result.get("healthy_online_boutique_pods")
        )
        print("Application healthy:", result.get("application_healthy"))
        print("Recovery verified:", result.get("recovery_verified"))

    except (URLError, HTTPError, TimeoutError) as error:
        print("AUTO-HEALING API ERROR:", error)

    except Exception as error:
        print("AUTO-HEALING ERROR:", error)


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

        baseline = records[:BASELINE_SAMPLES]
        current = records[-1]

        baseline_cpu = [
            float(r["cpu_millicores"]) for r in baseline
        ]
        baseline_memory = [
            float(r["memory_mib"]) for r in baseline
        ]

        cpu_mean = statistics.mean(baseline_cpu)
        memory_mean = statistics.mean(baseline_memory)

        cpu_std = statistics.pstdev(baseline_cpu)
        memory_std = statistics.pstdev(baseline_memory)

        current_cpu = float(current["cpu_millicores"])
        current_memory = float(current["memory_mib"])

        cpu_limit = cpu_mean + max(3 * cpu_std, 5)
        memory_limit = memory_mean + max(3 * memory_std, 5)

        alerts = []

        if (
            current["pod"] == "cloudheal-cpu-test"
            and current_cpu > NEW_WORKLOAD_CPU_THRESHOLD
        ):
            alerts.append("HIGH CPU")
        elif current_cpu > cpu_limit:
            alerts.append("HIGH CPU")

        if current_memory > memory_limit:
            alerts.append("HIGH MEMORY")

        if alerts:
            print(f"ANOMALY: {service} - {', '.join(alerts)}")
        else:
            print(f"NORMAL: {service}")

        print(f"  Pod: {current['pod']}")
        print(
            f"  CPU: {current_cpu:.2f}m "
            f"(baseline {cpu_mean:.2f}m)"
        )
        print(
            f"  Memory: {current_memory:.2f}MiB "
            f"(baseline {memory_mean:.2f}MiB)"
        )

        if (
            current["pod"] == "cloudheal-cpu-test"
            and current_cpu > NEW_WORKLOAD_CPU_THRESHOLD
        ):
            trigger_auto_healing()
            return


if __name__ == "__main__":
    print("CloudHeal live anomaly detector started.")
    print("Using statistical and workload-specific CPU thresholds.")
    print("Automatic healing is enabled.")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            detect_anomalies()
            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("\nCloudHeal anomaly detector stopped.")
