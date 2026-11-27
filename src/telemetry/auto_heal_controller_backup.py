import csv
import time
import statistics
import sys

from pathlib import Path
from collections import defaultdict

PROJECT = Path("/mnt/c/CloudHeal")
CSV_FILE = PROJECT / "data/live/telemetry.csv"

# Import the existing remediation function
sys.path.insert(0, str(PROJECT / "src/remediation"))
from remediation import restart_service

BASELINE_SAMPLES = 5
CHECK_INTERVAL = 10

# Safety settings
REQUIRED_ANOMALIES = 3
COOLDOWN_SECONDS = 300

# Only these services can be automatically restarted
APPROVED_SERVICES = {
    "adservice",
    "cartservice",
    "checkoutservice",
    "currencyservice",
    "emailservice",
    "frontend",
    "productcatalogservice",
    "recommendationservice",
    "paymentservice",
    "shippingservice",
    "redis-cart",
    "loadgenerator"
}

consecutive_anomalies = defaultdict(int)
last_restart = {}


def get_latest_metrics():
    with CSV_FILE.open(newline="") as file:
        rows = list(csv.DictReader(file))

    services = defaultdict(list)

    for row in rows:
        if row["namespace"] == "default":
            services[row["service"]].append(row)

    return services


def check_service(service, records):
    if len(records) <= BASELINE_SAMPLES:
        print(f"{service}: Collecting baseline...")
        consecutive_anomalies[service] = 0
        return

    baseline = records[:BASELINE_SAMPLES]
    current = records[-1]

    baseline_cpu = [
        float(row["cpu_millicores"])
        for row in baseline
    ]

    cpu_mean = statistics.mean(baseline_cpu)
    cpu_std = statistics.pstdev(baseline_cpu)

    current_cpu = float(current["cpu_millicores"])

    cpu_limit = cpu_mean + max(3 * cpu_std, 5)

    if current_cpu > cpu_limit:
        consecutive_anomalies[service] += 1

        print(
            f"ANOMALY: {service} | "
            f"CPU: {current_cpu:.2f}m | "
            f"Limit: {cpu_limit:.2f}m | "
            f"Consecutive readings: "
            f"{consecutive_anomalies[service]}"
        )

        if consecutive_anomalies[service] >= REQUIRED_ANOMALIES:
            handle_remediation(service)

    else:
        consecutive_anomalies[service] = 0

        print(
            f"NORMAL: {service} | "
            f"CPU: {current_cpu:.2f}m"
        )


def handle_remediation(service):
    if service not in APPROVED_SERVICES:
        print(f"SAFETY BLOCK: {service} is not approved.")
        consecutive_anomalies[service] = 0
        return

    now = time.time()
    previous_restart = last_restart.get(service, 0)

    if now - previous_restart < COOLDOWN_SECONDS:
        print(
            f"COOLDOWN: {service} cannot be restarted yet."
        )
        return

    print("\nAUTOMATIC REMEDIATION TRIGGERED")
    print("--------------------------------")
    print(f"Service: {service}")
    print("Reason: Persistent HIGH CPU anomaly")

    # Record the restart time before starting remediation
    # to prevent repeated attempts if the restart fails.
    last_restart[service] = now
    consecutive_anomalies[service] = 0

    restart_service(service)


def main():
    print("\nCLOUDHEAL AUTOMATIC SELF-HEALING")
    print("--------------------------------")
    print("Monitoring live Kubernetes telemetry.")
    print("CPU anomalies require 3 consecutive readings.")
    print("Restart cooldown: 300 seconds.")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            services = get_latest_metrics()

            print("\nChecking application services...")
            print("--------------------------------")

            for service, records in sorted(services.items()):
                check_service(service, records)

            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("\nCloudHeal automatic controller stopped.")


if __name__ == "__main__":
    main()
