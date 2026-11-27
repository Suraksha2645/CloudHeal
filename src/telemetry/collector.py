import time
import json
import os
import subprocess
import csv
from datetime import datetime, timezone
from pathlib import Path
import threading


PROJECT_ROOT = Path("/mnt/c/CloudHeal")
OUTPUT_FILE = PROJECT_ROOT / "data" / "live" / "telemetry.csv"
LOCK_FILE = PROJECT_ROOT / "data" / "live" / "telemetry_collector.lock"
COLLECTION_INTERVAL_SECONDS = 10


class TelemetryCollector:
    def __init__(self):
        self._stop_event = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._running = False
        self._last_successful_collection = None
        self._records_collected = 0
        self._last_error = None
        self._lock_file_descriptor = None

    def start(self):
        with self._lock:
            if self._thread and self._thread.is_alive():
                return False

            if not self._acquire_process_lock():
                return False

            self._stop_event.clear()
            self._running = True
            try:
                self._last_error = None
                self._thread = threading.Thread(
                    target=self._run,
                    name="cloudheal-telemetry-collector",
                    daemon=True,
                )
                self._thread.start()
            except Exception:
                self._release_process_lock()
                self._running = False
                self._thread = None
                raise

            return True

    def stop(self):
        with self._lock:
            thread = self._thread
            self._stop_event.set()

        if thread and thread.is_alive():
            thread.join(timeout=2)

        with self._lock:
            self._running = False
            self._thread = None

    def status(self):
        with self._lock:
            running = bool(
                self._thread and self._thread.is_alive()
            )
            file_available = OUTPUT_FILE.is_file()

            return {
                "running": running,
                "last_successful_collection": (
                    self._last_successful_collection
                ),
                "records_collected": self._records_collected,
                "telemetry_file_available": file_available,
                "telemetry_file": str(OUTPUT_FILE),
                "last_error": self._last_error,
            }

    def _run(self):
        print("CloudHeal live telemetry started.")
        print("Collecting metrics every 10 seconds.")

        try:
            while not self._stop_event.is_set():
                try:
                    records = collect_metrics()
                    save_metrics(records)

                    with self._lock:
                        self._records_collected += len(records)
                        self._last_successful_collection = (
                            datetime.now(timezone.utc).isoformat()
                        )
                        self._last_error = None

                    print(
                        f"[{datetime.now().strftime('%H:%M:%S')}] "
                        f"Collected {len(records)} container metrics."
                    )

                except Exception as error:
                    with self._lock:
                        self._last_error = str(error)

                    print("Telemetry collection error:", error)

                self._stop_event.wait(COLLECTION_INTERVAL_SECONDS)

        finally:
            with self._lock:
                self._running = False
                self._release_process_lock()

            print("CloudHeal telemetry collection stopped.")

    def _acquire_process_lock(self):
        LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._lock_file_descriptor = os.open(
                LOCK_FILE,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
            os.write(
                self._lock_file_descriptor,
                str(os.getpid()).encode("ascii"),
            )
            return True
        except FileExistsError:
            try:
                owner_pid = int(LOCK_FILE.read_text().strip())
                os.kill(owner_pid, 0)
            except (OSError, ValueError):
                LOCK_FILE.unlink(missing_ok=True)
                return self._acquire_process_lock()

            return False

    def _release_process_lock(self):
        if self._lock_file_descriptor is not None:
            os.close(self._lock_file_descriptor)
            self._lock_file_descriptor = None
            LOCK_FILE.unlink(missing_ok=True)


telemetry_collector = TelemetryCollector()


def convert_cpu(value):
    """Convert Kubernetes CPU units into millicores."""
    if value.endswith("n"):
        return float(value[:-1]) / 1_000_000

    if value.endswith("u"):
        return float(value[:-1]) / 1_000

    if value.endswith("m"):
        return float(value[:-1])

    return float(value) * 1000


def convert_memory(value):
    """Convert Kubernetes memory units into MiB."""
    units = {
        "Ki": 1 / 1024,
        "Mi": 1,
        "Gi": 1024,
        "Ti": 1024 * 1024,
    }

    for unit, factor in units.items():
        if value.endswith(unit):
            return float(value[:-len(unit)]) * factor

    return float(value) / (1024 * 1024)


def collect_metrics():
    command = [
        "kubectl",
        "get",
        "--raw",
        "/apis/metrics.k8s.io/v1beta1/pods"
    ]
    command_environment = os.environ.copy()
    command_environment["KUBECONFIG"] = "/home/cloudheal/.kube/config"

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            env=command_environment,
        )
    except subprocess.CalledProcessError as error:
        error_detail = (error.stderr or "").strip()
        if error_detail:
            raise RuntimeError(
                f"kubectl metrics request failed: {error_detail}"
            ) from error
        raise RuntimeError(
            f"kubectl metrics request failed with exit code {error.returncode}"
        ) from error

    metrics = json.loads(result.stdout)
    records = []

    for pod in metrics.get("items", []):
        pod_name = pod["metadata"]["name"]
        namespace = pod["metadata"]["namespace"]

        service = pod["metadata"].get("labels", {}).get(
            "app", pod_name
        )

        for container in pod.get("containers", []):
            usage = container["usage"]

            records.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "namespace": namespace,
                "service": service,
                "pod": pod_name,
                "container": container["name"],
                "cpu_millicores": convert_cpu(usage["cpu"]),
                "memory_mib": convert_memory(usage["memory"])
            })

    return records


def save_metrics(records):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "timestamp",
        "namespace",
        "service",
        "pod",
        "container",
        "cpu_millicores",
        "memory_mib"
    ]

    file_exists = OUTPUT_FILE.exists()

    with OUTPUT_FILE.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)

        if not file_exists:
            writer.writeheader()

        writer.writerows(records)


if __name__ == "__main__":
    print("Press Ctrl+C to stop.\n")
    telemetry_collector.start()

    try:
        while telemetry_collector.status()["running"]:
            time.sleep(1)
    except KeyboardInterrupt:
        telemetry_collector.stop()
