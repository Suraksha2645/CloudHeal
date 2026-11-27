
import json
import math
import time
import statistics
import sys
import threading

from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path("/mnt/c/CloudHeal")
if not (PROJECT / "src" / "remediation").is_dir():
    PROJECT = Path(__file__).resolve().parents[2]

CSV_FILE = PROJECT / "data" / "live" / "telemetry.csv"
STATE_FILE = PROJECT / "data" / "live" / "auto_heal_state.json"

sys.path.insert(0, str(PROJECT / "src"))

from remediation.remediation import restart_service
from telemetry.live_preprocessing import read_and_validate_telemetry


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASELINE_SAMPLES = 5
CHECK_INTERVAL = 5
MAX_SAMPLE_AGE_SECONDS = 30

REQUIRED_ANOMALIES = 3
COOLDOWN_SECONDS = 300

# Safety: real Kubernetes restarts remain disabled.
AUTO_REMEDIATION_ENABLED = True

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
    "loadgenerator",
}


# --------------------------------------------------
# CONTROLLER STATE
# --------------------------------------------------

baselines = {}
current_pods = {}
last_processed = {}
consecutive_anomalies = {}
last_restart = {}

persistence_trusted = False
persistence_error = None
persistence_initialized = False

stop_event = threading.Event()
worker_thread = None

status_lock = threading.Lock()

controller_status = {
    "running": False,
    "mode": "DETECTION_ONLY",
    "remediation_enabled": AUTO_REMEDIATION_ENABLED,
    "last_check": None,
    "last_error": None,
    "services": {},
}


# --------------------------------------------------
# STATUS HELPERS
# --------------------------------------------------

def update_service_status(service, result, cpu=None, memory=None):
    with status_lock:
        controller_status["services"][service] = {
            **result,
            "cpu_millicores": cpu,
            "memory_mib": memory,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }


def get_controller_status():
    with status_lock:
        return {
            **controller_status,
            "services": {
                service: dict(data)
                for service, data in controller_status["services"].items()
            },
            "persistence_trusted": persistence_trusted,
            "persistence_error": persistence_error,
        }


# --------------------------------------------------
# PERSISTENT COOLDOWN STATE
# --------------------------------------------------

def load_state():
    global last_restart
    global persistence_trusted
    global persistence_error
    global persistence_initialized

    if not STATE_FILE.exists():
        if persistence_initialized or last_restart:
            persistence_trusted = False
            persistence_error = (
                "Cooldown state file disappeared after initialization."
            )
            print(f"WARNING: {persistence_error}")
            return

        last_restart = {}
        persistence_trusted = True
        persistence_error = None
        persistence_initialized = True
        return

    try:
        with STATE_FILE.open(encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError("state root must be an object")

        if "last_restart" not in data:
            raise ValueError("state is missing last_restart")

        saved_restarts = data["last_restart"]

        if not isinstance(saved_restarts, dict):
            raise ValueError("last_restart must be an object")

        restored_restarts = {}

        for service, value in saved_restarts.items():
            if service not in APPROVED_SERVICES:
                raise ValueError(f"invalid service key: {service}")

            timestamp = float(value)

            if not math.isfinite(timestamp) or timestamp < 0:
                raise ValueError(
                    f"invalid restart timestamp for {service}"
                )

            restored_restarts[service] = timestamp

    except (
        OSError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
        OverflowError,
    ) as error:
        persistence_trusted = False
        persistence_error = f"Cooldown state is untrusted: {error}"
        print(f"WARNING: {persistence_error}")
        return

    last_restart = restored_restarts
    persistence_trusted = True
    persistence_error = None
    persistence_initialized = True


def save_state(restart_history=None):
    global persistence_trusted
    global persistence_error
    global persistence_initialized

    history = last_restart if restart_history is None else restart_history

    try:
        if not isinstance(history, dict):
            raise ValueError("restart history must be an object")

        for service, timestamp in history.items():
            if service not in APPROVED_SERVICES:
                raise ValueError(f"invalid service key: {service}")

            value = float(timestamp)

            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"invalid restart timestamp for {service}"
                )

        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

        temporary_file = STATE_FILE.with_suffix(".tmp")

        with temporary_file.open("w", encoding="utf-8") as file:
            json.dump({"last_restart": history}, file, indent=2)

        temporary_file.replace(STATE_FILE)

        with STATE_FILE.open(encoding="utf-8") as file:
            written = json.load(file)

        if not isinstance(written, dict):
            raise ValueError("written state root is not an object")

        if written.get("last_restart") != history:
            raise ValueError(
                "written restart history could not be verified"
            )

    except (
        OSError,
        TypeError,
        ValueError,
        OverflowError,
        json.JSONDecodeError,
    ) as error:
        persistence_trusted = False
        persistence_error = (
            f"Cooldown state could not be persisted: {error}"
        )
        print(f"WARNING: {persistence_error}")
        return False

    persistence_trusted = True
    persistence_error = None
    persistence_initialized = True

    return True


# --------------------------------------------------
# TELEMETRY HELPERS
# --------------------------------------------------

def parse_timestamp(value):
    try:
        parsed = datetime.fromisoformat(value)

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)

        return parsed

    except (ValueError, TypeError):
        return None


def get_latest_metrics():
    records, _summary = read_and_validate_telemetry(CSV_FILE)

    if not records:
        print("Waiting for telemetry CSV...")
        return {}

    services = {}

    for row in records:
        if row.get("namespace") != "default":
            continue

        service = row.get("service")

        # Monitor only approved Online Boutique application services.
        if service not in APPROVED_SERVICES:
            continue

        timestamp = parse_timestamp(row.get("timestamp"))

        if timestamp is None:
            continue

        row["_parsed_timestamp"] = timestamp

        if (
            service not in services
            or timestamp > services[service]["_parsed_timestamp"]
        ):
            services[service] = row

    return services


# --------------------------------------------------
# KUBERNETES REMEDIATION
# --------------------------------------------------

def handle_remediation(service):
    if not AUTO_REMEDIATION_ENABLED:
        return {
            "service": service,
            "status": "ANOMALY_CONFIRMED",
            "message": "Detection-only mode. No restart was performed.",
            "error": None,
        }

    if not persistence_trusted:
        message = persistence_error or "Cooldown state is untrusted."

        print(f"BLOCKED: {message}")

        return {
            "service": service,
            "status": "REJECTED",
            "message": "Automatic remediation is blocked.",
            "error": "persistence_untrusted",
            "details": message,
        }

    if service not in APPROVED_SERVICES:
        print(f"REJECTED: {service} is not approved for restart.")

        return {
            "service": service,
            "status": "REJECTED",
            "message": "Service is not approved for restart.",
            "error": "unapproved_service",
        }

    now = time.time()
    previous_restart = float(last_restart.get(service, 0))

    if now - previous_restart < COOLDOWN_SECONDS:
        remaining = int(
            COOLDOWN_SECONDS - (now - previous_restart)
        )

        print(
            f"COOLDOWN: {service} restart blocked for {remaining}s."
        )

        return {
            "service": service,
            "status": "REJECTED",
            "message": "Restart is within the cooldown period.",
            "error": "cooldown_active",
        }

    print(f"\nAUTO-HEAL: Restarting {service}.")

    proposed_history = dict(last_restart)
    proposed_history[service] = now

    if not save_state(proposed_history):
        return {
            "service": service,
            "status": "FAILED",
            "message": (
                "Remediation blocked because restart history "
                "could not be persisted."
            ),
            "error": "persistence_failed",
            "details": persistence_error,
        }

    last_restart[service] = now
    consecutive_anomalies[service] = 0

    result = restart_service(
        service,
        persistence_verified=True,
    )

    if result is None:
        return {
            "service": service,
            "status": "FAILED",
            "message": "Remediation returned no result.",
            "error": "empty_remediation_result",
        }

    return result


# --------------------------------------------------
# CPU AND MEMORY ANOMALY DETECTION
# --------------------------------------------------

def check_service(service, record):
    try:
        required_fields = (
            "timestamp",
            "namespace",
            "service",
            "pod",
            "container",
            "cpu_millicores",
            "memory_mib",
        )

        if any(
            field not in record
            or record[field] is None
            or not str(record[field]).strip()
            for field in required_fields
        ):
            raise ValueError("missing required telemetry field")

        if (
            record["namespace"] != "default"
            or record["service"] != service
        ):
            raise ValueError("unsupported telemetry identity")

        timestamp = parse_timestamp(record["timestamp"])

        if timestamp is None:
            raise ValueError("invalid telemetry timestamp")

        cpu = float(record["cpu_millicores"])
        memory = float(record["memory_mib"])

    except (
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
    ):
        print(f"INVALID: Ignoring malformed telemetry for {service}.")

        consecutive_anomalies[service] = 0

        result = {
            "service": service,
            "status": "REJECTED",
            "message": "Malformed telemetry record was ignored.",
            "error": "invalid_telemetry",
        }

        update_service_status(service, result)
        return result

    if (
        not math.isfinite(cpu)
        or cpu < 0
        or not math.isfinite(memory)
        or memory < 0
    ):
        consecutive_anomalies[service] = 0

        result = {
            "service": service,
            "status": "REJECTED",
            "message": "Invalid CPU or memory measurement.",
            "error": "invalid_telemetry",
        }

        update_service_status(service, result)
        return result

    now = datetime.now(timezone.utc)
    age = (now - timestamp).total_seconds()

    if (
        age > MAX_SAMPLE_AGE_SECONDS
        or age < -MAX_SAMPLE_AGE_SECONDS
    ):
        print(f"STALE: {service} telemetry is not fresh.")

        consecutive_anomalies[service] = 0

        result = {
            "service": service,
            "status": "REJECTED",
            "message": "Telemetry record is stale.",
            "error": "stale_telemetry",
        }

        update_service_status(service, result, cpu, memory)
        return result

    pod = record["pod"]
    sample_key = (pod, timestamp.isoformat())

    if last_processed.get(service) == sample_key:
        return {
            "service": service,
            "status": "IGNORED",
            "message": "Duplicate telemetry record was ignored.",
            "error": "duplicate_record",
        }

    last_processed[service] = sample_key

    # Reset baseline whenever Kubernetes replaces a pod.
    if current_pods.get(service) != pod:
        current_pods[service] = pod
        baselines[service] = []
        consecutive_anomalies[service] = 0

        print(f"\nNEW POD: {service} ({pod})")
        print("Collecting a fresh CPU and memory baseline.")

    baseline = baselines.setdefault(service, [])

    # Collect initial baseline.
    if len(baseline) < BASELINE_SAMPLES:
        baseline.append((cpu, memory))

        print(
            f"BASELINE: {service} "
            f"{len(baseline)}/{BASELINE_SAMPLES} readings"
        )

        result = {
            "service": service,
            "status": "BASELINE",
            "message": (
                "CPU and memory baseline is being collected."
            ),
            "error": None,
        }

        update_service_status(service, result, cpu, memory)
        return result

    # Calculate separate CPU and memory thresholds.
    cpu_values = [sample[0] for sample in baseline]
    memory_values = [sample[1] for sample in baseline]

    cpu_mean = statistics.mean(cpu_values)
    cpu_std = statistics.pstdev(cpu_values)

    memory_mean = statistics.mean(memory_values)
    memory_std = statistics.pstdev(memory_values)

    cpu_limit = cpu_mean + max(3 * cpu_std, 5)
    memory_limit = memory_mean + max(3 * memory_std, 5)

    cpu_anomaly = cpu > cpu_limit
    memory_anomaly = memory > memory_limit

    # Either CPU or memory exceeding its threshold is an anomaly.
    if cpu_anomaly or memory_anomaly:
        consecutive_anomalies[service] = (
            consecutive_anomalies.get(service, 0) + 1
        )

        count = consecutive_anomalies[service]

        print(
            f"ANOMALY: {service} | "
            f"CPU: {cpu:.2f}m / Limit: {cpu_limit:.2f}m | "
            f"Memory: {memory:.2f}MiB / "
            f"Limit: {memory_limit:.2f}MiB | "
            f"Consecutive: {count}/{REQUIRED_ANOMALIES}"
        )

        if count >= REQUIRED_ANOMALIES:
            result = handle_remediation(service)

            if not AUTO_REMEDIATION_ENABLED:
                result["status"] = "ANOMALY_CONFIRMED"

            update_service_status(service, result, cpu, memory)
            return result

        result = {
            "service": service,
            "status": "ANOMALY",
            "message": (
                "CPU or memory anomaly detected. "
                "Waiting for consecutive confirmation."
            ),
            "error": None,
        }

        update_service_status(service, result, cpu, memory)
        return result

    # Normal reading resets the consecutive anomaly counter.
    consecutive_anomalies[service] = 0

    print(
        f"NORMAL: {service} | "
        f"CPU: {cpu:.2f}m / Limit: {cpu_limit:.2f}m | "
        f"Memory: {memory:.2f}MiB / Limit: {memory_limit:.2f}MiB"
    )

    result = {
        "service": service,
        "status": "NORMAL",
        "message": (
            "CPU and memory are within their baseline limits."
        ),
        "error": None,
    }

    update_service_status(service, result, cpu, memory)
    return result


# --------------------------------------------------
# BACKGROUND WORKER
# --------------------------------------------------

def controller_worker():
    load_state()

    print("\nCLOUDHEAL ANOMALY DETECTION CONTROLLER")
    print("-" * 45)
    print("CPU and memory monitoring enabled.")
    print(f"Baseline readings: {BASELINE_SAMPLES}")
    print(f"Required consecutive anomalies: {REQUIRED_ANOMALIES}")
    print(f"Automatic remediation enabled: {AUTO_REMEDIATION_ENABLED}")

    with status_lock:
        controller_status["running"] = True
        controller_status["mode"] = (
            "AUTOMATIC_HEALING"
            if AUTO_REMEDIATION_ENABLED
            else "DETECTION_ONLY"
        )
        controller_status["remediation_enabled"] = (
            AUTO_REMEDIATION_ENABLED
        )

    while not stop_event.is_set():
        try:
            services = get_latest_metrics()

            for service, record in sorted(services.items()):
                if stop_event.is_set():
                    break

                check_service(service, record)

            with status_lock:
                controller_status["last_check"] = (
                    datetime.now(timezone.utc).isoformat()
                )
                controller_status["last_error"] = None

        except Exception as error:
            message = str(error)

            print(f"CONTROLLER ERROR: {message}")

            with status_lock:
                controller_status["last_error"] = message

        stop_event.wait(CHECK_INTERVAL)

    with status_lock:
        controller_status["running"] = False

    print("CloudHeal anomaly detection controller stopped.")


def start_controller():
    global worker_thread

    if worker_thread is not None and worker_thread.is_alive():
        print("Anomaly detection controller is already running.")
        return

    stop_event.clear()

    worker_thread = threading.Thread(
        target=controller_worker,
        name="cloudheal-anomaly-controller",
        daemon=True,
    )

    worker_thread.start()

    print("CloudHeal anomaly detection background worker started.")


def stop_controller():
    stop_event.set()

    if worker_thread is not None and worker_thread.is_alive():
        worker_thread.join(timeout=10)

    print("CloudHeal anomaly detection stop requested.")


# --------------------------------------------------
# STANDALONE EXECUTION
# --------------------------------------------------

def main():
    print("Starting CloudHeal anomaly detection...")
    print("Press Ctrl+C to stop.\n")

    start_controller()

    try:
        while worker_thread is not None and worker_thread.is_alive():
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping CloudHeal controller...")
        stop_controller()


if __name__ == "__main__":
    main()
