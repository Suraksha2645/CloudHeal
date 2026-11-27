import subprocess
import json
from datetime import datetime
from pathlib import Path

# Only these deployments are allowed to be restarted
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

PROJECT = Path("/mnt/c/CloudHeal")
if not (PROJECT / "src" / "remediation").is_dir():
    PROJECT = Path(__file__).resolve().parents[2]
LOG_FILE = PROJECT / "data" / "live" / "remediation_log.jsonl"


def run_command(command):
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=150
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())

    return result.stdout.strip()


def record_action(service, status, message):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "service": service,
        "status": status,
        "message": message
    }

    with LOG_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


def restart_service(service, *, persistence_verified=False):
    if not persistence_verified:
        return {
            "service": service,
            "status": "REJECTED",
            "message": "Remediation requires verified restart-history persistence.",
            "error": "persistence_required",
        }

    if service not in APPROVED_SERVICES:
        print(f"REJECTED: {service} is not an approved service.")
        return {
            "service": service,
            "status": "REJECTED",
            "message": "Service is not approved for restart.",
            "error": "unapproved_service",
        }

    print(f"\nCLOUDHEAL REMEDIATION")
    print("------------------------------")
    print(f"Service: {service}")
    print("Action: Restarting Kubernetes deployment...")

    try:
        run_command([
            "kubectl", "rollout", "restart",
            f"deployment/{service}", "-n", "default"
        ])

        print("Waiting for rollout to complete...")

        output = run_command([
            "kubectl", "rollout", "status",
            f"deployment/{service}",
            "-n", "default",
            "--timeout=120s"
        ])

        print(output)

        ready = run_command([
            "kubectl", "get", "deployment", service,
            "-n", "default",
            "-o", "json"
        ])

        deployment = json.loads(ready)
        status = deployment.get("status", {})

        desired = deployment.get("spec", {}).get(
            "replicas", 1
        )
        available = status.get("availableReplicas", 0)
        updated = status.get("updatedReplicas", 0)
        ready = status.get("readyReplicas", 0)
        if (
            available == desired
            and updated == desired
            and ready == desired
        ):
            message = "Deployment restarted and verified healthy."
            record_action(service, "SUCCESS", message)
            print("RECOVERY VERIFIED: Deployment is healthy.")
            return {
                "service": service,
                "status": "SUCCESS",
                "message": message,
                "error": None,
            }
        else:
            message = "Rollout completed, but health verification failed."
            record_action(service, "WARNING", message)
            print("WARNING: Deployment is not fully healthy.")
            return {
                "service": service,
                "status": "WARNING",
                "message": message,
                "error": "health_verification_failed",
            }

    except Exception as error:
        message = str(error)
        record_action(service, "FAILED", message)
        print(f"REMEDIATION FAILED: {message}")
        return {
            "service": service,
            "status": "FAILED",
            "message": "Remediation command failed.",
            "error": message,
        }


if __name__ == "__main__":
    service_name = input(
        "Enter the service name to restart: "
    ).strip()

    restart_service(service_name)
