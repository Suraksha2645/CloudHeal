from src.telemetry.auto_heal_controller import (
    start_controller,
    stop_controller,
    get_controller_status,
)
from datetime import datetime
from pathlib import Path
import threading
import time
from time import monotonic

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from kubernetes import client, config
from kubernetes.config.config_exception import ConfigException
from kubernetes.client.exceptions import ApiException

from src.telemetry.live_ai import get_live_ai_status
from src.telemetry.live_remediation import get_dry_run_status
from src.telemetry.collector import telemetry_collector
from src.telemetry.live_preprocessing import read_and_validate_telemetry
from src.telemetry import remediation_authorization
from src.models.offline_evaluation import get_offline_evaluation


# ==========================================
# FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="CloudHeal API",
    description="Self-Healing Cloud-Native Microservices Framework",
    version="2.0.0"
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TELEMETRY_FILE = Path("/mnt/c/CloudHeal/data/live/telemetry.csv")
PREPROCESSING_CACHE_TTL_SECONDS = 10
PREPROCESSING_CACHE_LOCK = threading.Lock()
PREPROCESSING_CACHE = None
PREPROCESSING_CACHE_TIME = 0.0
PREPROCESSING_UPDATED_AT = None

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static"
)


@app.on_event("startup")
def start_background_services():
    telemetry_collector.start()
    start_controller()


@app.on_event("shutdown")
def stop_background_services():
    stop_controller()
    telemetry_collector.stop()

@app.get("/api/anomalies/status")
def get_anomaly_status():
    return get_controller_status()

# ==========================================
# KUBERNETES CONNECTION
# ==========================================

def load_kubernetes_config():
    try:
        config.load_incluster_config()
    except ConfigException:
        config.load_kube_config()


def get_kubernetes_core():
    load_kubernetes_config()
    return client.CoreV1Api()


def get_kubernetes_apps():
    load_kubernetes_config()
    return client.AppsV1Api()


# ==========================================
# DEMONSTRATION STATE
# ==========================================

DEMO_STATE = {
    "running": False,
    "stage": "Idle",
    "progress": 0,
    "message": "Ready to start the self-healing test.",
    "affected_service": None,
    "fault": None,
    "old_pod": None,
    "replacement_pod": None,
    "recovery_action": None,
    "recovery_verified": False,
    "error": None
}

DEMO_LOCK = threading.Lock()


def update_demo(**values):
    with DEMO_LOCK:
        DEMO_STATE.update(values)


def get_demo_state():
    with DEMO_LOCK:
        return DEMO_STATE.copy()


# ==========================================
# HOME PAGE AND HEALTH
# ==========================================

@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {
        "project": "CloudHeal",
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/dashboard")
def dashboard():
    return {
        "project": "CloudHeal",
        "application": "Online Boutique",
        "monitoring": "Connected",
        "anomaly_detection": "Kubernetes pod failure monitoring",
        "self_healing": "Automatic recovery verification"
    }


# ==========================================
# LIVE AI MONITORING
# ==========================================

@app.get("/api/ai/status")
def ai_status():
    return get_live_ai_status()


@app.get("/api/ai/offline-evaluation")
def offline_evaluation():
    try:
        return get_offline_evaluation()
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/remediation/dry-run")
def remediation_dry_run():
    return get_dry_run_status()


class RemediationAuthorizationRequest(BaseModel):
    incident_id: str
    service: str
    pod: str
    container: str
    anomaly_type: str
    proposed_action: str
    expires_at: str
    operator_id: str


class RemediationConsumptionRequest(BaseModel):
    authorization_token: str
    authorization_id: str
    incident_id: str
    service: str
    pod: str
    container: str
    anomaly_type: str
    proposed_action: str


@app.post("/api/remediation/authorize")
def remediation_authorize(request: RemediationAuthorizationRequest):
    try:
        return remediation_authorization.approve_authorization(
            **request.model_dump()
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/api/remediation/consume-authorization")
def remediation_consume(request: RemediationConsumptionRequest):
    try:
        return remediation_authorization.consume_authorization(
            **request.model_dump()
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/api/telemetry/status")
def telemetry_status():
    return telemetry_collector.status()


@app.get("/api/telemetry/preprocessing")
def telemetry_preprocessing():
    global PREPROCESSING_CACHE
    global PREPROCESSING_CACHE_TIME
    global PREPROCESSING_UPDATED_AT

    now = monotonic()
    with PREPROCESSING_CACHE_LOCK:
        cache_is_fresh = (
            PREPROCESSING_CACHE is not None
            and now - PREPROCESSING_CACHE_TIME
            < PREPROCESSING_CACHE_TTL_SECONDS
        )
        if not cache_is_fresh:
            _, summary = read_and_validate_telemetry(TELEMETRY_FILE)
            PREPROCESSING_CACHE = {
                "status": summary["status"],
                "total_rows": summary["total_rows"],
                "valid_records": summary["valid_records"],
                "rejected_records": summary["rejected_records"],
                "exact_duplicate_records": summary[
                    "exact_duplicate_records"
                ],
                "missing_field_counts": summary["missing_field_counts"],
                "invalid_field_counts": summary["invalid_field_counts"],
                "rejection_reasons": summary["rejection_reasons"],
                "timestamp_range": summary["timestamp_range"],
                "group_count": len(summary["groups"]),
                "service_count": len(summary["pod_changes"]),
                "pod_changes_by_service": {
                    service: details["pod_change_count"]
                    for service, details in summary["pod_changes"].items()
                },
            }
            PREPROCESSING_CACHE_TIME = now
            PREPROCESSING_UPDATED_AT = datetime.now().isoformat()

        response = dict(PREPROCESSING_CACHE)
        response["last_preprocessing_update"] = PREPROCESSING_UPDATED_AT
        response["cache_ttl_seconds"] = PREPROCESSING_CACHE_TTL_SECONDS
        return response


# ==========================================
# APPLICATION POD MONITORING
# ==========================================

@app.get("/api/kubernetes/pods")
def get_pods():
    try:
        core_api = get_kubernetes_core()

        result = core_api.list_namespaced_pod(
            namespace="default"
        )

        pod_data = []

        for pod in result.items:
            containers = pod.status.container_statuses or []

            pod_data.append({
                "name": pod.metadata.name,
                "status": pod.status.phase,
                "ready": sum(
                    1 for container in containers
                    if container.ready
                ),
                "total_containers": len(pod.spec.containers),
                "restarts": sum(
                    container.restart_count
                    for container in containers
                )
            })

        return {
            "cluster": "K3s",
            "namespace": "default",
            "total_pods": len(pod_data),
            "pods": pod_data
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ==========================================
# CPU AND MEMORY MONITORING
# ==========================================

@app.get("/api/kubernetes/metrics")
def get_pod_metrics():
    try:
        load_kubernetes_config()

        metrics_api = client.CustomObjectsApi()

        result = metrics_api.list_namespaced_custom_object(
            group="metrics.k8s.io",
            version="v1beta1",
            namespace="default",
            plural="pods"
        )

        pod_metrics = []

        for pod in result.get("items", []):
            total_cpu = 0.0
            total_memory = 0.0

            for container in pod.get("containers", []):
                cpu = container["usage"]["cpu"]
                memory = container["usage"]["memory"]

                if cpu.endswith("n"):
                    total_cpu += float(cpu[:-1]) / 1_000_000
                elif cpu.endswith("u"):
                    total_cpu += float(cpu[:-1]) / 1_000
                elif cpu.endswith("m"):
                    total_cpu += float(cpu[:-1])
                else:
                    total_cpu += float(cpu) * 1000

                if memory.endswith("Ki"):
                    total_memory += float(memory[:-2]) / 1024
                elif memory.endswith("Mi"):
                    total_memory += float(memory[:-2])
                elif memory.endswith("Gi"):
                    total_memory += float(memory[:-2]) * 1024
                elif memory.endswith("Ti"):
                    total_memory += float(memory[:-2]) * 1024 * 1024
                else:
                    total_memory += float(memory) / (1024 * 1024)

            pod_metrics.append({
                "name": pod["metadata"]["name"],
                "cpu_millicores": round(total_cpu, 2),
                "memory_mib": round(total_memory, 2)
            })

        return {
            "metrics": pod_metrics,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ==========================================
# AUTOMATIC SELF-HEALING PROCESS
# ==========================================

def run_checkoutservice_recovery():

    service = "checkoutservice"
    namespace = "default"

    try:
        core_api = get_kubernetes_core()
        apps_api = get_kubernetes_apps()

        # STEP 1: Find the Deployment

        update_demo(
            stage="Preparing",
            progress=10,
            message="Finding the checkoutservice application.",
            affected_service=service,
            fault="Controlled pod failure",
            recovery_verified=False
        )

        deployment = apps_api.read_namespaced_deployment(
            name=service,
            namespace=namespace,
            _request_timeout=10
        )

        
        selector = deployment.spec.selector.match_labels

        # Use the Deployment's actual selector without adding extra labels.
        label_selector = ",".join(
            f"{key}={value}"
            for key, value in selector.items()
            if value is not None
        )

        old_pod = None

        # Wait up to 120 seconds for a genuinely healthy pod.
        for attempt in range(60):

            pods = core_api.list_namespaced_pod(
                namespace=namespace,
                label_selector=label_selector,
                _request_timeout=10
            )

            for pod in pods.items:

                containers = pod.status.container_statuses or []
                expected = len(pod.spec.containers or [])

                if (
                    pod.metadata.deletion_timestamp is None
                    and pod.status.phase == "Running"
                    and expected > 0
                    and len(containers) == expected
                    and all(container.ready for container in containers)
                ):
                    old_pod = pod
                    break

            if old_pod is not None:
                break

            update_demo(
                stage="Preparing",
                progress=10,
                message="Waiting for a healthy checkoutservice pod."
            )

            time.sleep(2)

        if old_pod is None:
            raise RuntimeError(
                "Checkoutservice did not become healthy within 120 seconds."
            )

        old_pod_name = old_pod.metadata.name


        # STEP 2: INJECT CONTROLLED FAILURE

        update_demo(
            stage="Fault Injection",
            progress=25,
            message="Injecting a controlled failure into checkoutservice.",
            old_pod=old_pod_name
        )

        core_api.delete_namespaced_pod(
            name=old_pod_name,
            namespace=namespace,
            body=client.V1DeleteOptions(
                grace_period_seconds=0
            )
        )

        # STEP 3: DETECT FAILURE

        detected = False

        for _ in range(30):
            try:
                core_api.read_namespaced_pod(
                    name=old_pod_name,
                    namespace=namespace,
                    _request_timeout=10
                )
            except ApiException as error:
                if error.status == 404:
                    detected = True
                    break
                raise

            time.sleep(1)

        if not detected:
            raise RuntimeError(
                "The injected pod failure could not be confirmed."
            )

        update_demo(
            stage="Anomaly Detected",
            progress=45,
            message="CloudHeal confirmed that the original checkoutservice pod is unavailable.",
            recovery_action="Kubernetes Deployment controller recreates the pod."
        )

        # STEP 4: WAIT FOR KUBERNETES RECOVERY

        update_demo(
            stage="Recovery in Progress",
            progress=55,
            message="Waiting for Kubernetes to create a replacement pod."
        )

        replacement_name = None
        replacement_ready = False

        for _ in range(90):

            current_pods = core_api.list_namespaced_pod(
                namespace=namespace,
                label_selector=label_selector,
                _request_timeout=10
            )

            for pod in current_pods.items:

                pod_name = pod.metadata.name

                if pod_name == old_pod_name:
                    continue

                update_demo(
                    stage="Replacement Created",
                    progress=75,
                    message="Kubernetes created a replacement checkoutservice pod.",
                    replacement_pod=pod_name
                )

                containers = pod.status.container_statuses or []

                if (
                    pod.status.phase == "Running"
                    and len(containers) == len(pod.spec.containers)
                    and all(container.ready for container in containers)
                ):
                    replacement_name = pod_name
                    replacement_ready = True
                    break

            if replacement_ready:
                break

            time.sleep(1)

        if not replacement_ready:
            raise RuntimeError(
                "Replacement pod did not become ready within 90 seconds."
            )

        # STEP 5: VERIFY DEPLOYMENT RECOVERY

        update_demo(
            stage="Recovery Verification",
            progress=90,
            message="Replacement pod is ready. Verifying the Deployment."
        )

        final_deployment = apps_api.read_namespaced_deployment(
            name=service,
            namespace=namespace,
            _request_timeout=10
        )

        desired = final_deployment.spec.replicas or 1
        ready = final_deployment.status.ready_replicas or 0
        available = final_deployment.status.available_replicas or 0

        verified = (
            ready >= desired
            and available >= desired
            and replacement_ready
        )

        if not verified:
            raise RuntimeError(
                "The replacement pod is ready, but the Deployment "
                "has not yet reached its desired available replicas."
            )

        # STEP 6: SUCCESS

        update_demo(
            running=False,
            stage="Completed",
            progress=100,
            message="Checkoutservice recovered successfully. Recovery verified.",
            replacement_pod=replacement_name,
            recovery_action=(
                "Kubernetes automatically recreated the deleted "
                "checkoutservice pod through its Deployment controller."
            ),
            recovery_verified=True,
            error=None
        )

    except Exception as error:

        update_demo(
            running=False,
            stage="Failed",
            message="The self-healing process could not be completed.",
            error=str(error),
            recovery_verified=False
        )


# ==========================================
# START SELF-HEALING WITH ONE BUTTON
# ==========================================


@app.post("/api/demo/start")
def start_self_healing_demo():
    with DEMO_LOCK:
        if DEMO_STATE["running"]:
            return {
                "message": "Demonstration is already running.",
                "demo": DEMO_STATE.copy()
            }

        DEMO_STATE.update({
            "running": True,
            "stage": "Starting",
            "progress": 5,
            "message": "Starting the controlled self-healing demonstration.",
            "affected_service": None,
            "fault": None,
            "old_pod": None,
            "replacement_pod": None,
            "recovery_action": None,
            "recovery_verified": False,
            "error": None
        })

    worker = threading.Thread(
        target=run_checkoutservice_recovery,
        daemon=True,
        name="cloudheal-demo"
    )
    worker.start()

    return {
        "message": "Self-healing demonstration started.",
        "demo": get_demo_state()
    }


# ==========================================
# LIVE SELF-HEALING STATUS
# ==========================================

@app.get("/api/demo/status")
def get_self_healing_demo_status():

    return {
        "demo": get_demo_state(),
        "timestamp": datetime.now().isoformat()
    }