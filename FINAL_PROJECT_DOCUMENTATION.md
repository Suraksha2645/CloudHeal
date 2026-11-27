# CloudHeal Final Project Documentation

## 1. Project title

**CloudHeal: A Deep Learning-Based Self-Healing Cloud-Native Application**

## 2. Abstract

CloudHeal monitors the Online Boutique microservices application running on
Kubernetes. A FastAPI backend collects CPU and memory telemetry, validates the
records, performs a live statistical comparison against recent behaviour, and
reports possible service anomalies to a React dashboard. An LSTM Autoencoder is
trained and evaluated offline on the RCAEval RE1 dataset. The current live
telemetry contains only CPU and memory, while the trained model requires 47
features, so the LSTM is not used for live inference. CloudHeal also provides a
safe dry-run remediation decision path and a controlled recovery demonstration
where Kubernetes replaces a deleted checkoutservice pod. The system reports
what is genuinely implemented and does not present the controlled demo or
offline model as automatic production healing.

## 3. Problem statement

Cloud-native applications contain many services and pods. A failure in one
component can affect the complete application. Operators need timely telemetry,
anomaly detection, diagnosis, and evidence that recovery really occurred.
CloudHeal investigates a practical, safe workflow for these activities.

## 4. Objectives

1. Monitor a Kubernetes microservice application.
2. Collect real CPU and memory metrics.
3. Validate and organize live telemetry.
4. Study deep-learning anomaly detection offline.
5. Diagnose likely services using live statistical evidence.
6. Safely evaluate remediation decisions without enabling uncontrolled healing.
7. Verify Kubernetes replacement of a failed pod.
8. Report measurable offline model results and honest limitations.

## 5. Research gap

Many self-healing examples demonstrate either anomaly detection or Kubernetes
restart logic in isolation. CloudHeal combines telemetry validation, live
statistical diagnosis, offline LSTM evaluation, dry-run safeguards, explicit
authorization capabilities, and recovery verification in one student-scale
system. The contribution is the safe, explainable integration and the clear
separation between experimental deep learning and currently available live
telemetry.

## 6. Proposed methodology

1. Kubernetes Metrics API supplies pod/container CPU and memory usage.
2. The collector converts CPU to millicores and memory to MiB and appends CSV
   records every 10 seconds.
3. The preprocessing module validates schema, timestamps, identities,
   measurements, duplicates, ordering, groups, and pod changes.
4. The live diagnosis adapter uses recent same-pod CPU/memory history.
5. The offline LSTM uses RCAEval processed features and saved model artifacts.
6. The dry-run orchestrator evaluates guarded remediation eligibility but never
   executes a restart.
7. The controlled demonstration deliberately deletes one checkoutservice pod
   only when an operator presses the demo button; Kubernetes then replaces it.
8. The dashboard displays the evidence and labels offline or simulated paths
   accurately.

## 7. System architecture

```text
Kubernetes Metrics API
        |
        v
src/telemetry/collector.py
        |
        v
data/live/telemetry.csv
        |
        v
live_preprocessing.py -----> /api/telemetry/preprocessing -----> React dashboard
        |
        v
live_ai.py ---------------> /api/ai/status -------------------> live diagnosis

RCAEval RE1 data -> offline preprocessing/training -> saved LSTM artifacts
                                             |
                                             v
                             /api/ai/offline-evaluation -> dashboard

live diagnosis -> live_remediation.py -> /api/remediation/dry-run
operator approval -> remediation_authorization.py

controlled operator demo -> /api/demo/start -> Kubernetes Deployment controller
```

## 8. Technology stack

| Area | Technology |
|---|---|
| Backend | Python 3.14 environment, FastAPI 0.162.0, Pydantic 2.14.2 |
| Frontend | React 19.3.1, Vite 8.3.5, Node 24.14.1 |
| Cluster | K3s in WSL Ubuntu 22.04 |
| Orchestration | Kubernetes Deployments, Pods, Services |
| Data | CSV telemetry and RCAEval RE1 |
| Model | PyTorch LSTM Autoencoder; saved artifact |
| Testing | pytest 9.2.0 |

## 9. Eight planned functionalities

| Functionality | Final status | Evidence and honest scope |
|---|---|---|
| Cloud-native application management | Fully implemented and tested | Kubernetes pod/metric APIs and dashboard are wired; runtime pod data was returned successfully. |
| Real-time telemetry collection | Fully implemented and tested | Collector uses Metrics API, converts units, appends CSV data, and reports running status. |
| Telemetry preprocessing | Fully implemented and tested | Read-only validator and API; current inspection found 42,702 valid rows and no rejected or duplicate rows. |
| Deep-learning anomaly detection | Partially implemented; offline evaluation | LSTM is trained/evaluated offline. Live LSTM inference is blocked because 45 of 47 required features are unavailable. |
| Failure diagnosis | Fully implemented and tested within live statistical scope | `live_ai.py` compares recent same-pod CPU/memory history and reports a likely service/fault. It is not LSTM diagnosis. |
| Automated self-healing | Partially implemented / dry-run and controlled demo | Dry-run and authorization paths do not execute remediation. General automatic healing is inactive. |
| Recovery verification | Demonstration-only, tested through the controlled flow | The checkoutservice demo deletes one pod and verifies replacement readiness and Deployment availability. |
| Fault testing and performance evaluation | Partially implemented | Offline confusion-matrix metrics and controlled demo evidence exist; live latency, live recovery-time experiments, and live precision/recall are not measured. |

## 10. Complete system workflow

The normal monitoring workflow is:

1. Start the existing launcher.
2. The backend starts the collector.
3. The collector reads Kubernetes Metrics API data.
4. CPU and memory are written to the live CSV.
5. The preprocessing endpoint validates the CSV.
6. The live statistical adapter reads validated records.
7. The dashboard displays service health, telemetry, diagnosis, and dry-run
   state.

The separate deep-learning workflow is offline:

1. RCAEval RE1 cases are cleaned and labelled.
2. The LSTM Autoencoder learns normal temporal patterns.
3. Reconstruction error is compared with a validation-selected threshold.
4. Saved test metrics are shown as offline results.

The recovery demonstration is separate:

1. An operator presses the controlled demo button.
2. CloudHeal deletes one `checkoutservice` pod.
3. The Kubernetes Deployment controller creates a replacement pod.
4. CloudHeal checks replacement readiness and Deployment availability.

## 11. Dataset

The offline experiment uses RCAEval RE1 Online Boutique cases. The model
configuration contains 47 features and a sequence length of 30. Features
include service CPU, memory, latency, load, and error measurements. Live
CloudHeal telemetry currently contains only:

```text
timestamp, namespace, service, pod, container,
cpu_millicores, memory_mib
```

Therefore, live telemetry cannot validly be reshaped into the 47-feature model
input without obtaining the missing measurements.

## 12. LSTM Autoencoder

The encoder reads a sequence of 30 rows, compresses the sequence into a latent
representation, and the decoder reconstructs the original sequence.
Reconstruction error is the mean squared difference between input and output.
Large error indicates that the sequence differs from learned normal patterns.

The saved configuration uses:

- 47 input features
- 30 time steps
- hidden size 32
- latent size 16
- saved threshold approximately 5.238637

## 13. Model training and evaluation

The saved LSTM test results are:

| Metric | Value |
|---|---:|
| Accuracy | 0.993750 |
| Precision | 0.988897 |
| Recall | 0.998712 |
| F1-score | 0.993781 |
| True negative | 18,430 |
| False positive | 209 |
| False negative | 24 |
| True positive | 18,615 |

These are offline RCAEval test results. They are not live dashboard
performance, and they do not prove that the current CPU/memory live detector
has the same accuracy.

## 14. Anomaly detection and diagnosis

The live detector calculates relative CPU and memory increases against recent
history for the same namespace, service, pod, and container. It requires
historical evidence, isolates replacement pods, and selects equal-score
candidates deterministically. Its score is a relative deviation, not a
calibrated probability.

## 15. Self-healing mechanism

The generalized remediation controller exists with safeguards, but it is not
started by the launcher and is not connected to the live dashboard path.
`/api/remediation/dry-run` only evaluates what would be required and returns
`dry_run: true` and `execution_performed: false`. The authorization module
creates single-use, expiring capabilities for a future stage, but consuming a
capability currently does not execute Kubernetes remediation.

The controlled demo is different: it is an explicit operator-triggered
demonstration of Kubernetes Deployment recovery, not generalized automatic
healing.

## 16. Testing and results

The project test suite covers preprocessing, live diagnosis, dry-run state,
cross-process coordination, authorization, controller safeguards, API
contracts, and offline evaluation reporting.

Latest recorded validation:

- Full backend suite: 121 passed, 4 expected FastAPI deprecation warnings.
- Python compilation: passed.
- Frontend production build: passed.
- Real telemetry quality inspection: 42,702 valid, 0 rejected, 0 exact
  duplicates; source hash unchanged.
- Runtime health endpoint: healthy.
- Runtime collector: running.
- Runtime controlled demo status: completed and recovery verified.

No detection latency or recovery duration is recorded as a formal experimental
metric in the current evaluation outputs.

## 17. Limitations

1. Live telemetry has only CPU and memory.
2. The LSTM is offline-only.
3. Live statistical diagnosis has no published live precision/recall.
4. General automatic remediation is disabled.
5. The authorization API records an operator identifier but does not provide
   production authentication.
6. Controlled demo recovery time is not stored as a formal metric.
7. The current running backend may need a manual restart after source changes.

## 18. Future enhancements

1. Add legitimate sources for latency, load, error, and other missing features.
2. Build a validated live feature adapter and only then enable live LSTM
   inference.
3. Measure live detection latency and diagnosis precision/recall.
4. Add authenticated operator approval.
5. Connect remediation only through a separately approved controlled stage.
6. Record recovery duration and compare Kubernetes-only versus CloudHeal-
   initiated recovery.
