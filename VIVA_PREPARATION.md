# CloudHeal MCA Viva Preparation

## Core questions and simple answers

### 1. What is cloud computing?

Cloud computing provides computing resources such as servers, storage, and
networks on demand. In this project, Kubernetes manages application workloads
in a cloud-native style.

### 2. What is a cloud-native application?

It is an application designed around independently deployable services,
containers, automation, and orchestration. Online Boutique is composed of
multiple microservices.

### 3. What is a microservice?

A microservice is a small service responsible for one business capability.
Services can be deployed and monitored independently.

### 4. What is Docker?

Docker packages an application and its dependencies into a container image.
Containers make deployment more consistent.

### 5. What is Kubernetes?

Kubernetes is an orchestration platform that schedules containers, maintains
desired state, exposes services, and replaces failed pods.

### 6. What is a Pod?

A Pod is the smallest Kubernetes execution unit. It contains one or more
closely related containers.

### 7. What is a Deployment?

A Deployment declares how many pod replicas should run and manages replacement
pods when pods are deleted or fail.

### 8. What is a Service?

A Service gives a stable network endpoint for a group of pods selected by
labels.

### 9. What is telemetry?

Telemetry is machine-generated operational data. CloudHeal collects CPU and
memory usage from the Kubernetes Metrics API.

### 10. Why are CPU and memory converted?

Kubernetes may return values such as `250m`, `500u`, `128Mi`, or `1Gi`.
CloudHeal converts them to common units: millicores and MiB.

### 11. What does preprocessing do?

It validates required columns, timestamps, identities, numeric measurements,
duplicates, missing values, ordering, groups, and pod changes. It never fills
missing measurements or overwrites the original CSV.

### 12. What is anomaly detection?

Anomaly detection identifies behaviour that differs significantly from normal
behaviour. CloudHeal's live detector compares current CPU/memory with recent
same-pod history.

### 13. What is deep learning?

Deep learning uses layered neural networks to learn patterns from data. The
LSTM Autoencoder is the deep-learning experiment in this project.

### 14. Why use an LSTM?

LSTMs are suitable for sequences because they can learn relationships across
time. Cloud telemetry is naturally a time series.

### 15. What is an Autoencoder?

An Autoencoder compresses input data and reconstructs it. It can learn normal
patterns without requiring every normal pattern to be manually labelled.

### 16. What is reconstruction error?

It is the difference between the original sequence and the reconstructed
sequence. A high error means the sequence is unlike the learned normal
patterns.

### 17. What is the anomaly threshold?

The threshold is the reconstruction-error value used to separate normal and
anomalous windows. CloudHeal's saved LSTM threshold is approximately 5.238637.

### 18. What are precision and recall?

Precision answers: of the windows predicted anomalous, how many were really
anomalous? Recall answers: of the real anomalies, how many were detected?

### 19. What is F1-score?

F1-score is the harmonic mean of precision and recall. It is useful when both
false alarms and missed anomalies matter.

### 20. What is a confusion matrix?

It counts true negatives, false positives, false negatives, and true
positives. The saved LSTM results are TN 18,430, FP 209, FN 24, and TP 18,615.

### 21. What is failure diagnosis?

Diagnosis estimates which service and fault type are associated with the
evidence. The current live diagnosis reports CPU or memory suspicion using
statistical evidence.

### 22. Is the live diagnosis produced by the LSTM?

No. The live diagnosis is statistical CPU/memory analysis. The LSTM results
shown in the dashboard are offline RCAEval evaluation results.

### 23. Why is the LSTM currently offline?

The trained model requires 47 features, including latency, load, and error
features. Current live telemetry provides only CPU and memory. Inventing the
other 45 features would make the inference invalid.

### 24. What does CloudHeal do when a live anomaly is found?

It reports a diagnosis and evaluates a guarded dry-run decision. The current
path does not automatically restart a service.

### 25. What is self-healing in this project?

The project demonstrates two separate ideas: a protected dry-run remediation
decision path, and an explicit controlled demo of Kubernetes replacing a
deleted pod. General automatic healing is not active.

### 26. What happens in the controlled recovery demo?

CloudHeal finds a ready checkoutservice pod, deletes that one pod, waits for
the old pod to disappear, observes a replacement pod, and verifies that the
replacement is ready and the Deployment has the required available replicas.

### 27. Who actually recreates the replacement pod?

The Kubernetes Deployment controller recreates it because the Deployment's
desired state still requires the replica. CloudHeal verifies this process.

### 28. What is the novelty of CloudHeal?

The practical novelty is the safe integration of validated telemetry, live
explainable diagnosis, an offline LSTM study, dry-run remediation safeguards,
and Kubernetes recovery verification without falsely claiming live deep
learning or uncontrolled automatic healing.

### 29. What are the evaluation limitations?

The LSTM metrics come from an offline RCAEval test set. They do not measure the
current live statistical detector. Live detection latency, live precision and
recall, and formal recovery duration were not measured.

### 30. What would you improve with more time?

I would collect the missing legitimate telemetry features, validate a
47-feature live adapter, measure live detection performance, add authenticated
operator approval, and enable remediation only through a separately controlled
and tested stage.

## Difficult panel questions

### Why not duplicate CPU and memory to make 47 inputs?

Because that would fabricate data and change the meaning of the trained model.
It would produce an apparently working result that is scientifically invalid.

### Does the project really perform self-healing?

It demonstrates recovery verification and has a dry-run remediation design.
The generalized automatic controller is intentionally inactive. The controlled
demo is explicit and operator-triggered.

### Is the recovery caused by CloudHeal or Kubernetes?

The replacement is caused by the Kubernetes Deployment controller. CloudHeal
initiates the controlled pod deletion and verifies the replacement. This
distinction is shown in the dashboard and is important for accuracy.

### Why use a statistical detector if the title mentions deep learning?

The deep-learning model is a valid offline research component, but live data
does not yet satisfy its input schema. The statistical detector is a valid,
transparent live alternative rather than a false claim of LSTM inference.

### What is the strongest evidence that the project works?

The backend health, collector status, live telemetry preprocessing, dashboard
build, 121 passing backend tests, saved LSTM evaluation results, and completed
controlled recovery status provide separate evidence for the implemented
parts.

## Demonstration script

1. Run `Start-CloudHeal.bat` manually.
2. Open `http://localhost:5173/`.
3. Show `/api/health` or the healthy dashboard banner.
4. Show Kubernetes pods and current CPU/memory values.
5. Show preprocessing counts and timestamp range.
6. Explain that live detection is statistical.
7. Show the offline LSTM panel and its 47-feature limitation.
8. Show the dry-run label: **DRY RUN — NO ACTION EXECUTED**.
9. Only with approval, press the controlled recovery demo button.
10. Explain that Kubernetes replaces the deleted checkoutservice pod.
11. Show the completed recovery status and replacement pod.

## Manual actions before presentation

- Start the project manually with `Start-CloudHeal.bat`.
- If the dashboard reports 404 for newer routes, stop the old project
  processes using the existing stop script and run the launcher again.
- Confirm K3s and the Online Boutique deployment are already available.
- Do not press the recovery demo button unless the panel demonstration is
  approved and the cluster can tolerate deleting one checkoutservice pod.
- Do not claim live LSTM inference, automatic healing, or live performance
  metrics.
