from pathlib import Path
import sys
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# PROJECT PATH
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "data"))

from src.data.preprocess_re1 import load_case, clean_case
from dataset_group_l import case_ids_for_split


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

GROUP = "L"

Z_THRESHOLD = 3.0
SUSTAINED_SECONDS = 5
DIAGNOSIS_WINDOW_SECONDS = 300

EPSILON = 1e-9


# ------------------------------------------------------------
# BASIC CASE INFORMATION
# ------------------------------------------------------------

def get_service_from_case(case_id):
    case_name = case_id.split("/")[0]
    return case_name.rsplit("_", 1)[0]


def get_fault_from_case(case_id):
    case_name = case_id.split("/")[0]
    return case_name.rsplit("_", 1)[1]


# ------------------------------------------------------------
# TELEMETRY COLUMNS
# ------------------------------------------------------------

def telemetry_columns(first_case):
    case_dir = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "rcaeval_re1"
        / "RE1"
        / "RE1"
        / "RE1-OB"
        / first_case
    )

    df_raw, _ = load_case(case_dir)
    df = clean_case(df_raw)

    excluded = {
        "time",
        "time.1",
        "label",
        "PassthroughCluster_error"
    }

    columns = [
        column
        for column in df.columns
        if column not in excluded
    ]

    return columns


# ------------------------------------------------------------
# ROBUST BASELINE
# ------------------------------------------------------------

def robust_location_scale(series):
    values = pd.to_numeric(series, errors="coerce").dropna()

    if len(values) == 0:
        return 0.0, 1.0

    median = values.median()

    mad = np.median(
        np.abs(values - median)
    )

    robust_scale = mad * 1.4826

    minimum_scale = 0.05 * max(
        abs(median),
        EPSILON
    )

    scale = max(
        robust_scale,
        minimum_scale,
        EPSILON
    )

    return median, scale


# ------------------------------------------------------------
# FEATURE EVIDENCE
# ------------------------------------------------------------

def find_feature_evidence(
    df,
    feature,
    injection_time
):
    """
    Find the first sustained abnormal period for one feature.
    """

    if feature not in df.columns:
        return None

    times = pd.to_numeric(
        df["time"],
        errors="coerce"
    )

    values = pd.to_numeric(
        df[feature],
        errors="coerce"
    )

    valid = (
        times.notna()
        & values.notna()
    )

    times = times[valid].reset_index(drop=True)
    values = values[valid].reset_index(drop=True)

    if len(values) == 0:
        return None

    # Use only data before injection for the normal baseline.
    normal_mask = times < injection_time

    normal_values = values[normal_mask]

    if len(normal_values) < 10:
        return None

    median, scale = robust_location_scale(
        normal_values
    )

    z_scores = (
        np.abs(values - median)
        / (scale + EPSILON)
    )

    # Only inspect the first diagnosis window.
    post_injection_mask = (
        times >= injection_time
    ) & (
        times <= injection_time + DIAGNOSIS_WINDOW_SECONDS
    )

    candidate_indices = np.where(
        post_injection_mask.to_numpy()
    )[0]

    if len(candidate_indices) == 0:
        return None

    # Find sustained abnormal behaviour.
    sustained_start = None

    for index in candidate_indices:

        if z_scores.iloc[index] < Z_THRESHOLD:
            continue

        current_time = times.iloc[index]

        end_time = current_time + SUSTAINED_SECONDS

        sustained_mask = (
            (times >= current_time)
            & (times <= end_time)
            & (z_scores >= Z_THRESHOLD)
        )

        if sustained_mask.sum() >= SUSTAINED_SECONDS:
            sustained_start = current_time
            break

    if sustained_start is None:
        return None

    # Evidence is measured around the detected onset.
    evidence_end = min(
        sustained_start + 30,
        injection_time + DIAGNOSIS_WINDOW_SECONDS
    )

    evidence_mask = (
        (times >= sustained_start)
        & (times <= evidence_end)
        & (z_scores >= Z_THRESHOLD)
    )

    evidence_values = z_scores[evidence_mask]

    if len(evidence_values) == 0:
        return None

    raw_score = float(
        evidence_values.mean()
    )

    onset_seconds = float(
        sustained_start - injection_time
    )

    # Earlier abnormalities are more useful for diagnosis.
    onset_weight = 1.0 / (
        1.0 + onset_seconds / 30.0
    )

    weighted_score = (
        raw_score * onset_weight
    )

    return {
        "feature": feature,
        "raw_score": raw_score,
        "score": weighted_score,
        "onset": onset_seconds
    }


# ------------------------------------------------------------
# SERVICE / FEATURE TYPE
# ------------------------------------------------------------

def service_from_feature(feature):
    parts = feature.rsplit("_", 1)

    if len(parts) != 2:
        return None

    return parts[0]


def feature_type(feature):
    parts = feature.rsplit("_", 1)

    if len(parts) != 2:
        return None

    return parts[1].lower()


# ------------------------------------------------------------
# COLLECT ALL FEATURE EVIDENCE
# ------------------------------------------------------------

def collect_feature_evidence(
    df,
    features,
    injection_time
):

    evidence = []

    for feature in features:

        result = find_feature_evidence(
            df,
            feature,
            injection_time
        )

        if result is not None:
            result["service"] = service_from_feature(
                feature
            )

            result["type"] = feature_type(
                feature
            )

            evidence.append(result)

    return pd.DataFrame(evidence)


# ------------------------------------------------------------
# FAULT DIAGNOSIS
# ------------------------------------------------------------

def calculate_fault_evidence_from_features(
    evidence
):

    if evidence.empty:
        return None

    cpu_score = evidence.loc[
        evidence["type"] == "cpu",
        "score"
    ]

    mem_score = evidence.loc[
        evidence["type"] == "mem",
        "score"
    ]

    cpu_score = (
        cpu_score.nlargest(3).mean()
        if len(cpu_score) > 0
        else 0.0
    )

    mem_score = (
        mem_score.nlargest(3).mean()
        if len(mem_score) > 0
        else 0.0
    )

    if cpu_score >= mem_score:
        return "cpu"

    return "mem"


# ------------------------------------------------------------
# SERVICE DIAGNOSIS
# ------------------------------------------------------------

def calculate_service_evidence(
    evidence,
    predicted_fault
):
    """
    Rank services using direct fault-specific evidence.

    Direct evidence:
        service_cpu for CPU faults
        service_mem for MEM faults

    Supporting evidence:
        other telemetry from the service

    Supporting evidence is capped so that very large
    downstream values cannot overpower direct evidence.
    """

    if evidence.empty:
        return pd.DataFrame()

    services = evidence[
        "service"
    ].dropna().unique()

    results = []

    for service in services:

        service_data = evidence[
            evidence["service"] == service
        ].copy()

        # ----------------------------------------------------
        # DIRECT EVIDENCE
        # ----------------------------------------------------

        direct = service_data[
            service_data["type"] == predicted_fault
        ]

        if not direct.empty:

            direct_values = direct[
                "score"
            ].nlargest(3)

            direct_score = float(
                direct_values.mean()
            )

            earliest_direct_onset = float(
                direct["onset"].min()
            )

        else:

            direct_score = 0.0
            earliest_direct_onset = np.inf

        # ----------------------------------------------------
        # SUPPORTING EVIDENCE
        # ----------------------------------------------------

        supporting = service_data[
            service_data["type"] != predicted_fault
        ]

        if not supporting.empty:

            supporting_values = supporting[
                "score"
            ].nlargest(3)

            supporting_score = float(
                supporting_values.mean()
            )

            earliest_supporting_onset = float(
                supporting["onset"].min()
            )

        else:

            supporting_score = 0.0
            earliest_supporting_onset = np.inf

        # ----------------------------------------------------
        # IMPORTANT CHANGE
        # ----------------------------------------------------
        #
        # Supporting evidence is capped.
        #
        # This prevents values such as:
        #
        # 1.1e9
        #
        # from defeating a genuine direct CPU/MEM signal.
        # ----------------------------------------------------

        supporting_score_capped = min(
            supporting_score,
            10.0
        )

        # Direct evidence is the main signal.
        #
        # If direct evidence exists, it gets strong priority.
        #
        # If there is no direct evidence, supporting telemetry
        # receives only a small score.

        if direct_score > 0:

            service_score = (
                direct_score
                + 0.10 * supporting_score_capped
            )

            earliest_onset = (
                earliest_direct_onset
            )

        else:

            service_score = (
                0.10 * supporting_score_capped
            )

            earliest_onset = (
                earliest_supporting_onset
            )

        results.append({
            "service": service,
            "score": service_score,
            "direct_score": direct_score,
            "supporting_score": supporting_score_capped,
            "earliest_onset": earliest_onset
        })

    result_df = pd.DataFrame(results)

    if result_df.empty:
        return result_df

    # Direct evidence should be the first deciding factor.
    result_df = result_df.sort_values(
        by=[
            "direct_score",
            "score",
            "earliest_onset"
        ],
        ascending=[
            False,
            False,
            True
        ]
    ).reset_index(drop=True)

    return result_df


# ------------------------------------------------------------
# DIAGNOSE ONE CASE
# ------------------------------------------------------------

def diagnose_case(
    case_id,
    features
):

    case_dir = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "rcaeval_re1"
        / "RE1"
        / "RE1"
        / "RE1-OB"
        / case_id
    )

    df_raw, injection_time = load_case(
        case_dir
    )

    df = clean_case(
        df_raw
    )

    evidence = collect_feature_evidence(
        df,
        features,
        injection_time
    )

    predicted_fault = (
        calculate_fault_evidence_from_features(
            evidence
        )
    )

    service_scores = (
        calculate_service_evidence(
            evidence,
            predicted_fault
        )
    )

    if service_scores.empty:
        predicted_service = "unknown"

    else:
        predicted_service = (
            service_scores.iloc[0]["service"]
        )

    return (
        predicted_service,
        predicted_fault,
        service_scores
    )


# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

def evaluate_validation_cases(split="val"):

    cases = case_ids_for_split(
    split,
    group=GROUP
)

    

    features = telemetry_columns(
        cases[0]
    )

    print("=" * 70)
    print("CLOUDHEAL TEMPORAL FAILURE DIAGNOSIS")
    print("=" * 70)

    print(
        f"{split.upper()} cases: {len(cases)}"
    )

    print(
        f"Telemetry columns: {len(features)}"
    )

    print()

    correct_service = 0
    correct_fault = 0

    results = []

    for case_id in cases:

        actual_service = (
            get_service_from_case(case_id)
        )

        actual_fault = (
            get_fault_from_case(case_id)
        )

        (
            predicted_service,
            predicted_fault,
            service_scores
        ) = diagnose_case(
            case_id,
            features
        )

        if predicted_service == actual_service:
            correct_service += 1

        if predicted_fault == actual_fault:
            correct_fault += 1

        results.append({
            "case": case_id,
            "actual_service": actual_service,
            "predicted_service": predicted_service,
            "actual_fault": actual_fault,
            "predicted_fault": predicted_fault
        })

        print(
            f"{case_id:<35}"
            f" Service: {predicted_service:<22}"
            f" Fault: {predicted_fault}"
        )

    service_accuracy = (
        correct_service / len(cases)
    )

    fault_accuracy = (
        correct_fault / len(cases)
    )

    print()
    print("=" * 70)
    print(f"{split.upper()} DIAGNOSIS RESULTS")
    print("=" * 70)

    print(
        f"Service diagnosis accuracy: "
        f"{service_accuracy:.4f}"
    )

    print(
        f"Fault diagnosis accuracy:   "
        f"{fault_accuracy:.4f}"
    )

    print(
        f"Correct service diagnoses: "
        f"{correct_service}/{len(cases)}"
    )

    print(
        f"Correct fault diagnoses:   "
        f"{correct_fault}/{len(cases)}"
    )

    print()
    print("=" * 70)
    print("SERVICE DIAGNOSIS DETAILS")
    print("=" * 70)

    for item in results:

        print()
        print(
            f"Case: {item['case']}"
        )

        print(
            f"Actual service: "
            f"{item['actual_service']}"
        )

        print(
            f"Predicted service: "
            f"{item['predicted_service']}"
        )

        print(
            f"Actual fault: "
            f"{item['actual_fault']}"
        )

        print(
            f"Predicted fault: "
            f"{item['predicted_fault']}"
        )

        _, _, service_scores = diagnose_case(
            item["case"],
            features
        )

        print()

        print(
            service_scores[
                [
                    "service",
                    "score",
                    "direct_score",
                    "supporting_score",
                    "earliest_onset"
                ]
            ].head(5).to_string(
                index=False
            )
        )


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

if __name__ == "__main__":
    evaluate_validation_cases("test")