from pathlib import Path
import pandas as pd


# ============================================================
# 1. PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(r"C:\CloudHeal")


# ============================================================
# 2. ONLINE BOUTIQUE DATASET PATH
# ============================================================

OB_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "rcaeval_re1"
    / "RE1"
    / "RE1"
    / "RE1-OB"
)


# ============================================================
# 3. FAULT CASE INFORMATION
# ============================================================

SERVICES = [
    "adservice",
    "cartservice",
    "checkoutservice",
    "currencyservice",
    "productcatalogservice",
]

FAULT_TYPES = [
    "cpu",
    "mem",
    "disk",
    "delay",
    "loss",
]


# ============================================================
# 4. INSPECT ALL ONLINE BOUTIQUE CASES
# ============================================================

print("===== RE1 ONLINE BOUTIQUE CASE INSPECTION =====")

total_cases = 0

for service in SERVICES:

    print("\n==============================================")
    print("SERVICE:", service)
    print("==============================================")

    for fault in FAULT_TYPES:

        case_name = f"{service}_{fault}"
        case_path = OB_PATH / case_name

        print(f"\n--- {case_name} ---")

        for repetition in range(1, 6):

            data_file = (
                case_path
                / str(repetition)
                / "data.csv"
            )

            injection_file = (
                case_path
                / str(repetition)
                / "inject_time.txt"
            )

            if not data_file.exists():
                print(
                    f"Repetition {repetition}: "
                    "data.csv not found"
                )
                continue

            # Load only this one case
            df = pd.read_csv(data_file)

            # Read injection timestamp
            injection_time = int(
                injection_file.read_text().strip()
            )

            total_cases += 1

            # Find first and last timestamp
            start_time = df["time"].iloc[0]
            end_time = df["time"].iloc[-1]

            # Number of rows before and after injection
            normal_rows = len(
                df[df["time"] < injection_time]
            )

            post_fault_rows = len(
                df[df["time"] >= injection_time]
            )

            print(
                f"Repetition {repetition}: "
                f"rows={len(df)}, "
                f"normal={normal_rows}, "
                f"post_fault={post_fault_rows}, "
                f"duration={end_time - start_time}s"
            )


# ============================================================
# 5. FINAL SUMMARY
# ============================================================

print("\n==============================================")
print("FINAL SUMMARY")
print("==============================================")

print("Services:", len(SERVICES))
print("Fault types:", len(FAULT_TYPES))
print("Repetitions per case:", 5)
print("Total cases inspected:", total_cases)