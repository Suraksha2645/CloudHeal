from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent

# Main folders
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# RCAEval RE1
RCAEval_DIR = RAW_DATA_DIR / "rcaeval_re1" / "RE1" / "RE1"
RE1_OB_DIR = RCAEval_DIR / "RE1-OB"

# Processed RCAEval files
RCAEval_PROCESSED_DIR = PROCESSED_DATA_DIR / "rcaeval_re1"
MANIFEST_FILE = RCAEval_PROCESSED_DIR / "manifest_re1_ob.csv"

# Source code
SRC_DIR = PROJECT_ROOT / "src"
DATA_CODE_DIR = SRC_DIR / "data"
MODELS_DIR = SRC_DIR / "models"