from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()

SIMPLIFIED_DIR = PROJECT_DIR / "simplified"
COMPLETE_DIR = PROJECT_DIR / "complete"

SRC_DIR = PROJECT_DIR / "src"

# Complete Model directories
DATA_DIR = COMPLETE_DIR / "data"
MODEL_DIR = COMPLETE_DIR / "models"
REPORT_DIR = COMPLETE_DIR / "reports"

# data
RAW_DATA_DIR = DATA_DIR / "raw"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


# Simplified Model directories
DATA_DIR_S = SIMPLIFIED_DIR / "data"
MODEL_DIR_S = SIMPLIFIED_DIR / "models"
REPORT_DIR_S = SIMPLIFIED_DIR / "reports"

# data
RAW_DATA_DIR_S = DATA_DIR_S / "raw"
EXTERNAL_DATA_DIR_S = DATA_DIR_S / "external"
INTERIM_DATA_DIR_S = DATA_DIR_S / "interim"
PROCESSED_DATA_DIR_S = DATA_DIR_S / "processed"
