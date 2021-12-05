from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()

SIMPLIFIED_DIR = PROJECT_DIR / "simplified"
COMPLETE_DIR = PROJECT_DIR / "complete"

SRC_DIR = PROJECT_DIR / "src"

# Complete Model directories
DATA_DIR = COMPLETE_DIR / "data"
MODEL_DIR = COMPLETE_DIR / "models"
REPORT_DIR = COMPLETE_DIR / "reports"

# Complete Data directories
RAW_DATA_DIR = DATA_DIR / "raw"
COMPLEMENTARY_DATA_DIR = DATA_DIR / "complementary"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


# Simplified Model directories
DATA_DIR_S = SIMPLIFIED_DIR / "data"
MODEL_DIR_S = SIMPLIFIED_DIR / "models"
REPORT_DIR_S = SIMPLIFIED_DIR / "reports"

# Simplified Data directories
RAW_DATA_DIR_S = DATA_DIR_S / "raw"
COMPLEMENTARY_DATA_DIR_S = DATA_DIR_S / "complementary"
INTERIM_DATA_DIR_S = DATA_DIR_S / "interim"
PROCESSED_DATA_DIR_S = DATA_DIR_S / "processed"
