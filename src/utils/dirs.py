from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()

# local directories
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
SRC_DIR = PROJECT_DIR / "src"

# data
RAW_DATA_DIR = DATA_DIR / "raw"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# model
EXPERIMENT_MODEL_DIR = MODEL_DIR / "experiments"
REGISTERED_MODEL_DIR = MODEL_DIR / "registered"
