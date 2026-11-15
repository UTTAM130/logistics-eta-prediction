from pathlib import Path

# Root project directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Machine learning model directory
MODEL_DIR = ROOT_DIR / "models"


def ensure_dirs():
    """Create all required project directories."""
    directories = [
        RAW_DIR,
        PROCESSED_DIR,
        MODEL_DIR
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
