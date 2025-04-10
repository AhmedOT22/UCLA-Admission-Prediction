from pathlib import Path

# === PATHS ===
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "Admission.csv"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed" / "Admission_processed.csv"
MODEL_DIR = BASE_DIR / "models"

# === TRAINING CONFIGURATION ===
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 100
BATCH_SIZE = 10
INPUT_DIM = 8  # Update if different after feature check
