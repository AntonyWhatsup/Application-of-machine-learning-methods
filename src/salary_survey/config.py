from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "Data_Professional_Salary_Survey_Responses.xlsx"
DATA_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_data.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
RANDOM_STATE = 42

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
