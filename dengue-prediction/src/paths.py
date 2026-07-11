"""
Central path configuration.

Previously, main.py used root-relative paths ('data/raw/...') while
gis_mapper.py, model.py, and dashboard.py used paths relative to being
run from inside src/ ('../data/...'). That only works if every script is
launched from exactly the right working directory. Anchoring everything
to this file's location removes that fragility - it doesn't matter where
you run `python main.py` or `streamlit run src/dashboard.py` from anymore.
"""
from pathlib import Path

# src/paths.py -> project root is one level up
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_FILE = DATA_DIR / "raw" / "historical_patients.xlsx"
POP_DATA_FILE = DATA_DIR / "population" / "vnm_admpop_adm1_2024.csv"
BOUNDARY_FILE = DATA_DIR / "population" / "vietnam_34_provinces_2025.geojson"
PROCESSED_DIR = DATA_DIR / "processed"
HISTORICAL_CSV = PROCESSED_DIR / "historical.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODELS_DIR / "dengue_xgboost.pkl"

OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "maps"
MAP_OUTPUT_FILE = OUTPUTS_DIR / "dengue_risk_map.geojson"