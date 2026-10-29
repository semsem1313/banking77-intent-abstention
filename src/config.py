"""Shared paths and constants. Values mirror PREREGISTRATION.md; change both together (via an amendment)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
DOCS_ASSETS = ROOT / "docs" / "assets"

TRAIN_CSV = DATA_RAW / "train.csv"
TEST_CSV = DATA_RAW / "test.csv"  # read ONLY by src/evaluate_final.py

SEED = 335
SEEDS = [335, 336, 337, 338, 339]
UNSEEN_SEED = 2026
N_UNSEEN_INTENTS = 10
N_FOLDS = 5
NEAR_DUP_COSINE = 0.90
SENSITIVITY_NEAR_DUP_COSINE = 0.80  # amendment 2026-10-29: sensitivity check only
N_BOOTSTRAP = 1000
ECE_BINS = 15
COST_RATIOS = [2, 5, 10]
COVERAGE_GRID = [round(0.50 + 0.05 * i, 2) for i in range(10)]  # 0.50 ... 0.95
LEARNING_CURVE_FRACTIONS = [0.05, 0.10, 0.25, 0.50, 1.00]
LEARNING_CURVE_DRAWS = 5
