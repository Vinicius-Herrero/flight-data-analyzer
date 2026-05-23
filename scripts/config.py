"""
Central configuration for the supervised ML pipeline.

All constants, paths and feature lists live here so that
data_processing.py, models.py and train_model.py share the same source of truth.
"""

from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================

DB_PATH = Path("database")
VIZ_PATH = Path("visualizations")

# =============================================================================
# SAMPLING AND SPLIT
# =============================================================================

RANDOM_STATE = 42
SAMPLE_SIZE = 300_000
TEST_SIZE = 0.30

# =============================================================================
# DECISION THRESHOLDS
# =============================================================================

SGD_DECISION_THRESHOLD = 0.40
# Threshold below the standard (0.5): with class_weight='balanced', SGD pushes
# probabilities upward and over-predicts "delayed" (inflated recall, low precision).
# Lowering the cutoff to 0.40 rebalances precision/recall (F1).

HGB_DECISION_THRESHOLD = 0.60
# Threshold above the standard (0.5): with class_weight='balanced', HistGradientBoosting
# tends to predict more delays (high recall, low precision). Raising the cutoff to 0.60
# improves the precision/recall balance for the operational use case (F1).

# =============================================================================
# PERMUTATION IMPORTANCE
# =============================================================================

PERM_IMPORTANCE_SAMPLE = 15_000
PERM_IMPORTANCE_N_REPEATS = 5

# =============================================================================
# COLUMNS AND FEATURES
# =============================================================================

# Columns loaded from CSV (saves RAM — only what is strictly needed)
COLUMNS_TO_USE = [
    "MONTH",
    "DAY",
    "DAY_OF_WEEK",
    "AIRLINE",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT",
    "DISTANCE",
    "SCHEDULED_DEPARTURE",
    "ARRIVAL_DELAY",
    "CANCELLED",
    "DIVERTED",
]

# Pre-flight numeric features (no post-departure data allowed)
FEATURES_NUMERIC = [
    "MONTH",
    "DAY",
    "DAY_OF_WEEK",
    "DEP_HOUR",
    "DEP_MINUTE",
    "DISTANCE",
    "IS_WEEKEND",
]

# Pre-flight categorical features
FEATURES_CATEGORICAL = [
    "AIRLINE",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT",
    "PERIOD_OF_DAY",
]
