"""
Orchestrator for the regression pipeline for continuous flight delay prediction.

Predicts ARRIVAL_DELAY in minutes for flights that actually arrived late.
Run from the project root:
    python scripts/train_regression.py

Modules used:
    config.py                  — constants, paths and feature lists
    regression_processing.py   — loading, cleaning, feature engineering and split
    regression_models.py       — pipelines, evaluation, visualizations and summary
"""

import sys
import numpy as np
from pathlib import Path

# Ensures relative imports work when running directly from the terminal
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    COLUMNS_TO_USE,
    DB_PATH,
    FEATURES_CATEGORICAL,
    FEATURES_NUMERIC,
    RANDOM_STATE,
    SAMPLE_SIZE,
    TEST_SIZE,
    VIZ_PATH,
)
from regression_processing import (
    clean_data_regression,
    feature_engineering_regression,
    load_flights_regression,
    sample_and_split_regression,
)
from regression_models import (
    build_pipeline_hgb_regressor,
    build_pipeline_ridge,
    evaluate_regressor,
    plot_regression_performance,
    print_regression_summary,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

RIDGE_ALPHA = 1.0
SCATTER_SAMPLE = 5_000

# =============================================================================
# START
# =============================================================================

print("=" * 80)
print("MACHINE LEARNING MODEL TRAINING - REGRESSION (DELAY IN MINUTES)")
print("=" * 80)

VIZ_PATH.mkdir(exist_ok=True)

# =============================================================================
# 1. DATA LOADING
# =============================================================================

print("\n[1] DATA LOADING")
print("-" * 80)
print("-> Loading flight dataset (explicit dtypes to prevent DtypeWarning and save RAM)...")

flights = load_flights_regression(DB_PATH, COLUMNS_TO_USE)
print(f"   Records loaded: {len(flights):,}")

# =============================================================================
# 2. DATA CLEANING
# =============================================================================

print("\n[2] DATA CLEANING")
print("-" * 80)
print("-> Removing cancelled and diverted flights...")
print("-> Filtering to flights with ARRIVAL_DELAY > 0 (only truly late arrivals)...")
print("-> Capping ARRIVAL_DELAY at the 99th percentile to remove unpredictable outliers...")

flights = clean_data_regression(flights)
print(f"   Records after cleaning: {len(flights):,}")
print(f"   Target range: {flights['ARRIVAL_DELAY'].min():.0f} – {flights['ARRIVAL_DELAY'].max():.0f} min")

# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================

print("\n[3] FEATURE ENGINEERING")
print("-" * 80)
print("-> Creating derived features using only pre-departure information...")
print("-> Applying log1p transformation to ARRIVAL_DELAY target...")

X, y = feature_engineering_regression(flights, FEATURES_NUMERIC, FEATURES_CATEGORICAL)

# =============================================================================
# 4. SAMPLING & SPLIT
# =============================================================================

print("\n[4] SAMPLING & SPLIT")
print("-" * 80)
print("-> Target distribution in filtered dataset (log1p scale):")
print(f"   Mean delay:    {y.mean():.3f}")
print(f"   Median delay:  {y.median():.3f}")
print(f"   Std deviation: {y.std():.3f}")

print(f"-> Extracting random sample of {min(SAMPLE_SIZE, len(X)):,} records...")

X_train, X_test, y_train, y_test, y_sample = sample_and_split_regression(
    X, y, SAMPLE_SIZE, TEST_SIZE, RANDOM_STATE
)

print(f"   Mean delay in sample (log1p): {y_sample.mean():.3f}")
print(f"   Training records: {X_train.shape[0]:,}")
print(f"   Test records:     {X_test.shape[0]:,}")

# =============================================================================
# 5. PREPROCESSING (embedded inside each pipeline)
# =============================================================================

print("\n[5] PREPROCESSING")
print("-" * 80)

# =============================================================================
# 6. TRAINING
# =============================================================================

print("\n[6] TRAINING")
print("-" * 80)

print("-> Model 1: Ridge Regression...")
pipeline_ridge = build_pipeline_ridge(FEATURES_NUMERIC, FEATURES_CATEGORICAL, RIDGE_ALPHA)
pipeline_ridge.fit(X_train, y_train)

print("-> Model 2: HistGradientBoostingRegressor...")
pipeline_hgb = build_pipeline_hgb_regressor(FEATURES_CATEGORICAL, RANDOM_STATE)
pipeline_hgb.fit(X_train, y_train)

# =============================================================================
# 7. EVALUATION
# =============================================================================

print("\n[7] EVALUATION")
print("-" * 80)

# Predições em escala log1p (mesma escala do treinamento)
preds_ridge_log = pipeline_ridge.predict(X_test)
preds_hgb_log   = pipeline_hgb.predict(X_test)

# Métricas na escala log1p — refletem o que o modelo realmente aprendeu
metrics_ridge = evaluate_regressor("Ridge Regression", y_test, preds_ridge_log)
metrics_hgb   = evaluate_regressor("HistGradientBoostingRegressor", y_test, preds_hgb_log)

# Revertidos apenas para o gráfico (escala de minutos para visualização)
preds_ridge_orig = np.expm1(preds_ridge_log)
preds_hgb_orig   = np.expm1(preds_hgb_log)
y_test_orig      = np.expm1(y_test)

# =============================================================================
# 8. VISUALIZATIONS
# =============================================================================

print("\n[8] VISUALIZATIONS")
print("-" * 80)

plot_regression_performance(
    y_test_orig,
    preds_ridge_orig,
    preds_hgb_orig,
    RIDGE_ALPHA,
    SCATTER_SAMPLE,
    RANDOM_STATE,
    VIZ_PATH,
)

# =============================================================================
# 9. COMPARATIVE SUMMARY
# =============================================================================

print_regression_summary(metrics_ridge, metrics_hgb)

print("\n" + "=" * 80)
print("Pipeline completed successfully.")
print("=" * 80)