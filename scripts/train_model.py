"""
Orchestrator for the supervised ML pipeline for flight delay prediction.

Run from the project root:
    python scripts/train_model.py

Modules used:
    config.py          — constants, paths and feature lists
    data_processing.py — loading, cleaning, feature engineering and split
    models.py          — pipelines, evaluation, visualizations and summary
"""

import sys
from pathlib import Path

# Ensures relative imports work when running directly from the terminal
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    COLUMNS_TO_USE,
    DB_PATH,
    FEATURES_CATEGORICAL,
    FEATURES_NUMERIC,
    HGB_DECISION_THRESHOLD,
    PERM_IMPORTANCE_N_REPEATS,
    PERM_IMPORTANCE_SAMPLE,
    RANDOM_STATE,
    SAMPLE_SIZE,
    SGD_DECISION_THRESHOLD,
    TEST_SIZE,
    VIZ_PATH,
)
from data_processing import clean_data, feature_engineering, load_flights, sample_and_split
from models import (
    build_pipeline_hgb,
    build_pipeline_sgd,
    evaluate_model,
    plot_feature_importance,
    plot_model_performance,
    print_summary,
)

# =============================================================================
# START
# =============================================================================

print("=" * 80)
print("MACHINE LEARNING MODEL TRAINING - SUPERVISED LEARNING")
print("=" * 80)

VIZ_PATH.mkdir(exist_ok=True)

# =============================================================================
# 1. DATA LOADING
# =============================================================================

print("\n[1] DATA LOADING")
print("-" * 80)
print("-> Loading flight dataset (only required columns to save RAM)...")

flights = load_flights(DB_PATH, COLUMNS_TO_USE)

# =============================================================================
# 2. DATA CLEANING
# =============================================================================

print("\n[2] DATA CLEANING")
print("-" * 80)
print("-> Filtering cancelled and diverted flights (no useful arrival delay data)...")

flights = clean_data(flights)

# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================

print("\n[3] FEATURE ENGINEERING")
print("-" * 80)
print("-> Creating derived features using only pre-departure information...")

X, y = feature_engineering(flights, FEATURES_NUMERIC, FEATURES_CATEGORICAL)

# =============================================================================
# 4. SAMPLING & SPLIT
# =============================================================================

print("\n[4] SAMPLING & SPLIT")
print("-" * 80)

delay_pct = y.mean() * 100
print("-> Target distribution in filtered dataset:")
print(f"   Delayed flights  (IS_DELAYED=1): {delay_pct:.2f}%")
print(f"   On-time flights  (IS_DELAYED=0): {(100 - delay_pct):.2f}%")

print(f"-> Extracting stratified sample of {SAMPLE_SIZE:,} records...")

X_train, X_test, y_train, y_test, y_sample = sample_and_split(
    X, y, SAMPLE_SIZE, TEST_SIZE, RANDOM_STATE
)

print(f"   Delayed proportion in sample: {y_sample.mean() * 100:.2f}%")
print(f"   Training records: {X_train.shape[0]:,}")
print(f"   Test records:     {X_test.shape[0]:,}")
print(f"   Delayed in train: {y_train.mean() * 100:.2f}%")
print(f"   Delayed in test:  {y_test.mean() * 100:.2f}%")

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

print("-> Model 1: Logistic Regression via SGDClassifier...")
pipeline_lr = build_pipeline_sgd(FEATURES_NUMERIC, FEATURES_CATEGORICAL, RANDOM_STATE)
pipeline_lr.fit(X_train, y_train)

print("-> Model 2: HistGradientBoostingClassifier...")
pipeline_hgb = build_pipeline_hgb(FEATURES_CATEGORICAL, RANDOM_STATE)
pipeline_hgb.fit(X_train, y_train)

# =============================================================================
# 7. EVALUATION
# =============================================================================

print("\n[7] EVALUATION")
print("-" * 80)

proba_lr = pipeline_lr.predict_proba(X_test)[:, 1]
preds_lr = (proba_lr >= SGD_DECISION_THRESHOLD).astype(int)
metrics_lr = evaluate_model(
    f"Logistic Regression (SGDClassifier, threshold={SGD_DECISION_THRESHOLD})",
    y_test,
    preds_lr,
    proba_lr,
)

proba_hgb = pipeline_hgb.predict_proba(X_test)[:, 1]
preds_hgb = (proba_hgb >= HGB_DECISION_THRESHOLD).astype(int)
metrics_hgb = evaluate_model(
    f"HistGradientBoosting (threshold={HGB_DECISION_THRESHOLD})",
    y_test,
    preds_hgb,
    proba_hgb,
)

# =============================================================================
# 8. VISUALIZATIONS
# =============================================================================

print("\n[8] VISUALIZATIONS")
print("-" * 80)

plot_model_performance(
    y_test,
    preds_lr,
    preds_hgb,
    proba_lr,
    proba_hgb,
    SGD_DECISION_THRESHOLD,
    HGB_DECISION_THRESHOLD,
    VIZ_PATH,
)

print("-> Computing permutation importance (subsample for local execution)...")
plot_feature_importance(
    pipeline_hgb,
    X_test,
    y_test,
    PERM_IMPORTANCE_SAMPLE,
    PERM_IMPORTANCE_N_REPEATS,
    RANDOM_STATE,
    VIZ_PATH,
)

# =============================================================================
# 9. COMPARATIVE SUMMARY
# =============================================================================

print_summary(metrics_lr, metrics_hgb, SGD_DECISION_THRESHOLD, HGB_DECISION_THRESHOLD)

print("\n" + "=" * 80)
print("Pipeline completed successfully.")
print("=" * 80)
