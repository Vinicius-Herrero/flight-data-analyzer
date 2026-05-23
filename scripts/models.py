"""
ML pipeline definitions, model evaluation and visualization generation.

Each function has a single responsibility and receives all required data
as explicit parameters — no global variable dependencies.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder


# =============================================================================
# PIPELINES
# =============================================================================


def build_pipeline_sgd(
    features_numeric: List[str],
    features_categorical: List[str],
    random_state: int,
) -> Pipeline:
    """
    Build the SGDClassifier (Logistic Regression) pipeline.

    Preprocessing:
    - StandardScaler on numeric features: gradient descent requires features on a
      comparable scale.
    - OneHotEncoder on categorical features: a linear model needs independent
      coefficients per category. OrdinalEncoder would impose an arbitrary order
      (AA < DL...), corrupting the weights. min_frequency=0.001 drops rare airports
      (< 0.1% of samples).

    Model:
    - class_weight='balanced': compensates the natural class imbalance (~81% on-time).
    - early_stopping + tol + n_iter_no_change: controlled convergence criterion.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), features_numeric),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=0.001,
                    sparse_output=False,
                ),
                features_categorical,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                SGDClassifier(
                    loss="log_loss",
                    class_weight="balanced",
                    max_iter=1000,
                    tol=1e-3,
                    n_iter_no_change=10,
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_pipeline_hgb(
    features_categorical: List[str],
    random_state: int,
) -> Pipeline:
    """
    Build the HistGradientBoostingClassifier pipeline.

    Preprocessing:
    - OrdinalEncoder on categorical features: tree-based models are invariant to the
      scale and order of numeric codes — OHE is unnecessary and slower here.
    - remainder='passthrough': numeric features pass through without transformation
      (trees do not require feature scaling).

    Model:
    - max_depth=5: limits tree depth to prevent memorisation.
    - learning_rate=0.05: smaller, more precise boosting steps.
    - class_weight='balanced': compensates the natural class imbalance.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                features_categorical,
            ),
        ],
        remainder="passthrough",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    class_weight="balanced",
                    max_iter=150,
                    max_depth=5,
                    learning_rate=0.05,
                    random_state=random_state,
                ),
            ),
        ]
    )


# =============================================================================
# EVALUATION
# =============================================================================


def evaluate_model(
    name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, float]:
    """
    Compute, display and return the main classification metrics.

    Reported metrics:
    - ROC-AUC: model discrimination power, independent of the chosen threshold.
    - Precision / Recall / F1: performance at the chosen operating threshold.
      F1 is the reference metric because it balances precision and recall on
      imbalanced datasets.
    """
    metrics = {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    print(f"\n{'=' * 40}")
    print(f"  {name}")
    print(f"{'=' * 40}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-score:  {metrics['f1']:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["No Delay", "Delayed"]))

    return metrics


# =============================================================================
# VISUALIZATIONS
# =============================================================================


def _operating_point(
    prec: np.ndarray,
    rec: np.ndarray,
    thresh: np.ndarray,
    chosen: float,
) -> Tuple[float, float]:
    """Return (recall, precision) at the threshold closest to the chosen value."""
    idx = np.argmin(np.abs(thresh - chosen))
    return rec[idx], prec[idx]


def plot_model_performance(
    y_test: pd.Series,
    preds_lr: np.ndarray,
    preds_hgb: np.ndarray,
    proba_lr: np.ndarray,
    proba_hgb: np.ndarray,
    sgd_threshold: float,
    hgb_threshold: float,
    viz_path: Path,
) -> None:
    """
    Generate and save model_performance.png with a 2x2 layout:
    - Top row: confusion matrices (SGD and HGB).
    - Bottom row: Precision-Recall curve for both models with the chosen operating
      threshold marked — visually justifies the threshold selection.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Confusion matrices
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        preds_lr,
        ax=axes[0, 0],
        cmap="Blues",
        display_labels=["No Delay", "Delayed"],
        colorbar=False,
    )
    axes[0, 0].set_title(
        f"Logistic Regression (threshold={sgd_threshold})",
        fontsize=12,
        fontweight="bold",
    )
    axes[0, 0].set_xlabel("Predicted")
    axes[0, 0].set_ylabel("Actual")

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        preds_hgb,
        ax=axes[0, 1],
        cmap="Purples",
        display_labels=["No Delay", "Delayed"],
        colorbar=False,
    )
    axes[0, 1].set_title(
        f"HistGradientBoosting (threshold={hgb_threshold})",
        fontsize=12,
        fontweight="bold",
    )
    axes[0, 1].set_xlabel("Predicted")
    axes[0, 1].set_ylabel("Actual")

    # Precision-Recall curve (bottom row, spanning 2 columns)
    ax_pr = plt.subplot2grid((2, 2), (1, 0), colspan=2, fig=fig)

    prec_lr, rec_lr, thresh_lr = precision_recall_curve(y_test, proba_lr)
    ap_lr = average_precision_score(y_test, proba_lr)
    ax_pr.plot(
        rec_lr, prec_lr, color="steelblue", lw=2,
        label=f"SGDClassifier (AP={ap_lr:.3f})",
    )

    prec_hgb, rec_hgb, thresh_hgb = precision_recall_curve(y_test, proba_hgb)
    ap_hgb = average_precision_score(y_test, proba_hgb)
    ax_pr.plot(
        rec_hgb, prec_hgb, color="mediumpurple", lw=2,
        label=f"HistGradientBoosting (AP={ap_hgb:.3f})",
    )

    op_rec_lr, op_prec_lr = _operating_point(prec_lr, rec_lr, thresh_lr, sgd_threshold)
    op_rec_hgb, op_prec_hgb = _operating_point(prec_hgb, rec_hgb, thresh_hgb, hgb_threshold)

    ax_pr.scatter(
        op_rec_lr, op_prec_lr, s=100, color="steelblue", zorder=5,
        label=f"SGD threshold={sgd_threshold}",
    )
    ax_pr.scatter(
        op_rec_hgb, op_prec_hgb, s=100, color="mediumpurple", zorder=5,
        label=f"HGB threshold={hgb_threshold}",
    )

    baseline = y_test.mean()
    ax_pr.axhline(
        baseline, color="gray", linestyle="--", alpha=0.7,
        label=f"Baseline (random) = {baseline:.2f}",
    )

    ax_pr.set_xlabel("Recall", fontsize=11)
    ax_pr.set_ylabel("Precision", fontsize=11)
    ax_pr.set_title(
        "Precision-Recall Curve — model comparison\n"
        "(marked point = chosen operating threshold)",
        fontsize=12,
        fontweight="bold",
    )
    ax_pr.legend(fontsize=9)
    ax_pr.grid(alpha=0.3)

    fig.suptitle(
        "Model Comparison — Test Set",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()

    out_path = viz_path / "model_performance.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"-> Confusion matrices + Precision-Recall curve saved to: {out_path}")


def plot_feature_importance(
    pipeline_hgb: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    perm_sample: int,
    perm_n_repeats: int,
    random_state: int,
    viz_path: Path,
) -> None:
    """
    Compute permutation importance on HistGradientBoosting and save feature_importance.png.

    Permutation importance is model-agnostic (treats the model as a black box) and
    measures the drop in F1 when each feature is shuffled — the larger the drop, the
    more relevant the feature. A subsample of the test set is used to keep runtime
    feasible on a local machine.
    """
    if len(X_test) > perm_sample:
        idx = np.random.RandomState(random_state).choice(
            len(X_test), size=perm_sample, replace=False
        )
        X_perm = X_test.iloc[idx]
        y_perm = y_test.iloc[idx]
    else:
        X_perm, y_perm = X_test, y_test

    result = permutation_importance(
        pipeline_hgb,
        X_perm,
        y_perm,
        n_repeats=perm_n_repeats,
        random_state=random_state,
        scoring="f1",
        n_jobs=-1,
    )

    feature_names = pipeline_hgb.named_steps["preprocessor"].get_feature_names_out()
    importances = result.importances_mean
    top_k = 10
    top_indices = np.argsort(importances)[::-1][:top_k]
    top_names = [feature_names[i] for i in top_indices]
    top_values = importances[top_indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        range(top_k),
        top_values[::-1],
        color=plt.cm.viridis(np.linspace(0.3, 0.9, top_k)),
        edgecolor="white",
    )
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(top_names[::-1])
    ax.set_xlabel("Mean importance (F1 drop when permuted)", fontsize=11)
    ax.set_title(
        "Top 10 Features — Permutation Importance\n(HistGradientBoosting, scoring=F1)",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, top_values[::-1]):
        ax.text(
            bar.get_width() + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    out_path = viz_path / "feature_importance.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"-> Feature importance chart saved to: {out_path}")


# =============================================================================
# COMPARATIVE SUMMARY
# =============================================================================


def print_summary(
    metrics_lr: Dict[str, float],
    metrics_hgb: Dict[str, float],
    sgd_threshold: float,
    hgb_threshold: float,
) -> None:
    """
    Display the final comparison table and highlight the best model by F1 score.

    F1 is the reference metric for comparing models on imbalanced classes: it
    penalises low precision and low recall equally, reflecting real-world performance
    on a dataset with ~81% on-time flights.
    """
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)

    comparison = pd.DataFrame(
        {
            f"SGDClassifier (t={sgd_threshold})": metrics_lr,
            f"HistGradientBoosting (t={hgb_threshold})": metrics_hgb,
        }
    ).T
    print(comparison.round(4).to_string())

    if metrics_hgb["f1"] >= metrics_lr["f1"]:
        best_name = "HistGradientBoosting"
        best_metrics = metrics_hgb
    else:
        best_name = "SGDClassifier (Logistic Regression)"
        best_metrics = metrics_lr

    print(f"\n-> Best overall performance (F1): {best_name}")
    print(f"   F1={best_metrics['f1']:.4f} | ROC-AUC={best_metrics['roc_auc']:.4f}")

    print("\nInterpretability vs. performance trade-off:")
    print(
        "   • SGDClassifier: linear coefficients, fast, more interpretable — "
        "useful for explaining the direction of feature effects."
    )
    print(
        "   • HistGradientBoosting: captures non-linearities and interactions — "
        "generally higher accuracy, but black-box (use permutation importance)."
    )
