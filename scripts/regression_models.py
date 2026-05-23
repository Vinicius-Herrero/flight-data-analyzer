"""
Regression pipeline definitions, model evaluation and visualization generation.

Each function has a single responsibility and receives all required data
as explicit parameters — no global variable dependencies.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder


# =============================================================================
# PIPELINES
# =============================================================================


def build_pipeline_ridge(
    features_numeric: List[str],
    features_categorical: List[str],
    ridge_alpha: float,
) -> Pipeline:
    """
    Build the Ridge (L2-regularised linear regression) pipeline.

    Preprocessing:
    - StandardScaler on numeric features: gradient-based linear solvers require
      features on a comparable scale for stable, unbiased coefficients.
    - OneHotEncoder on categoricals: a linear model needs an independent coefficient
      per category. OrdinalEncoder would impose arbitrary ordering (AA < DL...),
      corrupting the learned weights. min_frequency=0.001 drops rare airports.

    Model:
    - alpha: L2 regularisation strength. Keeps all features but shrinks
      coefficients toward zero, which prevents overfitting on the large OHE
      feature space produced by hundreds of airport dummy variables.
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
            ("regressor", Ridge(alpha=ridge_alpha)),
        ]
    )


def build_pipeline_hgb_regressor(
    features_categorical: List[str],
    random_state: int,
) -> Pipeline:
    """
    Build the HistGradientBoostingRegressor pipeline.

    Preprocessing:
    - OrdinalEncoder on categoricals: tree splits are invariant to the scale and
      order of numeric codes, so OHE is unnecessary and slower here.
    - remainder='passthrough': numeric features pass through without scaling
      (trees do not require feature normalisation).

    Model:
    - max_depth=5: limits tree depth to prevent memorisation.
    - learning_rate=0.05: smaller, more precise boosting steps.
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
                "regressor",
                HistGradientBoostingRegressor(
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


def evaluate_regressor(
    name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute, display and return the main regression metrics.

    Metrics:
    - MAE (Mean Absolute Error): average deviation in minutes — intuitive and
      robust to outliers. Primary metric for operational interpretation.
    - RMSE (Root Mean Squared Error): penalises large errors more heavily than MAE.
      Useful for detecting models that produce extreme mispredictions.
    - R² Score: proportion of variance explained by the model. Expected to be low
      (~0.03-0.10) since pre-flight features carry limited signal for the exact
      delay magnitude — this is a known limitation of the problem, not a bug.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    metrics = {"MAE": mae, "RMSE": rmse, "R²": r2}

    print(f"\n{'=' * 40}")
    print(f"  {name}")
    print(f"{'=' * 40}")
    print(f"  MAE:  {mae:.2f} min")
    print(f"  RMSE: {rmse:.2f} min")
    print(f"  R²:   {r2:.4f}")

    return metrics


# =============================================================================
# VISUALIZATIONS
# =============================================================================


def plot_regression_performance(
    y_test: pd.Series,
    preds_ridge: np.ndarray,
    preds_hgb: np.ndarray,
    ridge_alpha: float,
    scatter_sample: int,
    random_state: int,
    viz_path: Path,
) -> None:
    """
    Generate and save regression_performance.png with a 1x2 layout.

    Each subplot shows an Actual vs Predicted scatter plot with the identity line
    (y = x, "perfect prediction"). The scatter is subsampled to scatter_sample
    points to avoid an unreadable blob and keep rendering fast.
    """
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(y_test), size=min(scatter_sample, len(y_test)), replace=False)
    y_arr = y_test.to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, preds, name, colour in [
        (axes[0], preds_ridge, f"Ridge (α={ridge_alpha})", "steelblue"),
        (axes[1], preds_hgb, "HistGradientBoosting", "mediumpurple"),
    ]:
        x_plot = y_arr[idx]
        y_plot = preds[idx]

        ax.scatter(x_plot, y_plot, alpha=0.25, s=8, color=colour, label="Observations")

        # Identity line: perfect prediction lies on y = x
        axis_max = max(x_plot.max(), y_plot.max()) * 1.05
        ax.plot([0, axis_max], [0, axis_max], "k--", lw=1.2, label="Perfect prediction")

        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Actual delay (min)", fontsize=11)
        ax.set_ylabel("Predicted delay (min)", fontsize=11)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Actual vs Predicted Delay — Regression Models (test set)\n"
        f"(scatter subsampled to {scatter_sample:,} points for readability)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    out_path = viz_path / "regression_performance.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"-> Actual vs Predicted scatter plots saved to: {out_path}")


# =============================================================================
# COMPARATIVE SUMMARY
# =============================================================================


def print_regression_summary(
    metrics_ridge: Dict[str, float],
    metrics_hgb: Dict[str, float],
) -> None:
    """
    Display the final comparison table and highlight the best model by MAE.

    MAE is the reference metric for regression on delay minutes: it is directly
    interpretable as "on average, the prediction is off by X minutes", and is
    more robust to the extreme outliers present in ARRIVAL_DELAY distributions.
    """
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)

    comparison = pd.DataFrame(
        {
            "Ridge Regression": metrics_ridge,
            "HistGradientBoosting": metrics_hgb,
        }
    ).T
    comparison.index.name = "Model"
    print(comparison.round(4).to_string())

    if metrics_hgb["MAE"] <= metrics_ridge["MAE"]:
        best_name = "HistGradientBoosting"
        best_metrics = metrics_hgb
    else:
        best_name = "Ridge Regression"
        best_metrics = metrics_ridge

    print(f"\n-> Best overall performance (MAE): {best_name}")
    print(
        f"   MAE={best_metrics['MAE']:.2f} min"
        f" | RMSE={best_metrics['RMSE']:.2f} min"
        f" | R²={best_metrics['R²']:.4f}"
    )

    print("\nInterpretability vs. performance trade-off:")
    print(
        "   • Ridge: linear coefficients per feature — directly shows which routes,\n"
        "     airlines and departure hours contribute most to predicted delay magnitude."
    )
    print(
        "   • HistGradientBoosting: captures non-linear interactions between features\n"
        "     (e.g. airport × time-of-day) — generally lower MAE, but requires\n"
        "     permutation importance or SHAP for interpretability."
    )
    print(
        "\n   NOTE: low R² is expected in flight delay regression. Pre-departure features\n"
        "   (route, schedule, season) explain delay patterns at a population level but\n"
        "   cannot capture the actual cause of a specific delay (weather, crew, ATC).\n"
        "   MAE and RMSE are the operationally relevant metrics here."
    )
