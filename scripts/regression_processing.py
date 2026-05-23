"""
Data loading, cleaning, feature engineering and split functions for the regression pipeline.

All functions are pure: they receive explicit parameters and return data,
with no I/O side-effects — print statements are the orchestrator's responsibility.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_flights_regression(db_path: Path, columns_to_use: List[str]) -> pd.DataFrame:
    """
    Load the flights CSV with explicit dtypes to prevent DtypeWarning and reduce RAM.

    Explicit types prevent pandas from inferring mixed-type columns and allow the
    smallest adequate numeric representation per column (int8, float32, etc.).
    """
    return pd.read_csv(
        db_path / "flights.csv",
        usecols=columns_to_use,
        dtype={
            "MONTH": "int8",
            "DAY": "int8",
            "DAY_OF_WEEK": "int8",
            "AIRLINE": str,
            "ORIGIN_AIRPORT": str,
            "DESTINATION_AIRPORT": str,
            "DISTANCE": "float32",
            "SCHEDULED_DEPARTURE": "int32",
            "ARRIVAL_DELAY": "float32",
            "CANCELLED": "int8",
            "DIVERTED": "int8",
        },
    )


def clean_data_regression(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove cancelled and diverted flights, keep only truly delayed arrivals, and
    drop rows with missing ARRIVAL_DELAY.

    Filtering to ARRIVAL_DELAY > 0 scopes the problem to "given a flight that will
    be late, how late will it be?" — mixing on-time or early arrivals (magnitude = 0)
    would conflate the classification question with the regression magnitude question.

    IMPORTANT: ARRIVAL_DELAY is used as the regression target only — it never enters
    as a feature, which would cause direct data leakage.
    """
    df = df[(df["CANCELLED"] == 0) & (df["DIVERTED"] == 0)]
    df = df[df["ARRIVAL_DELAY"] > 0].copy()
    df = df.dropna(subset=["ARRIVAL_DELAY"])
    return df


def _assign_period_of_day(hour: pd.Series) -> pd.Series:
    """Map departure hour to a qualitative period of day (known at scheduling time)."""
    conditions = [
        (hour >= 0) & (hour <= 4),
        (hour >= 5) & (hour <= 11),
        (hour >= 12) & (hour <= 17),
        (hour >= 18) & (hour <= 23),
    ]
    choices = ["dawn", "morning", "afternoon", "night"]
    return pd.Series(
        np.select(conditions, choices, default="night"), index=hour.index
    )


def feature_engineering_regression(
    df: pd.DataFrame,
    features_numeric: List[str],
    features_categorical: List[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply feature engineering using only information available before departure.

    New columns created:
    - DEP_HOUR / DEP_MINUTE: SCHEDULED_DEPARTURE as a raw integer (e.g. 2355) is not
      linearly ordinal — splitting into hour and minute lets the model capture
      circadian delay patterns correctly.
    - IS_WEEKEND: binary flag for Saturday (6) and Sunday (7), which have a different
      traffic profile compared to weekdays.
    - PERIOD_OF_DAY: qualitative period (dawn/morning/afternoon/night) to capture
      non-linear time-of-day effects that a linear model can exploit via OHE.

    Returns (X, y) where y = ARRIVAL_DELAY (continuous float in minutes).
    """
    df = df.copy()

    # Decompose scheduled departure time into hour and minute
    df["DEP_HOUR"] = df["SCHEDULED_DEPARTURE"] // 100
    df["DEP_MINUTE"] = df["SCHEDULED_DEPARTURE"] % 100

    # DAY_OF_WEEK 6=Saturday, 7=Sunday (US flight dataset convention)
    df["IS_WEEKEND"] = df["DAY_OF_WEEK"].isin([6, 7]).astype("int8")

    df["PERIOD_OF_DAY"] = _assign_period_of_day(df["DEP_HOUR"])

    X = df[features_numeric + features_categorical].copy()
    y = df["ARRIVAL_DELAY"]  # continuous regression target (minutes)

    X[features_categorical] = X[features_categorical].astype(str)

    return X, y


def sample_and_split_regression(
    X: pd.DataFrame,
    y: pd.Series,
    sample_size: int,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Draw a random sample then perform a plain train/test split.

    Regression targets are continuous — stratify is not applicable here.
    A plain random sample preserves the delay distribution without stratification.

    Returns: (X_train, X_test, y_train, y_test, y_sample)
    y_sample is returned so the orchestrator can report summary statistics of the sample.
    """
    n_sample = min(sample_size, len(X))

    X_sample, _, y_sample, _ = train_test_split(
        X, y, train_size=n_sample, random_state=random_state
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, y_sample
