"""
Data loading, cleaning, feature engineering and train/test split functions.

All functions are pure: they receive explicit parameters and return data,
with no I/O side-effects — print statements are the orchestrator's responsibility.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_flights(db_path: Path, columns_to_use: List[str]) -> pd.DataFrame:
    """
    Load the flights CSV reading only the required columns to save RAM.
    """
    return pd.read_csv(
        db_path / "flights.csv",
        usecols=columns_to_use,
        dtype={
            "AIRLINE": str,
            "ORIGIN_AIRPORT": str,
            "DESTINATION_AIRPORT": str,
        },
    )


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove cancelled and diverted flights (they have no meaningful arrival delay)
    and create the binary target column IS_DELAYED.

    IMPORTANT: ARRIVAL_DELAY is used ONLY for labelling the target.
    It never enters as a feature — doing so would cause direct data leakage.
    IATA standard: a flight is delayed if ARRIVAL_DELAY >= 15 minutes.
    """
    df = df[(df["CANCELLED"] == 0) & (df["DIVERTED"] == 0)].copy()
    df["IS_DELAYED"] = (df["ARRIVAL_DELAY"] >= 15).astype(int)
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
    return pd.Series(np.select(conditions, choices, default="night"), index=hour.index)


def feature_engineering(
    df: pd.DataFrame,
    features_numeric: List[str],
    features_categorical: List[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply feature engineering using only information available before departure.

    New columns created:
    - DEP_HOUR / DEP_MINUTE: SCHEDULED_DEPARTURE as a raw integer (e.g. 2355) is not
      linearly ordinal — splitting into hour and minute lets the model capture
      circadian patterns correctly.
    - IS_WEEKEND: binary flag for Saturday (6) and Sunday (7), which have a different
      traffic profile compared to weekdays.
    - PERIOD_OF_DAY: qualitative period (dawn/morning/afternoon/night) to capture
      non-linear time-of-day effects that a linear model can exploit via OHE.

    Returns (X, y) — X contains only pre-flight features, y is IS_DELAYED.
    """
    df = df.copy()

    # Decompose scheduled departure time into hour and minute
    df["DEP_HOUR"] = df["SCHEDULED_DEPARTURE"] // 100
    df["DEP_MINUTE"] = df["SCHEDULED_DEPARTURE"] % 100

    # DAY_OF_WEEK 6=Saturday, 7=Sunday (US flight dataset convention)
    df["IS_WEEKEND"] = df["DAY_OF_WEEK"].isin([6, 7]).astype(int)

    df["PERIOD_OF_DAY"] = _assign_period_of_day(df["DEP_HOUR"])

    X = df[features_numeric + features_categorical].copy()
    y = df["IS_DELAYED"]

    X[features_categorical] = X[features_categorical].astype(str)

    return X, y


def sample_and_split(
    X: pd.DataFrame,
    y: pd.Series,
    sample_size: int,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Perform stratified sampling followed by a stratified train/test split.

    Stratification in both steps ensures the proportion of delayed flights is
    consistent across the full dataset, sample, training set and test set —
    preventing sampling selection bias.

    Returns: (X_train, X_test, y_train, y_test, y_sample)
    y_sample is returned so the orchestrator can report the delay ratio in the sample.
    """
    X_sample, _, y_sample, _ = train_test_split(
        X, y, train_size=sample_size, stratify=y, random_state=random_state
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_sample,
        y_sample,
        test_size=test_size,
        stratify=y_sample,
        random_state=random_state,
    )

    return X_train, X_test, y_train, y_test, y_sample

def aggregate_by_airline(flights: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate flight metrics by airline.

    Computes mean values for key performance indicators:
    - Departure and arrival delays
    - Cancellation rate
    - Average distance and air time

    Returns:
        DataFrame indexed by AIRLINE with aggregated metrics and flight count.
    """
    airline_stats = flights.groupby("AIRLINE").agg({
        "DEPARTURE_DELAY": "mean",
        "ARRIVAL_DELAY": "mean",
        "CANCELLED": "mean",
        "DISTANCE": "mean",
        "AIR_TIME": "mean",
    }).fillna(0)
    
    airline_stats["flight_count"] = flights["AIRLINE"].value_counts()
    
    return airline_stats