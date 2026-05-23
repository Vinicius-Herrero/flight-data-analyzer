"""
Exploratory Data Analysis (EDA) for flight data.

Comprehensive analysis pipeline demonstrating:
1. Airline statistics aggregation and distribution analysis
2. Normality testing using Shapiro-Wilk test
3. Outlier detection and visualization using boxplots
4. IQR-based outlier statistics across delay and distance metrics
"""

from pathlib import Path
import pandas as pd

from data_processing import load_flights, aggregate_by_airline
from statistical_analysis import (
    plot_distributions,
    shapiro_wilk_test,
    print_shapiro_wilk_results,
    plot_outlier_boxplots,
    calculate_iqr_outliers,
    print_iqr_outlier_results,
)


def analyze_airline_statistics():
    """
    Analyze airline statistics including distributions and normality tests.
    """
    print("\n" + "=" * 80)
    print("AIRLINE STATISTICS ANALYSIS")
    print("=" * 80)
    
    # Load data
    db_path = Path("database")
    flights = load_flights(
        db_path,
        columns_to_use=[
            "AIRLINE",
            "DEPARTURE_DELAY",
            "ARRIVAL_DELAY",
            "CANCELLED",
            "DISTANCE",
            "AIR_TIME",
        ],
    )

    # Aggregate by airline
    airline_stats = aggregate_by_airline(flights)

    print(f"\nAirlines analyzed: {len(airline_stats)}")
    print(f"Total flights: {airline_stats['flight_count'].sum():.0f}")
    print("\nAggregated Statistics:\n")
    print(airline_stats)

    # Visualize distributions
    features_to_plot = ["DEPARTURE_DELAY", "DISTANCE", "flight_count"]
    print(f"\n\nPlotting distributions for: {features_to_plot}")
    plot_distributions(airline_stats, features_to_plot)

    # Test for normality
    results = shapiro_wilk_test(airline_stats)
    print_shapiro_wilk_results(results)
    
    return airline_stats


def analyze_outliers(sample_size: int = 200000):
    """
    Analyze outliers in flight delay and distance metrics.
    
    Ajustado para trabalhar com uma amostra dos dados,
    evitando consumo excessivo de memória.
    """
    print("\n" + "=" * 80)
    print("OUTLIER DETECTION ANALYSIS")
    print("=" * 80)
    
    # Load full data
    db_path = Path("database")
    flights = load_flights(
        db_path,
        columns_to_use=[
            "DEPARTURE_DELAY",
            "ARRIVAL_DELAY",
            "DISTANCE",
        ],
    )

    # Reduzir para uma amostra
    if len(flights) > sample_size:
        flights = flights.sample(n=sample_size, random_state=42)
    print(f"\nUsing sample of {len(flights)} rows for outlier analysis")

    # Visualize outliers com boxplots
    print("\nGenerating boxplots for outlier visualization...")
    plot_outlier_boxplots(
        flights,
        delay_columns=["DEPARTURE_DELAY", "ARRIVAL_DELAY"],
        distance_column="DISTANCE",
    )

    # Calculate IQR-based outlier statistics
    print("\n" + "-" * 80)
    print("IQR-based Outlier Detection")
    print("-" * 80)

    for col in ["DEPARTURE_DELAY", "ARRIVAL_DELAY", "DISTANCE"]:
        outliers = calculate_iqr_outliers(
            flights[col],
            column_name=col,
            multiplier=1.5,
        )
        print_iqr_outlier_results(outliers)


def main():
    """Execute all exploratory data analysis pipelines."""
    print("=" * 80)
    print("FLIGHT DATA - EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 80)
    
    # Run airline statistics analysis
    airline_stats = analyze_airline_statistics()
    
    # Run outlier detection analysis (com amostra)
    analyze_outliers()
    
    print("\n" + "=" * 80)
    print("EDA COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
