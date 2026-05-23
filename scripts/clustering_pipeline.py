"""
Complete K-Means clustering pipeline.

Comprehensive analysis demonstrating:
1. K-Means optimization using Elbow Method and Silhouette Score
2. Final clustering visualization with centroids and profiles
3. Convergence analysis through iteration tracking
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from data_processing import load_flights, aggregate_by_airline
from statistical_analysis import (
    normalize_data,
    optimize_kmeans,
    plot_elbow_silhouette,
    fit_final_kmeans,
    plot_clusters_with_centroids,
    get_cluster_profiles,
    print_cluster_profiles,
    plot_cluster_heatmap,
    plot_kmeans_iterations,
    print_convergence_report,
    get_inertia_by_iteration,
)


def load_and_prepare_data():
    """
    Load and aggregate flight data.
    
    Returns:
        Tuple of (flights_df, airline_stats_df, scaled_data, scaler)
    """
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

    airline_stats = aggregate_by_airline(flights)
    
    print("=" * 80)
    print("DATA PREPARATION")
    print("=" * 80)
    print(f"\nData shape: {airline_stats.shape}")
    print(f"Number of airlines: {len(airline_stats)}")
    print(f"Number of features: {airline_stats.shape[1]}")
    
    # Normalize data
    print("\n" + "-" * 80)
    print("Normalizing data (StandardScaler)...")
    scaled_data, scaler = normalize_data(airline_stats)
    print(f"✓ Data normalized")
    print(f"  Scaled data shape: {scaled_data.shape}")
    
    return flights, airline_stats, scaled_data, scaler


def step1_optimize_kmeans(scaled_data: pd.DataFrame):
    """
    Step 1: Find optimal number of clusters using Elbow Method and Silhouette Score.
    
    Parameters:
        scaled_data: Normalized data array
        
    Returns:
        Tuple of (optimization_results, optimal_k)
    """
    print("\n" + "=" * 80)
    print("STEP 1: K-MEANS OPTIMIZATION")
    print("=" * 80)
    print("\nTesting k values from 2 to 9...")
    
    optimization_results = optimize_kmeans(
        scaled_data,
        k_range=range(2, 10),
        random_state=42,
        n_init=10,
    )

    # Print results table
    print("\nOptimization Results:")
    print(f"{'K':>3} | {'WCSS':>10} | {'Silhouette':>12}")
    print("-" * 30)
    for k, wcss, silhouette in zip(
        optimization_results["k_values"],
        optimization_results["wcss"],
        optimization_results["silhouette_scores"],
    ):
        print(f"{k:>3} | {wcss:>10.2f} | {silhouette:>12.4f}")

    # Visualize results
    print("\nGenerating optimization plots...")
    plot_elbow_silhouette(optimization_results)

    # Find optimal k
    optimal_k = optimization_results["k_values"][
        optimization_results["silhouette_scores"].index(
            max(optimization_results["silhouette_scores"])
        )
    ]
    print(f"\n✓ Optimal K found: {optimal_k}")
    print(f"  Silhouette Score: {max(optimization_results['silhouette_scores']):.4f}")
    
    return optimization_results, optimal_k


def step2_visualize_clusters(
    scaled_data: pd.DataFrame,
    airline_stats: pd.DataFrame,
    scaler,
    optimal_k: int,
):
    """
    Step 2: Fit final model and visualize clusters with centroids and profiles.
    
    Parameters:
        scaled_data: Normalized data array
        airline_stats: Original aggregated data
        scaler: Fitted StandardScaler
        optimal_k: Number of clusters to use
        
    Returns:
        Tuple of (final_model, data_clustered, profile_summary)
    """
    print("\n" + "=" * 80)
    print("STEP 2: CLUSTERING VISUALIZATION")
    print("=" * 80)

    # Fit final model
    print(f"\nFitting final K-Means model with k={optimal_k}...")
    final_model, data_clustered = fit_final_kmeans(
        scaled_data,
        airline_stats,
        optimal_k=optimal_k,
        random_state=42,
        n_init=10,
    )
    print(f"✓ Model fitted and clusters assigned")

    # Print cluster summary
    print(f"\nCluster Distribution:")
    print(data_clustered["cluster"].value_counts().sort_index())

    # Plot final visualization
    print(f"\nGenerating cluster scatter with centroids...")
    plot_clusters_with_centroids(
        data_clustered,
        final_model,
        scaler,
        x_col="DISTANCE",
        y_col="DEPARTURE_DELAY",
        size_col="flight_count",
        figsize=(14, 8),
        alpha_points=0.6,
        marker_size=300,
        text_offset=15,
        grid_linestyle="--",
    )

    # Print cluster profiles
    print(f"\nComputing cluster profiles...")
    profile_summary = get_cluster_profiles(data_clustered)
    print_cluster_profiles(profile_summary, optimal_k)

    # Plot cluster heatmap
    print(f"\nGenerating cluster profile heatmap...")
    plot_cluster_heatmap(profile_summary)
    
    return final_model, data_clustered, profile_summary


def step3_analyze_convergence(
    scaled_data: pd.DataFrame,
    airline_stats: pd.DataFrame,
    scaler,
    final_model,
    optimal_k: int,
):
    """
    Step 3: Analyze convergence through iteration tracking and inertia curves.
    
    Parameters:
        scaled_data: Normalized data array
        airline_stats: Original aggregated data
        scaler: Fitted StandardScaler
        final_model: Fitted final KMeans model
        optimal_k: Number of clusters used
    """
    print("\n" + "=" * 80)
    print("STEP 3: CONVERGENCE ANALYSIS")
    print("=" * 80)

    # 1. Visualize iterations
    print(f"\nVisualizing K-Means iterations (max 6)...")
    inertia_per_iter = plot_kmeans_iterations(
        scaled_data=scaled_data,
        data_original=airline_stats,
        scaler=scaler,
        n_clusters=optimal_k,
        x_col="DISTANCE",
        y_col="DEPARTURE_DELAY",
        max_iterations=6,
        figsize=(18, 10),
        alpha=0.6,
        marker_size=150,
        palette="viridis",
    )
    print(f"✓ Iteration visualization generated")
    print(f"  Inertia per iteration: {[f'{x:.2f}' for x in inertia_per_iter]}")

    # 2. Track inertia across more iterations
    print(f"\nTracking inertia across iterations (up to 20)...")
    all_inertias = get_inertia_by_iteration(
        scaled_data=scaled_data,
        n_clusters=optimal_k,
        max_iterations=20,
        random_state=42,
    )

    # Plot inertia convergence curve
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(all_inertias) + 1), all_inertias, "bo-", linewidth=2, markersize=8)
    plt.title(
        f"Inércia por Iteração (K-Means com k={optimal_k})",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Iteração")
    plt.ylabel("Inércia (WCSS)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 3. Print convergence report
    print_convergence_report(
        model=final_model,
        scaled_data=scaled_data,
        max_iterations=300,
    )


def main():
    """Execute complete clustering pipeline."""
    print("\n" + "=" * 80)
    print("FLIGHT DATA - K-MEANS CLUSTERING PIPELINE")
    print("=" * 80)
    
    # Load and prepare data
    flights, airline_stats, scaled_data, scaler = load_and_prepare_data()
    
    # Step 1: Optimize K
    optimization_results, optimal_k = step1_optimize_kmeans(scaled_data)
    
    # Step 2: Visualize clusters
    final_model, data_clustered, profile_summary = step2_visualize_clusters(
        scaled_data, airline_stats, scaler, optimal_k
    )
    
    # Step 3: Analyze convergence
    step3_analyze_convergence(
        scaled_data, airline_stats, scaler, final_model, optimal_k
    )
    
    print("\n" + "=" * 80)
    print("CLUSTERING PIPELINE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()