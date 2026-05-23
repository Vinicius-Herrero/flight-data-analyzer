"""
PCA (Principal Component Analysis) dimensionality reduction visualization.

Demonstrates how to:
1. Apply PCA to reduce dimensions while preserving variance
2. Create a PCA-transformed DataFrame with cluster assignments
3. Analyze variance explained by each component
4. Visualize clusters in the reduced PCA space
"""

from pathlib import Path

import pandas as pd

from data_processing import load_flights, aggregate_by_airline
from statistical_analysis import (
    normalize_data,
    fit_final_kmeans,
    apply_pca,
    create_pca_dataframe,
    print_pca_report,
    plot_pca_clusters,
)


def main():
    """Main pipeline for PCA analysis."""
    
    # Load and aggregate data
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
    print("PCA ANALYSIS - DIMENSIONALITY REDUCTION")
    print("=" * 80)
    print(f"\nOriginal data shape: {airline_stats.shape}")
    print(f"Features: {list(airline_stats.columns)}")

    # Normalize data
    print("\nNormalizing data...")
    scaled_data, scaler = normalize_data(airline_stats)

    # Define optimal k (from previous analysis)
    optimal_k = 5

    # Fit final clustering model
    print(f"Fitting K-Means model with k={optimal_k}...")
    final_model, data_clustered = fit_final_kmeans(
        scaled_data=scaled_data,
        data_with_labels=airline_stats,
        optimal_k=optimal_k,
        random_state=42,
        n_init=10,
    )

    # Apply PCA
    print(f"\nApplying PCA (reducing to 2 components)...")
    pca, pca_data = apply_pca(scaled_data, n_components=2)
    print(f"✓ PCA applied")
    print(f"  Reduced data shape: {pca_data.shape}")

    # Create PCA DataFrame
    print(f"Creating PCA DataFrame...")
    pca_df = create_pca_dataframe(
        pca_data=pca_data,
        data_with_clusters=data_clustered,
        component_names=["PC1", "PC2"],
    )
    print(f"✓ PCA DataFrame created")
    print(f"\nFirst few rows of PCA DataFrame:")
    print(pca_df.head())

    # Print PCA report
    print_pca_report(pca)

    # Plot PCA visualization
    print(f"\nGenerating PCA cluster visualization...")
    plot_pca_clusters(
        pca_df=pca_df,
        pca=pca,
        pc1_col="PC1",
        pc2_col="PC2",
        figsize=(12, 7),
        alpha=0.8,
        point_size=200,
        text_offset=0.05,
        palette="viridis",
    )

    # Additional analysis: try 3 components for more variance
    print(f"\n" + "=" * 80)
    print("Additional Analysis: 3-Component PCA")
    print("=" * 80)
    pca_3d, pca_data_3d = apply_pca(scaled_data, n_components=3)
    print_pca_report(pca_3d)
    
    pca_3d_df = create_pca_dataframe(
        pca_data=pca_data_3d,
        data_with_clusters=data_clustered,
        component_names=["PC1", "PC2", "PC3"],
    )
    print(f"\nPCA DataFrame (3 components):")
    print(pca_3d_df.head())
    print(f"Variance captured by first 3 components: {pca_3d.explained_variance_ratio_.sum()*100:.2f}%")


if __name__ == "__main__":
    main()
