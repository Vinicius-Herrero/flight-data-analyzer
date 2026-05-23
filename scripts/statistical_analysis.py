"""
Statistical analysis and visualization functions for flight data.

Provides functionality for normality testing (Shapiro-Wilk), distribution
visualization, outlier detection, and unsupervised clustering optimization.
"""

from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


def plot_distributions(
    data: pd.DataFrame,
    features: List[str],
    figsize: Tuple[int, int] = (18, 5),
) -> None:
    """
    Plot histograms with KDE for specified features.

    Parameters:
        data: DataFrame containing the features to plot
        features: List of column names to visualize
        figsize: Figure size (width, height)
    """
    plt.figure(figsize=figsize)

    for i, col in enumerate(features, start=1):
        plt.subplot(1, len(features), i)
        sns.histplot(data[col], kde=True, color="teal")
        plt.title(f"Distribuição: {col}")

    plt.tight_layout()
    plt.show()


def shapiro_wilk_test(data: pd.DataFrame) -> Dict[str, Tuple[float, str]]:
    """
    Perform Shapiro-Wilk test for normality on all numeric columns.

    The Shapiro-Wilk test assesses whether data follows a normal distribution.
    - p-value > 0.05: Data is approximately Gaussian (fail to reject null hypothesis)
    - p-value ≤ 0.05: Data is not Gaussian (reject null hypothesis)

    Parameters:
        data: DataFrame to test (numeric columns only)

    Returns:
        Dictionary mapping column names to (p_value, gaussian_status)
        where gaussian_status is 'Gaussiana' or 'Não-Gaussiana'
    """
    results = {}

    for col in data.columns:
        _, p_value = stats.shapiro(data[col])
        gaussian_status = "Gaussiana" if p_value > 0.05 else "Não-Gaussiana"
        results[col] = (p_value, gaussian_status)

    return results


def print_shapiro_wilk_results(results: Dict[str, Tuple[float, str]]) -> None:
    """
    Pretty-print Shapiro-Wilk test results.

    Parameters:
        results: Output from shapiro_wilk_test()
    """
    print("\n--- Teste de Shapiro-Wilk (p > 0.05 indica normalidade) ---")
    for col, (p_value, status) in results.items():
        print(f"{col:20}: p-value = {p_value:.4f} ({status})")


def plot_outlier_boxplots(
    data: pd.DataFrame,
    delay_columns: List[str],
    distance_column: str = "DISTANCE",
    figsize: Tuple[int, int] = (16, 6),
) -> None:
    """
    Plot boxplots to visualize outliers in delay and distance data.

    Boxplots show the distribution and identify potential outliers visually.
    Points beyond 1.5×IQR from Q1/Q3 are typically considered outliers.

    Parameters:
        data: DataFrame containing the data to visualize
        delay_columns: List of delay column names (e.g., ["DEPARTURE_DELAY", "ARRIVAL_DELAY"])
        distance_column: Name of the distance column
        figsize: Figure size (width, height)
    """
    plt.figure(figsize=figsize)

    # Boxplot for delays
    plt.subplot(1, 2, 1)
    sns.boxplot(data=data[delay_columns], palette="Set2")
    plt.title("Identificação de Outliers: Atrasos (Partida e Chegada)")
    plt.ylabel("Minutos")

    # Boxplot for distance
    plt.subplot(1, 2, 2)
    sns.boxplot(x=data[distance_column], color="skyblue")
    plt.title("Identificação de Outliers: Distância dos Voos")
    plt.xlabel("Milhas")

    plt.tight_layout()
    plt.show()


def calculate_iqr_outliers(
    data: pd.Series,
    column_name: str,
    multiplier: float = 1.5,
) -> Dict[str, float | int]:
    """
    Calculate outlier statistics using the Interquartile Range (IQR) method.

    The IQR method identifies outliers as points beyond:
    - Upper limit: Q3 + multiplier × IQR
    - Lower limit: Q1 - multiplier × IQR

    Standard multiplier is 1.5 (Tukey's fences). Multiplier of 3.0 identifies
    extreme outliers only.

    Parameters:
        data: Series containing the data to analyze
        column_name: Name of the column (for display purposes)
        multiplier: IQR multiplier (default 1.5)

    Returns:
        Dictionary with Q1, Q3, IQR, lower/upper limits, and outlier counts
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    lower_limit = Q1 - multiplier * IQR
    upper_limit = Q3 + multiplier * IQR

    outliers_lower = (data < lower_limit).sum()
    outliers_upper = (data > upper_limit).sum()
    outliers_total = outliers_lower + outliers_upper

    return {
        "column": column_name,
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "lower_limit": lower_limit,
        "upper_limit": upper_limit,
        "outliers_lower": outliers_lower,
        "outliers_upper": outliers_upper,
        "outliers_total": outliers_total,
        "outliers_percentage": (outliers_total / len(data)) * 100,
    }


def print_iqr_outlier_results(results: Dict[str, float | int]) -> None:
    """
    Pretty-print IQR outlier analysis results.

    Parameters:
        results: Output from calculate_iqr_outliers()
    """
    print(f"\nEstatísticas de Outliers ({results['column']}) - Método IQR:")
    print(f"  Q1 (25º percentil): {results['Q1']:.2f}")
    print(f"  Q3 (75º percentil): {results['Q3']:.2f}")
    print(f"  IQR (Q3 - Q1): {results['IQR']:.2f}")
    print(f"  Limite Inferior: {results['lower_limit']:.2f}")
    print(f"  Limite Superior: {results['upper_limit']:.2f}")
    print(f"  Outliers abaixo do limite: {results['outliers_lower']}")
    print(f"  Outliers acima do limite: {results['outliers_upper']}")
    print(f"  Total de outliers: {results['outliers_total']} ({results['outliers_percentage']:.2f}%)")


def normalize_data(data: pd.DataFrame | np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    """
    Normalize data using StandardScaler (zero mean, unit variance).

    Standardization is crucial for distance-based algorithms like K-Means,
    where features with larger scales would dominate the distance calculations.

    Parameters:
        data: DataFrame or array to normalize

    Returns:
        Tuple of (normalized_array, fitted_scaler)
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler


def optimize_kmeans(
    data: np.ndarray,
    k_range: range = range(2, 10),
    random_state: int = 42,
    n_init: int = 10,
) -> Dict[str, List[float]]:
    """
    Test multiple K values and compute optimization metrics for K-Means.

    Two key metrics are computed:
    - WCSS (Within-Cluster Sum of Squares / Inertia): minimized for "Elbow Method"
    - Silhouette Score: maximized to find optimal k (ranges -1 to 1)

    Parameters:
        data: Normalized data array
        k_range: Range of k values to test (default 2-9)
        random_state: Random seed for reproducibility
        n_init: Number of initializations (default 10)

    Returns:
        Dictionary with 'wcss' and 'silhouette_scores' lists
    """
    wcss = []
    silhouette_vals = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = km.fit_predict(data)
        wcss.append(km.inertia_)
        silhouette_vals.append(silhouette_score(data, labels))

    return {
        "k_values": list(k_range),
        "wcss": wcss,
        "silhouette_scores": silhouette_vals,
    }


def plot_elbow_silhouette(
    optimization_results: Dict[str, List[float]],
    figsize: Tuple[int, int] = (16, 5),
) -> None:
    """
    Plot Elbow Method and Silhouette Score for K-Means optimization.

    Elbow plot (WCSS) helps identify where the "bend" occurs (diminishing returns).
    Silhouette plot shows model quality — higher scores indicate better-separated clusters.

    Parameters:
        optimization_results: Output from optimize_kmeans()
        figsize: Figure size (width, height)
    """
    k_values = optimization_results["k_values"]
    wcss = optimization_results["wcss"]
    silhouette_vals = optimization_results["silhouette_scores"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Elbow Method
    ax1.plot(k_values, wcss, "go-")
    ax1.set_title("Método do Cotovelo (Busca por Inflexão)")
    ax1.set_xlabel("Nº de Clusters")
    ax1.set_ylabel("Inércia (WCSS)")
    ax1.grid(alpha=0.3)

    # Silhouette Score
    ax2.plot(k_values, silhouette_vals, "bo-")
    ax2.set_title("Silhouette Score (Busca pelo Máximo)")
    ax2.set_xlabel("Nº de Clusters")
    ax2.set_ylabel("Score")
    ax2.grid(alpha=0.3)

    # Highlight maximum silhouette score
    max_idx = np.argmax(silhouette_vals)
    optimal_k = k_values[max_idx]
    ax2.scatter(optimal_k, silhouette_vals[max_idx], color="red", s=100, zorder=5)
    ax2.annotate(
        f"Ótimo: k={optimal_k}",
        xy=(optimal_k, silhouette_vals[max_idx]),
        xytext=(10, 10),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
    )

    plt.tight_layout()
    plt.show()


def fit_final_kmeans(
    scaled_data: np.ndarray,
    data_with_labels: pd.DataFrame,
    optimal_k: int,
    random_state: int = 42,
    n_init: int = 10,
) -> tuple[KMeans, pd.DataFrame]:
    """
    Train final K-Means model with optimal k and add cluster labels to data.

    Parameters:
        scaled_data: Normalized data array
        data_with_labels: Original DataFrame with airline names/indices
        optimal_k: Optimal number of clusters to use
        random_state: Random seed for reproducibility
        n_init: Number of initializations

    Returns:
        Tuple of (fitted_model, data_with_cluster_assignments)
    """
    model = KMeans(n_clusters=optimal_k, random_state=random_state, n_init=n_init)
    cluster_labels = model.fit_predict(scaled_data)
    
    data_clustered = data_with_labels.copy()
    data_clustered["cluster"] = cluster_labels
    
    return model, data_clustered


def plot_clusters_with_centroids(
    data_clustered: pd.DataFrame,
    model: KMeans,
    scaler: StandardScaler,
    x_col: str = "DISTANCE",
    y_col: str = "DEPARTURE_DELAY",
    size_col: str = "flight_count",
    figsize: Tuple[int, int] = (14, 8),
    alpha_points: float = 0.6,
    marker_size: int = 300,
    text_offset: int = 15,
    grid_linestyle: str = "--",
) -> None:
    """
    Plot scatter of airlines with cluster colors and centroids marked.

    Shows airline clusters in a 2D space with:
    - Points colored by cluster assignment
    - Point size representing flight count
    - Centroids marked with red 'X' symbols
    - Airline names annotated for easy identification

    IMPORTANT: The order of columns in the DataFrame (excluding 'cluster') must
    match the order used when fitting the StandardScaler, otherwise centroid
    indices will be incorrect.

    Parameters:
        data_clustered: DataFrame with cluster assignments
        model: Fitted KMeans model
        scaler: Fitted StandardScaler (for inverse transform of centroids)
        x_col: Column name for x-axis (must be numeric and in data)
        y_col: Column name for y-axis (must be numeric and in data)
        size_col: Column name for point size (must be numeric and in data)
        figsize: Figure size (width, height), default (14, 8)
        alpha_points: Transparency of scatter points, default 0.6
        marker_size: Size of centroid 'X' markers, default 300
        text_offset: Horizontal offset for airline name labels, default 15
        grid_linestyle: Grid line style ('--', '-', ':', etc.), default '--'

    Raises:
        ValueError: If x_col, y_col, or size_col are not in the DataFrame
    """
    # Validate columns exist
    required_cols = {x_col, y_col, size_col}
    data_cols = set(data_clustered.columns) - {"cluster"}
    missing_cols = required_cols - data_cols
    if missing_cols:
        raise ValueError(f"Columns not found in data: {missing_cols}")
    
    # Inverse transform centroids to original scale
    centroids_scaled = model.cluster_centers_
    centroids_orig = scaler.inverse_transform(centroids_scaled)
    
    # Get column indices for plotting (exclude 'cluster' column)
    feature_names = [col for col in data_clustered.columns if col != "cluster"]
    x_idx = feature_names.index(x_col)
    y_idx = feature_names.index(y_col)
    
    plt.figure(figsize=figsize)
    
    # Plot airlines as scatter points
    sns.scatterplot(
        data=data_clustered,
        x=x_col,
        y=y_col,
        hue="cluster",
        size=size_col,
        sizes=(100, 1000),
        palette="viridis",
        alpha=alpha_points,
        legend="brief",
    )
    
    # Plot centroids as large red X markers
    plt.scatter(
        centroids_orig[:, x_idx],
        centroids_orig[:, y_idx],
        marker="X",
        s=marker_size,
        linewidths=2,
        color="red",
        edgecolors="black",
        label="Centroides",
        zorder=5,
    )
    
    # Annotate each centroid
    for i, (x, y) in enumerate(zip(centroids_orig[:, x_idx], centroids_orig[:, y_idx])):
        plt.annotate(
            f"C{i}",
            xy=(x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
            color="darkred",
        )
    
    # Annotate airline names
    for i in range(len(data_clustered)):
        plt.text(
            data_clustered[x_col].iloc[i] + text_offset,
            data_clustered[y_col].iloc[i],
            data_clustered.index[i],
            fontsize=10,
            weight="bold",
            alpha=0.8,
        )
    
    plt.title(
        f"Segmentação de Mercado: Companhias vs. Centroides dos Clusters",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel(f"{x_col} (Milhas)")
    plt.ylabel(f"{y_col} (Minutos)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, linestyle=grid_linestyle, alpha=0.5)
    plt.tight_layout()
    plt.show()


def get_cluster_profiles(
    data_clustered: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute mean metrics for each cluster.

    Summarizes the operational profile of each cluster by computing the mean
    value of all numeric features (excluding 'cluster') grouped by cluster.

    Parameters:
        data_clustered: DataFrame with cluster assignments

    Returns:
        DataFrame indexed by cluster with mean values for each metric
    """
    profile_summary = data_clustered.groupby("cluster").mean(numeric_only=True)
    return profile_summary


def print_cluster_profiles(profile_summary: pd.DataFrame, optimal_k: int) -> None:
    """
    Pretty-print cluster profile summary.

    Parameters:
        profile_summary: Output from get_cluster_profiles()
        optimal_k: Number of clusters (for display context)
    """
    print(f"\n--- Métricas Médias por Cluster (K={optimal_k}) ---")
    print(profile_summary.round(2))


def plot_cluster_heatmap(
    profile_summary: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 6),
) -> None:
    """
    Plot heatmap of cluster profiles for visual interpretation.

    Shows each feature (column) across all clusters (rows) to quickly identify
    which features distinguish each cluster.

    Parameters:
        profile_summary: Output from get_cluster_profiles()
        figsize: Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    sns.heatmap(
        profile_summary.T,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        cbar_kws={"label": "Mean Value"},
    )
    plt.title("Identidade Operacional por Cluster (Mapa de Calor de Médias)", fontsize=14)
    plt.xlabel("Cluster")
    plt.ylabel("Métrica")
    plt.tight_layout()
    plt.show()


def plot_kmeans_iterations(
    scaled_data: np.ndarray,
    data_original: pd.DataFrame,
    scaler: StandardScaler,
    n_clusters: int,
    x_col: str = "DISTANCE",
    y_col: str = "DEPARTURE_DELAY",
    max_iterations: int = 6,
    figsize: Tuple[int, int] = (18, 10),
    alpha: float = 0.6,
    marker_size: int = 150,
    palette: str = "viridis",
) -> List[float]:
    """
    Visualize K-Means convergence by showing iterations 1 through max_iterations.

    Shows the evolution of cluster assignments and centroid positions as the
    algorithm iterates. Useful for understanding how K-Means converges.

    IMPORTANT: Uses hardcoded column indices from the scaler, so the order
    of features must be consistent between scaling and visualization.

    Parameters:
        scaled_data: Normalized data array used for fitting
        data_original: Original (unscaled) DataFrame with labels
        scaler: Fitted StandardScaler for inverse transforming centroids
        n_clusters: Number of clusters to use (same for all iterations)
        x_col: Column name for x-axis visualization
        y_col: Column name for y-axis visualization
        max_iterations: Maximum number of iterations to display (1-9 recommended)
        figsize: Figure size (width, height), default (18, 10)
        alpha: Transparency of points, default 0.6
        marker_size: Size of centroid markers, default 150
        palette: Color palette for clusters, default 'viridis'

    Returns:
        List of inertia values for each iteration (useful for tracking convergence)

    Raises:
        ValueError: If x_col or y_col are not in data_original
    """
    # Validate columns
    if x_col not in data_original.columns or y_col not in data_original.columns:
        raise ValueError(f"Columns {x_col} and/or {y_col} not found in data")

    # Calculate grid layout
    n_rows = (max_iterations + 2) // 3
    n_cols = min(3, max_iterations)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if max_iterations == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    inertia_per_iteration = []
    
    # Get column indices for plotting
    feature_names = [col for col in data_original.columns if col != "cluster"]
    x_idx = feature_names.index(x_col)
    y_idx = feature_names.index(y_col)

    # Train models with increasing iterations
    for i in range(min(max_iterations, len(axes))):
        # Train K-Means limited to i+1 iterations
        km_step = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=1,
            max_iter=i + 1,
            random_state=42,
        )
        km_step.fit(scaled_data)

        labels = km_step.labels_
        inertia_per_iteration.append(km_step.inertia_)
        
        # Inverse transform centroids to original scale
        centers_orig = scaler.inverse_transform(km_step.cluster_centers_)

        # Plot on subplot
        ax = axes[i]
        sns.scatterplot(
            x=data_original[x_col],
            y=data_original[y_col],
            hue=labels,
            palette=palette,
            ax=ax,
            legend=False,
            alpha=alpha,
        )
        
        # Plot centroids
        ax.scatter(
            centers_orig[:, x_idx],
            centers_orig[:, y_idx],
            c="red",
            marker="X",
            s=marker_size,
            edgecolors="black",
            linewidths=1.5,
            zorder=5,
        )
        
        ax.set_title(f"Iteração {i + 1}", fontsize=12, fontweight="bold")
        ax.set_xlabel(f"{x_col} (Milhas)")
        ax.set_ylabel(f"{y_col} (Minutos)")
        ax.grid(True, alpha=0.2)

    # Hide unused subplots
    for j in range(len(axes), max_iterations):
        if j < len(axes):
            axes[j].set_visible(False)

    plt.tight_layout()
    plt.suptitle(
        "Evolução da Posição dos Centroides e Agrupamento (K-Means Iterations)",
        fontsize=16,
        fontweight="bold",
        y=1.00,
    )
    plt.show()

    return inertia_per_iteration


def print_convergence_report(
    model: KMeans,
    scaled_data: np.ndarray,
    max_iterations: int = 10,
) -> None:
    """
    Print convergence analysis for a fitted K-Means model.

    Shows whether the model converged and how many iterations it took,
    along with the final inertia (sum of squared distances to nearest centroid).

    Parameters:
        model: Fitted KMeans model
        scaled_data: Data that was fit to the model (for inertia recalculation)
        max_iterations: Maximum iterations the model was allowed (for context)
    """
    converged = model.n_iter_ < max_iterations
    status = "✓ CONVERGED" if converged else "✗ DID NOT CONVERGE"
    
    print(f"\n--- Relatório de Convergência do K-Means ---")
    print(f"Status: {status}")
    print(f"Iterações necessárias: {model.n_iter_}")
    print(f"Máximo de iterações: {max_iterations}")
    print(f"Inércia final (Soma dos quadrados intra-cluster): {model.inertia_:.2f}")
    print(f"Número de clusters: {model.n_clusters}")


def get_inertia_by_iteration(
    scaled_data: np.ndarray,
    n_clusters: int,
    max_iterations: int = 10,
    random_state: int = 42,
) -> List[float]:
    """
    Track inertia value for each iteration of K-Means.

    Shows how the model's inertia decreases with each iteration,
    which helps understand convergence speed.

    Parameters:
        scaled_data: Normalized data array
        n_clusters: Number of clusters to use
        max_iterations: Maximum iterations to track
        random_state: Random seed for reproducibility

    Returns:
        List of inertia values, one per iteration
    """
    inertia_values = []
    
    for i in range(1, max_iterations + 1):
        km = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=1,
            max_iter=i,
            random_state=random_state,
        )
        km.fit(scaled_data)
        inertia_values.append(km.inertia_)
    
    return inertia_values


def apply_pca(
    scaled_data: np.ndarray,
    n_components: int = 2,
) -> tuple[PCA, np.ndarray]:
    """
    Apply Principal Component Analysis (PCA) to dimensionality reduction.

    PCA transforms data into a new coordinate system where the first
    components capture the most variance in the data. Useful for visualization
    and understanding dominant patterns.

    Parameters:
        scaled_data: Normalized data array (must be pre-scaled)
        n_components: Number of principal components to keep (default 2)

    Returns:
        Tuple of (fitted_pca_model, transformed_data)
        transformed_data has shape (n_samples, n_components)
    """
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(scaled_data)
    return pca, pca_data


def create_pca_dataframe(
    pca_data: np.ndarray,
    data_with_clusters: pd.DataFrame,
    component_names: List[str] | None = None,
) -> pd.DataFrame:
    """
    Create a DataFrame with PCA components and cluster assignments.

    Parameters:
        pca_data: Transformed data from PCA (shape: n_samples x n_components)
        data_with_clusters: DataFrame with cluster assignments (index must match)
        component_names: Optional custom names for components (e.g., ['PC1', 'PC2'])
                        If None, auto-generates 'PC1', 'PC2', etc.

    Returns:
        DataFrame with PCA components and cluster column
    """
    n_components = pca_data.shape[1]
    
    if component_names is None:
        component_names = [f"PC{i+1}" for i in range(n_components)]
    
    pca_df = pd.DataFrame(
        data=pca_data,
        columns=component_names,
        index=data_with_clusters.index,
    )
    pca_df["cluster"] = data_with_clusters["cluster"]
    
    return pca_df


def print_pca_report(pca: PCA) -> None:
    """
    Print variance analysis for PCA components.

    Shows how much variance each component explains and the cumulative
    explained variance. Helps assess data compression quality.

    Parameters:
        pca: Fitted PCA model
    """
    total_variance = pca.explained_variance_ratio_.sum() * 100
    
    print(f"\n--- Relatório PCA ---")
    for i, variance in enumerate(pca.explained_variance_ratio_):
        print(f"Variância explicada pelo PC{i+1}: {variance*100:.2f}%")
    print(f"Variância total retida: {total_variance:.2f}%")


def plot_pca_clusters(
    pca_df: pd.DataFrame,
    pca: PCA,
    pc1_col: str = "PC1",
    pc2_col: str = "PC2",
    figsize: Tuple[int, int] = (12, 7),
    alpha: float = 0.8,
    point_size: int = 200,
    text_offset: float = 0.05,
    palette: str = "viridis",
) -> None:
    """
    Plot clusters in PCA-reduced 2D space with airline names annotated.

    Visualizes clusters in the principal component space, making it easy to
    see which airlines are similar and how clusters are separated.

    Parameters:
        pca_df: DataFrame from create_pca_dataframe()
        pca: Fitted PCA model (used for variance labels)
        pc1_col: Column name for first principal component (default 'PC1')
        pc2_col: Column name for second principal component (default 'PC2')
        figsize: Figure size (width, height), default (12, 7)
        alpha: Transparency of points, default 0.8
        point_size: Size of scatter points, default 200
        text_offset: Offset for airline name labels, default 0.05
        palette: Color palette for clusters, default 'viridis'

    Raises:
        ValueError: If pc1_col, pc2_col, or 'cluster' are not in pca_df
    """
    required_cols = {pc1_col, pc2_col, "cluster"}
    missing_cols = required_cols - set(pca_df.columns)
    if missing_cols:
        raise ValueError(f"Columns not found: {missing_cols}")

    plt.figure(figsize=figsize)
    
    # Plot clusters as scatter
    sns.scatterplot(
        x=pc1_col,
        y=pc2_col,
        hue="cluster",
        palette=palette,
        data=pca_df,
        s=point_size,
        alpha=alpha,
        edgecolor="black",
    )

    # Annotate airline names (using index)
    for i in range(len(pca_df)):
        plt.text(
            pca_df[pc1_col].iloc[i] + text_offset,
            pca_df[pc2_col].iloc[i] + text_offset,
            pca_df.index[i],
            fontsize=10,
            weight="bold",
        )

    # Create axis labels with variance explained
    var_pc1 = pca.explained_variance_ratio_[0] * 100
    var_pc2 = pca.explained_variance_ratio_[1] * 100
    
    plt.title(
        "Clusters de Companhias Aéreas no Espaço PCA (Reduzido)",
        fontsize=15,
        fontweight="bold",
    )
    plt.xlabel(f"{pc1_col} ({var_pc1:.1f}% da variância)")
    plt.ylabel(f"{pc2_col} ({var_pc2:.1f}% da variância)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()
