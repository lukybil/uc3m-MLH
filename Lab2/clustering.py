import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score


def evaluate_clustering(X, n_clusters_range=range(2, 8)):
    """
    Args:
        X: Feature matrix
        n_clusters_range: Range of cluster numbers to evaluate

    Returns:
        tuple: (inertias, silhouettes, davies_bouldin_scores, optimal_n_clusters)
    """
    print("\nEvaluating different numbers of clusters...")
    inertias = []
    silhouettes = []
    davies_bouldin_scores = []

    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X, labels))
        davies_bouldin_scores.append(davies_bouldin_score(X, labels))

        print(
            f"  n_clusters={n_clusters}: "
            f"Silhouette={silhouettes[-1]:.3f}, "
            f"Davies-Bouldin={davies_bouldin_scores[-1]:.3f}"
        )

    optimal_idx = np.argmax(silhouettes)
    optimal_n_clusters = list(n_clusters_range)[optimal_idx]

    print(f"\n{'*'*80}")
    print(
        f"Optimal number of clusters: {optimal_n_clusters} (based on Silhouette score)"
    )
    print(f"{'*'*80}")

    return inertias, silhouettes, davies_bouldin_scores, optimal_n_clusters


def cluster_patients(features_df, n_clusters_range=range(2, 8)):
    """
    Returns:
        tuple: (cluster_labels, kmeans_model, scaler, X, X_scaled, feature_cols,
                inertias, silhouettes, davies_bouldin_scores, n_clusters_range)
    """
    print("\n" + "=" * 80)
    print("CLUSTERING PATIENTS")
    print("=" * 80)

    feature_cols = [col for col in features_df.columns if col != "ID"]
    X = features_df[feature_cols].values

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Features used: {feature_cols}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Evaluate different cluster numbers
    inertias, silhouettes, davies_bouldin_scores, optimal_n_clusters = (
        evaluate_clustering(X_scaled, n_clusters_range)
    )

    # Fit final model with optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42, n_init=20)
    cluster_labels = kmeans.fit_predict(X_scaled)

    return (
        cluster_labels,
        kmeans,
        scaler,
        X,
        X_scaled,
        feature_cols,
        inertias,
        silhouettes,
        davies_bouldin_scores,
        n_clusters_range,
    )
