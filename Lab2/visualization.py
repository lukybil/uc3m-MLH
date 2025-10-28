import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

plt.ioff()


def plot_clustering_metrics(
    inertias, silhouettes, davies_bouldin_scores, n_clusters_range, output_dir="results"
):
    """
    Plot clustering evaluation metrics.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(n_clusters_range, inertias, "bo-")
    axes[0].set_xlabel("Number of Clusters")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow Method")
    axes[0].grid(True)

    axes[1].plot(n_clusters_range, silhouettes, "go-")
    axes[1].set_xlabel("Number of Clusters")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Score (higher is better)")
    axes[1].grid(True)

    axes[2].plot(n_clusters_range, davies_bouldin_scores, "ro-")
    axes[2].set_xlabel("Number of Clusters")
    axes[2].set_ylabel("Davies-Bouldin Index")
    axes[2].set_title("Davies-Bouldin Index (lower is better)")
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "clustering_metrics.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def visualize_clusters(
    X_scaled, cluster_labels, kmeans_model, feature_cols, output_dir="results"
):
    """
    Visualize clusters using PCA.
    """
    print("\nVisualizing clusters with PCA...")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=cluster_labels,
        cmap="viridis",
        alpha=0.6,
        s=50,
    )

    centers_pca = pca.transform(kmeans_model.cluster_centers_)
    ax.scatter(
        centers_pca[:, 0],
        centers_pca[:, 1],
        c="red",
        marker="X",
        s=200,
        edgecolors="black",
        linewidths=2,
        label="Centroids",
    )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    ax.set_title("Patient Clusters (PCA Visualization)")
    ax.legend()
    plt.colorbar(scatter, label="Cluster")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "cluster_visualization.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(
        f"Total variance explained by PC1 and PC2: {sum(pca.explained_variance_ratio_)*100:.1f}%"
    )


def plot_event_distributions(cluster_sequences, output_dir="results"):
    """
    Plot event distributions by cluster.
    """
    print("\nPlotting event distributions by cluster...")

    n_clusters = len(cluster_sequences)
    fig, axes = plt.subplots(n_clusters, 2, figsize=(14, 4 * n_clusters))

    if n_clusters == 1:
        axes = axes.reshape(1, -1)

    for cluster_id, sequences in cluster_sequences.items():
        if len(sequences) == 0:
            continue

        all_events = []
        all_inter_arrival = []
        for seq in sequences:
            events = seq["events"]
            all_events.extend(events)
            if len(events) > 1:
                inter_arrival = np.diff(events)
                all_inter_arrival.extend(inter_arrival)

        axes[cluster_id, 0].hist(
            all_events, bins=30, alpha=0.7, color="steelblue", edgecolor="black"
        )
        axes[cluster_id, 0].set_xlabel("Time (months)")
        axes[cluster_id, 0].set_ylabel("Frequency")
        axes[cluster_id, 0].set_title(f"Cluster {cluster_id}: Event Times Distribution")
        axes[cluster_id, 0].grid(True, alpha=0.3)

        if all_inter_arrival:
            axes[cluster_id, 1].hist(
                all_inter_arrival,
                bins=30,
                alpha=0.7,
                color="coral",
                edgecolor="black",
            )
            axes[cluster_id, 1].set_xlabel("Inter-arrival Time (months)")
            axes[cluster_id, 1].set_ylabel("Frequency")
            axes[cluster_id, 1].set_title(f"Cluster {cluster_id}: Inter-arrival Times")
            axes[cluster_id, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "event_distributions_by_cluster.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_hawkes_parameters(params_df, output_dir="results"):
    """
    Plot Hawkes process parameters comparison.
    """
    colors = sns.color_palette("Set2", n_colors=4)

    if "baseline" in params_df.columns:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        params_df["baseline"].plot(kind="bar", ax=axes[0, 0], color=colors[0])
        axes[0, 0].set_title("Baseline Intensity (μ)")
        axes[0, 0].set_ylabel("μ")
        axes[0, 0].grid(True, alpha=0.3)

        params_df["adjacency"].plot(kind="bar", ax=axes[0, 1], color=colors[1])
        axes[0, 1].set_title("Adjacency (α)")
        axes[0, 1].set_ylabel("α")
        axes[0, 1].grid(True, alpha=0.3)

        params_df["decay"].plot(kind="bar", ax=axes[1, 0], color=colors[2])
        axes[1, 0].set_title("Decay (β)")
        axes[1, 0].set_ylabel("β")
        axes[1, 0].grid(True, alpha=0.3)

        params_df["branching_ratio"].plot(kind="bar", ax=axes[1, 1], color=colors[3])
        axes[1, 1].set_title("Branching Ratio (α/β)")
        axes[1, 1].set_ylabel("α/β")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=1, color="red", linestyle="--", label="Critical value")
        axes[1, 1].legend()
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        params_df["event_rate"].plot(kind="bar", ax=axes[0, 0], color=colors[0])
        axes[0, 0].set_title("Event Rate (events/month)")
        axes[0, 0].set_ylabel("Rate")
        axes[0, 0].grid(True, alpha=0.3)

        params_df["mean_inter_arrival"].plot(kind="bar", ax=axes[0, 1], color=colors[1])
        axes[0, 1].set_title("Mean Inter-arrival Time")
        axes[0, 1].set_ylabel("Months")
        axes[0, 1].grid(True, alpha=0.3)

        params_df["std_inter_arrival"].plot(kind="bar", ax=axes[1, 0], color=colors[2])
        axes[1, 0].set_title("Std Inter-arrival Time")
        axes[1, 0].set_ylabel("Months")
        axes[1, 0].grid(True, alpha=0.3)

        params_df["total_events"].plot(kind="bar", ax=axes[1, 1], color=colors[3])
        axes[1, 1].set_title("Total Events")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "hawkes_parameters_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
