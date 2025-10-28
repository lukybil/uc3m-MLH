import numpy as np


def analyze_clusters(X, feature_cols, cluster_labels):
    print("\n" + "=" * 80)
    print("CLUSTER ANALYSIS")
    print("=" * 80)

    n_clusters = len(np.unique(cluster_labels))

    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_size = np.sum(cluster_mask)
        cluster_data = X[cluster_mask]

        print(f"\n{'='*80}")
        print(
            f"CLUSTER {cluster_id} (n={cluster_size}, {cluster_size/len(X)*100:.1f}%)"
        )
        print(f"{'='*80}")

        for i, feature in enumerate(feature_cols):
            mean_val = cluster_data[:, i].mean()
            std_val = cluster_data[:, i].std()
            print(f"  {feature}: mean={mean_val:.3f}, std={std_val:.3f}")
