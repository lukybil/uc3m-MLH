import os
import warnings
import pandas as pd

from data_loader import DataLoader
from preprocessing import preprocess_timelines
from clustering import cluster_patients
from cluster_analysis import analyze_clusters
from visualization import (
    plot_clustering_metrics,
    visualize_clusters,
    plot_event_distributions,
    plot_hawkes_parameters,
)
from hawkes_process import (
    prepare_hawkes_data,
    fit_hawkes_processes,
    compare_hawkes_parameters,
)

warnings.filterwarnings("ignore")

OUTPUT_DIR = "results"


def run_full_analysis(
    features_path="data/features.csv", timelines_path="data/timelines.csv"
):
    print("\n" + "=" * 80)
    print("STARTING FULL ANALYSIS")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    loader = DataLoader(features_path, timelines_path)
    features_df, timelines_df = loader.load_data()

    event_sequences, patient_ids = preprocess_timelines(timelines_df)

    (
        cluster_labels,
        kmeans_model,
        scaler,
        X,
        X_scaled,
        feature_cols,
        inertias,
        silhouettes,
        davies_bouldin_scores,
        n_clusters_range,
    ) = cluster_patients(features_df)

    features_df["Cluster"] = cluster_labels

    analyze_clusters(X, feature_cols, cluster_labels)

    plot_clustering_metrics(
        inertias, silhouettes, davies_bouldin_scores, n_clusters_range, OUTPUT_DIR
    )
    visualize_clusters(X_scaled, cluster_labels, kmeans_model, feature_cols, OUTPUT_DIR)

    cluster_sequences = prepare_hawkes_data(
        event_sequences, features_df, cluster_labels
    )

    hawkes_models, hawkes_params = fit_hawkes_processes(cluster_sequences)

    if hawkes_params:
        params_df = compare_hawkes_parameters(hawkes_params, OUTPUT_DIR)
        plot_hawkes_parameters(params_df, OUTPUT_DIR)

    plot_event_distributions(cluster_sequences, OUTPUT_DIR)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_full_analysis(
        features_path="data/features.csv", timelines_path="data/timelines.csv"
    )
