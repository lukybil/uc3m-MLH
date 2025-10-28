"""
Hawkes process fitting and analysis module.
"""

import os
import numpy as np
import pandas as pd

try:
    from hawkeslib.model import exp_kernel as hawkes_exp
except ImportError:
    hawkes_exp = None


def prepare_hawkes_data(event_sequences, features_df, cluster_labels):
    """
    Returns:
        dict: Dictionary mapping cluster IDs to event sequences
    """
    print("\n" + "=" * 80)
    print("PREPARING DATA FOR HAWKES PROCESS")
    print("=" * 80)

    cluster_sequences = {i: [] for i in range(len(np.unique(cluster_labels)))}

    for event_seq in event_sequences:
        patient_id = event_seq["patient_id"]
        patient_idx = features_df[features_df["ID"] == patient_id].index
        if len(patient_idx) > 0:
            cluster = cluster_labels[patient_idx[0]]
            cluster_sequences[cluster].append(event_seq)

    for cluster_id, sequences in cluster_sequences.items():
        n_patients = len(sequences)
        n_events = sum([seq["n_events"] for seq in sequences])
        avg_events = n_events / n_patients if n_patients > 0 else 0
        print(f"\nCluster {cluster_id}:")
        print(f"  Patients: {n_patients}")
        print(f"  Total events: {n_events}")
        print(f"  Average events per patient: {avg_events:.2f}")

    return cluster_sequences


def compute_basic_stats(cluster_id, sequences):
    """
    Returns:
        dict: Dictionary containing computed statistics
    """
    all_events = []
    all_inter_arrival = []

    for seq in sequences:
        events = seq["events"]
        all_events.extend(events)
        if len(events) > 1:
            inter_arrival = np.diff(events)
            all_inter_arrival.extend(inter_arrival)

    mean_inter_arrival = np.mean(all_inter_arrival) if all_inter_arrival else 0
    std_inter_arrival = np.std(all_inter_arrival) if all_inter_arrival else 0
    total_time = sum([seq["end_time"] for seq in sequences])
    total_events = len(all_events)
    event_rate = total_events / total_time if total_time > 0 else 0

    params = {
        "event_rate": event_rate,
        "mean_inter_arrival": mean_inter_arrival,
        "std_inter_arrival": std_inter_arrival,
        "total_events": total_events,
        "n_patients": len(sequences),
    }

    print(f"  Event rate (events/month): {event_rate:.4f}")
    print(f"  Mean inter-arrival time: {mean_inter_arrival:.4f} months")
    print(f"  Std inter-arrival time: {std_inter_arrival:.4f} months")
    print(f"  Total events: {total_events}")

    return params


def fit_hawkes_processes(cluster_sequences):
    """
    Returns:
        tuple: (hawkes_models, hawkes_params)
    """
    print("\n" + "=" * 80)
    print("FITTING HAWKES PROCESSES")
    print("=" * 80)

    hawkes_models = {}
    hawkes_params = {}

    for cluster_id, sequences in cluster_sequences.items():
        print(f"\n{'='*80}")
        print(f"CLUSTER {cluster_id}")
        print(f"{'='*80}")

        if len(sequences) == 0:
            print("  No patients in this cluster, skipping...")
            continue

        params = compute_basic_stats(cluster_id, sequences)
        hawkes_params[cluster_id] = params

    return hawkes_models, hawkes_params


def compare_hawkes_parameters(hawkes_params, output_dir="results"):
    """
    Returns:
        DataFrame: Parameters comparison table
    """
    print("\n" + "=" * 80)
    print("COMPARISON OF HAWKES PARAMETERS ACROSS CLUSTERS")
    print("=" * 80)

    params_df = pd.DataFrame(hawkes_params).T
    print("\n", params_df)

    params_df.to_csv(os.path.join(output_dir, "hawkes_parameters_comparison.csv"))
    print(
        f"\nParameters saved to '{os.path.join(output_dir, 'hawkes_parameters_comparison.csv')}'"
    )

    return params_df
