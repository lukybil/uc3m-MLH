import numpy as np


def preprocess_timelines(timelines_df):
    """
    Returns:
        tuple: (event_sequences, patient_ids)
    """
    print("\nPreprocessing timelines...")

    time_cols = [col for col in timelines_df.columns if col.startswith("time_")]
    timeline_data = timelines_df[time_cols].values
    absolute_times = np.cumsum(timeline_data, axis=1)

    event_sequences = []
    patient_ids = []

    for idx, patient_id in enumerate(timelines_df["ID"]):
        times = absolute_times[idx]
        valid_events = times[times > 0]

        if len(valid_events) > 0:
            end_time = valid_events[-1] + 1
            event_list = list(valid_events)

            event_sequences.append(
                {
                    "patient_id": patient_id,
                    "events": event_list,
                    "end_time": end_time,
                    "n_events": len(event_list),
                }
            )
            patient_ids.append(patient_id)

    print(f"Processed {len(event_sequences)} patient timelines")
    print(
        f"Average number of events per patient: {np.mean([seq['n_events'] for seq in event_sequences]):.2f}"
    )

    return event_sequences, patient_ids
