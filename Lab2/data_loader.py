import pandas as pd


class DataLoader:
    def __init__(
        self, features_path="data/features.csv", timelines_path="data/timelines.csv"
    ):
        self.features_path = features_path
        self.timelines_path = timelines_path
        self.features_df = None
        self.timelines_df = None

    def load_data(self):
        """
        Returns:
            tuple: (features_df, timelines_df)
        """
        print("Loading data...")
        self.features_df = pd.read_csv(self.features_path, sep=";")
        self.timelines_df = pd.read_csv(self.timelines_path, sep=";")

        print(f"Features shape: {self.features_df.shape}")
        print(f"Timelines shape: {self.timelines_df.shape}")
        print(f"\nFeatures columns: {self.features_df.columns.tolist()}")
        print(f"\nFirst few rows of features:")
        print(self.features_df.head())

        return self.features_df, self.timelines_df
