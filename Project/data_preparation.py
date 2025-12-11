"""
Data preparation pipeline for disagreement prediction model.
Handles loading, preprocessing, feature engineering, and missing data imputation.
"""

import polars as pl
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import sys
from datetime import datetime

from disagreement_model.model_config import ModelConfig


class DataPreparator:
    """Handles all data preparation steps for the disagreement model"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.numeric_imputer = None
        self.categorical_imputer = None
        self.feature_names: List[str] = []
        self.missing_indicators: List[str] = []

    def load_data(self, filepath: Optional[str] = None) -> pl.DataFrame:
        """Load the CSV file with proper date parsing"""
        if filepath is None:
            filepath = self.config.data_path

        if self.config.verbose > 0:
            print(f"Loading data from {filepath}...")

        try:
            df = pl.read_csv(filepath, try_parse_dates=False)

            # Parse date columns
            for col in self.config.date_columns:
                if col in df.columns:
                    df = df.with_columns(
                        pl.col(col)
                        .str.strptime(pl.Date, format="%Y-%m-%d %H:%M:%S", strict=False)
                        .alias(col)
                    )

            if self.config.verbose > 0:
                print(f"✓ Loaded {df.shape[0]:,} rows × {df.shape[1]} columns\n")

            return df

        except Exception as e:
            print(f"✗ Error loading data: {e}")
            raise

    def create_target_variable(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create disagreement target variable(s)"""
        if self.config.verbose > 0:
            print("Creating target variable(s)...")

        # Filter out rows with missing recommendations
        df = df.filter(
            pl.col("algorithm_recommendation").is_not_null()
            & pl.col("professional_recommendation_clinician").is_not_null()
        )

        # Binary disagreement target
        df = df.with_columns(
            [
                (
                    pl.col("algorithm_recommendation")
                    != pl.col("professional_recommendation_clinician")
                )
                .cast(pl.Int32)
                .alias(self.config.target_column)
            ]
        )

        # Ordinal disagreement magnitude (if requested)
        if self.config.create_magnitude_target:
            df = df.with_columns(
                [
                    (
                        pl.col("algorithm_recommendation")
                        - pl.col("professional_recommendation_clinician")
                    )
                    .abs()
                    .cast(pl.Int32)
                    .alias(self.config.magnitude_target_column)
                ]
            )

        n_disagree = df.filter(pl.col(self.config.target_column) == 1).shape[0]
        n_agree = df.filter(pl.col(self.config.target_column) == 0).shape[0]

        if self.config.verbose > 0:
            print(
                f"  Disagreement cases: {n_disagree:,} ({n_disagree/df.shape[0]*100:.2f}%)"
            )
            print(f"  Agreement cases: {n_agree:,} ({n_agree/df.shape[0]*100:.2f}%)")
            if self.config.create_magnitude_target:
                mag_dist = (
                    df.group_by(self.config.magnitude_target_column)
                    .agg(pl.count().alias("count"))
                    .sort(self.config.magnitude_target_column)
                )
                print(f"\n  Disagreement magnitude distribution:")
                for row in mag_dist.iter_rows():
                    print(f"    |Δ| = {row[0]}: {row[1]:,} cases")
            print()

        return df

    def engineer_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create engineered features from existing columns"""
        if self.config.verbose > 0:
            print("Engineering features...")

        # Calculate age from date_of_birth
        if "date_of_birth" in df.columns and "assessment_date" in df.columns:
            df = df.with_columns(
                [
                    (
                        (
                            pl.col("assessment_date") - pl.col("date_of_birth")
                        ).dt.total_days()
                        / 365.25
                    ).alias("age")
                ]
            )
            if self.config.verbose > 1:
                print("  ✓ Created 'age' from date_of_birth")

        # Calculate time delays between dates
        date_pairs = [
            ("request_date", "assessment_date", "days_request_to_assessment"),
            ("assessment_date", "evaluation_date", "days_assessment_to_evaluation"),
            ("request_date", "publication_date", "days_request_to_publication"),
        ]

        for start_col, end_col, new_col in date_pairs:
            if start_col in df.columns and end_col in df.columns:
                df = df.with_columns(
                    [
                        ((pl.col(end_col) - pl.col(start_col)).dt.total_days()).alias(
                            new_col
                        )
                    ]
                )
                if self.config.verbose > 1:
                    print(f"  ✓ Created '{new_col}'")

        # PHQ-9 individual item scores (if needed)
        phq9_items = [f"phq9_{i}" for i in range(1, 10)]
        if all(col in df.columns for col in phq9_items):
            df = df.with_columns(
                [
                    pl.sum_horizontal([pl.col(item) for item in phq9_items]).alias(
                        "phq9_computed_total"
                    )
                ]
            )
            if self.config.verbose > 1:
                print("  ✓ Created 'phq9_computed_total'")

        # GAD-7 individual item scores
        gad7_items = [f"gad7_{i}" for i in range(1, 8)]
        if all(col in df.columns for col in gad7_items):
            df = df.with_columns(
                [
                    pl.sum_horizontal([pl.col(item) for item in gad7_items]).alias(
                        "gad7_computed_total"
                    )
                ]
            )
            if self.config.verbose > 1:
                print("  ✓ Created 'gad7_computed_total'")

        # Risk score combinations
        if "phq9_total_score" in df.columns and "gad7_total_score" in df.columns:
            df = df.with_columns(
                [
                    (pl.col("phq9_total_score") + pl.col("gad7_total_score")).alias(
                        "combined_depression_anxiety_score"
                    )
                ]
            )
            if self.config.verbose > 1:
                print("  ✓ Created 'combined_depression_anxiety_score'")

        # Psychosis symptom count
        psychosis_cols = [col for col in df.columns if col.startswith("psychosis_")]
        if len(psychosis_cols) > 0:
            df = df.with_columns(
                [
                    pl.sum_horizontal(
                        [pl.col(col).fill_null(0) for col in psychosis_cols]
                    ).alias("psychosis_symptom_count")
                ]
            )
            if self.config.verbose > 1:
                print("  ✓ Created 'psychosis_symptom_count'")

        if self.config.verbose > 0:
            print(f"  Total features after engineering: {df.shape[1]}\n")

        return df

    def select_features(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, List[str]]:
        """Select features based on configuration"""
        if self.config.verbose > 0:
            print("Selecting features...")

        # Get target columns
        target_cols = [self.config.target_column]
        if self.config.create_magnitude_target:
            target_cols.append(self.config.magnitude_target_column)

        # Get feature specification
        feature_spec = self.config.get_features()

        if feature_spec == "all":
            # Use all columns except excluded ones and targets
            all_cols = df.columns
            feature_cols = [
                col
                for col in all_cols
                if col not in self.config.exclude_columns
                and col not in target_cols
                and col not in self.config.date_columns  # Exclude raw date columns
            ]
        else:
            # Use specified features
            feature_cols = [col for col in feature_spec if col in df.columns]
            missing_features = [col for col in feature_spec if col not in df.columns]
            if missing_features and self.config.verbose > 0:
                print(f"  Warning: Features not found in data: {missing_features}")

        # Keep only target and feature columns
        keep_cols = target_cols + feature_cols
        df = df.select(keep_cols)

        if self.config.verbose > 0:
            print(f"  Selected {len(feature_cols)} features")
            print(f"  Target column(s): {target_cols}\n")

        return df, feature_cols

    def handle_missing_data(
        self, X: pd.DataFrame, y: pd.Series, is_training: bool = True
    ) -> pd.DataFrame:
        """Handle missing data according to configuration strategy"""
        if self.config.verbose > 0 and is_training:
            print(
                f"Handling missing data (strategy: {self.config.missing_strategy})..."
            )

        strategy = self.config.missing_strategy

        if strategy == "drop":
            return self._handle_missing_drop(X, is_training)
        elif strategy == "simple":
            return self._handle_missing_simple(X, is_training)
        elif strategy == "indicator":
            return self._handle_missing_indicator(X, is_training)
        elif strategy == "iterative":
            return self._handle_missing_iterative(X, is_training)
        else:
            raise ValueError(f"Unknown missing_strategy: {strategy}")

    def _handle_missing_drop(self, X: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Drop columns with too many missing values"""
        if is_training:
            missing_pct = X.isnull().sum() / len(X)
            self.columns_to_keep = missing_pct[
                missing_pct <= self.config.missing_threshold
            ].index.tolist()

            if self.config.verbose > 0:
                n_dropped = len(X.columns) - len(self.columns_to_keep)
                print(
                    f"  Dropped {n_dropped} columns with >{self.config.missing_threshold*100}% missing"
                )

        X = X[self.columns_to_keep].copy()

        # Still need to impute remaining missing values
        return self._handle_missing_simple(X, is_training)

    def _handle_missing_simple(
        self, X: pd.DataFrame, is_training: bool
    ) -> pd.DataFrame:
        """Simple imputation: median for numeric, mode for categorical"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        if is_training:
            # Filter out columns with all missing values (can't be imputed)
            if len(numeric_cols) > 0:
                numeric_cols_valid = [
                    col for col in numeric_cols if X[col].notna().any()
                ]
                numeric_cols_dropped = [
                    col for col in numeric_cols if col not in numeric_cols_valid
                ]

                if numeric_cols_dropped and self.config.verbose > 0:
                    print(
                        f"  Warning: Dropping {len(numeric_cols_dropped)} numeric columns with all missing values: {numeric_cols_dropped}"
                    )

                # Drop columns with all missing values
                X = X.drop(columns=numeric_cols_dropped)
                numeric_cols = numeric_cols_valid

            if len(categorical_cols) > 0:
                categorical_cols_valid = [
                    col for col in categorical_cols if X[col].notna().any()
                ]
                categorical_cols_dropped = [
                    col for col in categorical_cols if col not in categorical_cols_valid
                ]

                if categorical_cols_dropped and self.config.verbose > 0:
                    print(
                        f"  Warning: Dropping {len(categorical_cols_dropped)} categorical columns with all missing values: {categorical_cols_dropped}"
                    )

                # Drop columns with all missing values
                X = X.drop(columns=categorical_cols_dropped)
                categorical_cols = categorical_cols_valid

            # Store valid columns for test data
            self.numeric_cols_to_impute = numeric_cols
            self.categorical_cols_to_impute = categorical_cols

            # Fit imputers
            if len(numeric_cols) > 0:
                self.numeric_imputer = SimpleImputer(
                    strategy=self.config.numeric_imputation
                )
                X[numeric_cols] = self.numeric_imputer.fit_transform(X[numeric_cols])

            if len(categorical_cols) > 0:
                self.categorical_imputer = SimpleImputer(
                    strategy=self.config.categorical_imputation
                )
                X[categorical_cols] = self.categorical_imputer.fit_transform(
                    X[categorical_cols]
                )

            if self.config.verbose > 0:
                print(
                    f"  Imputed {len(numeric_cols)} numeric and {len(categorical_cols)} categorical columns"
                )
        else:
            # Transform only - use the same columns as training
            # Drop columns that were dropped during training
            all_dropped = set(numeric_cols + categorical_cols) - set(
                self.numeric_cols_to_impute + self.categorical_cols_to_impute
            )
            if all_dropped:
                X = X.drop(
                    columns=[col for col in all_dropped if col in X.columns],
                    errors="ignore",
                )

            numeric_cols = self.numeric_cols_to_impute
            categorical_cols = self.categorical_cols_to_impute

            if len(numeric_cols) > 0 and self.numeric_imputer is not None:
                X[numeric_cols] = self.numeric_imputer.transform(X[numeric_cols])

            if len(categorical_cols) > 0 and self.categorical_imputer is not None:
                X[categorical_cols] = self.categorical_imputer.transform(
                    X[categorical_cols]
                )

        return X

    def _handle_missing_indicator(
        self, X: pd.DataFrame, is_training: bool
    ) -> pd.DataFrame:
        """Simple imputation + add binary indicators for missingness"""
        if is_training:
            # Identify columns with missing values
            missing_cols = X.columns[X.isnull().any()].tolist()
            self.missing_indicators = [f"{col}_missing" for col in missing_cols]

            # Add missingness indicators before imputation
            for col in missing_cols:
                X[f"{col}_missing"] = X[col].isnull().astype(int)

            if self.config.verbose > 0:
                print(f"  Added {len(missing_cols)} missingness indicators")
        else:
            # Add indicators for test data
            for col, indicator_name in zip(
                [ind.replace("_missing", "") for ind in self.missing_indicators],
                self.missing_indicators,
            ):
                if col in X.columns:
                    X[indicator_name] = X[col].isnull().astype(int)

        # Now do simple imputation
        return self._handle_missing_simple(X, is_training)

    def _handle_missing_iterative(
        self, X: pd.DataFrame, is_training: bool
    ) -> pd.DataFrame:
        """Iterative imputation using IterativeImputer (MICE algorithm)"""
        # Only impute numeric columns with iterative method
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        if is_training:
            # Filter out columns with all missing values (can't be imputed)
            if len(numeric_cols) > 0:
                numeric_cols_valid = [
                    col for col in numeric_cols if X[col].notna().any()
                ]
                numeric_cols_dropped = [
                    col for col in numeric_cols if col not in numeric_cols_valid
                ]

                if numeric_cols_dropped and self.config.verbose > 0:
                    print(
                        f"  Warning: Dropping {len(numeric_cols_dropped)} numeric columns with all missing values: {numeric_cols_dropped}"
                    )

                X = X.drop(columns=numeric_cols_dropped)
                numeric_cols = numeric_cols_valid

            if len(categorical_cols) > 0:
                categorical_cols_valid = [
                    col for col in categorical_cols if X[col].notna().any()
                ]
                categorical_cols_dropped = [
                    col for col in categorical_cols if col not in categorical_cols_valid
                ]

                if categorical_cols_dropped and self.config.verbose > 0:
                    print(
                        f"  Warning: Dropping {len(categorical_cols_dropped)} categorical columns with all missing values: {categorical_cols_dropped}"
                    )

                X = X.drop(columns=categorical_cols_dropped)
                categorical_cols = categorical_cols_valid

            # Store valid columns for test data
            self.numeric_cols_to_impute = numeric_cols
            self.categorical_cols_to_impute = categorical_cols

            if len(numeric_cols) > 0:
                self.numeric_imputer = IterativeImputer(
                    random_state=self.config.random_state, max_iter=10, verbose=0
                )
                X[numeric_cols] = self.numeric_imputer.fit_transform(X[numeric_cols])

            if len(categorical_cols) > 0:
                # Use simple imputation for categorical
                self.categorical_imputer = SimpleImputer(strategy="most_frequent")
                X[categorical_cols] = self.categorical_imputer.fit_transform(
                    X[categorical_cols]
                )

            if self.config.verbose > 0:
                print(f"  Iteratively imputed {len(numeric_cols)} numeric columns")
                print(f"  Simple imputed {len(categorical_cols)} categorical columns")
        else:
            # Transform only - use the same columns as training
            # Drop columns that were dropped during training
            all_dropped = set(numeric_cols + categorical_cols) - set(
                self.numeric_cols_to_impute + self.categorical_cols_to_impute
            )
            if all_dropped:
                X = X.drop(
                    columns=[col for col in all_dropped if col in X.columns],
                    errors="ignore",
                )

            numeric_cols = self.numeric_cols_to_impute
            categorical_cols = self.categorical_cols_to_impute

            if len(numeric_cols) > 0 and self.numeric_imputer is not None:
                X[numeric_cols] = self.numeric_imputer.transform(X[numeric_cols])

            if len(categorical_cols) > 0 and self.categorical_imputer is not None:
                X[categorical_cols] = self.categorical_imputer.transform(
                    X[categorical_cols]
                )

        return X

    def encode_categorical(
        self, X: pd.DataFrame, is_training: bool = True
    ) -> pd.DataFrame:
        """Encode categorical variables with label encoding"""
        categorical_cols = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        if len(categorical_cols) == 0:
            return X

        if is_training:
            if self.config.verbose > 0:
                print(f"Encoding {len(categorical_cols)} categorical columns...")

            for col in categorical_cols:
                le = LabelEncoder()
                # Handle unseen categories by adding a placeholder
                X[col] = X[col].fillna("MISSING_CATEGORY")
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
        else:
            for col in categorical_cols:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    X[col] = X[col].fillna("MISSING_CATEGORY")
                    # Handle unseen categories
                    X[col] = X[col].apply(
                        lambda x: (
                            le.transform([str(x)])[0] if str(x) in le.classes_ else -1
                        )
                    )

        return X

    def prepare_data(
        self, filepath: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str]]:
        """
        Complete data preparation pipeline.

        Returns:
            X_train, y_train, X_test, y_test, feature_names
        """
        # Load data
        df = self.load_data(filepath)

        # Create target variable
        df = self.create_target_variable(df)

        # Engineer features
        df = self.engineer_features(df)

        # Select features
        df, feature_cols = self.select_features(df)

        # Convert to pandas for sklearn compatibility
        df_pd = df.to_pandas()

        # Split features and target
        X = df_pd[feature_cols].copy()
        y = df_pd[self.config.target_column].copy()

        # Store original feature names
        self.feature_names = X.columns.tolist()

        if self.config.verbose > 0:
            print(f"Dataset shape before preprocessing: {X.shape}")
            print(f"  Features: {X.shape[1]}")
            print(f"  Samples: {X.shape[0]}")
            print(f"  Target distribution: {y.value_counts().to_dict()}\n")

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )

        if self.config.verbose > 0:
            print(f"Train-test split:")
            print(
                f"  Train: {X_train.shape[0]:,} samples ({X_train.shape[0]/X.shape[0]*100:.1f}%)"
            )
            print(
                f"  Test:  {X_test.shape[0]:,} samples ({X_test.shape[0]/X.shape[0]*100:.1f}%)\n"
            )

        # Handle missing data
        X_train = self.handle_missing_data(X_train, y_train, is_training=True)
        X_test = self.handle_missing_data(X_test, y_test, is_training=False)

        # Encode categorical variables
        X_train = self.encode_categorical(X_train, is_training=True)
        X_test = self.encode_categorical(X_test, is_training=False)

        # Update feature names after preprocessing
        final_feature_names = X_train.columns.tolist()

        if self.config.verbose > 0:
            print(f"Final dataset shape: {X_train.shape}")
            print(f"  Final features: {len(final_feature_names)}\n")
            print("=" * 80)

        return X_train, y_train, X_test, y_test, final_feature_names
