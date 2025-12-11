"""
Decision tree model with SMAC3 hyperparameter optimization for disagreement prediction.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformIntegerHyperparameter,
    UniformFloatHyperparameter,
    CategoricalHyperparameter,
)
from smac import HyperparameterOptimizationFacade, Scenario
import pickle
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import sys
from datetime import datetime

from .model_config import ModelConfig
from data_preparation import DataPreparator


class DisagreementModel:
    """Decision tree model with SMAC3 optimization"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model: Optional[DecisionTreeClassifier] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.cv_score: Optional[float] = None
        self.test_metrics: Dict[str, float] = {}
        self.feature_names: list = []

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def create_configspace(self) -> ConfigurationSpace:
        """Create the hyperparameter search space for SMAC3"""
        cs = ConfigurationSpace()

        # Max depth
        max_depth = UniformIntegerHyperparameter(
            "max_depth",
            lower=self.config.hp_max_depth[0],
            upper=self.config.hp_max_depth[1],
        )

        # Min samples split
        min_samples_split = UniformIntegerHyperparameter(
            "min_samples_split",
            lower=self.config.hp_min_samples_split[0],
            upper=self.config.hp_min_samples_split[1],
        )

        # Min samples leaf
        min_samples_leaf = UniformIntegerHyperparameter(
            "min_samples_leaf",
            lower=self.config.hp_min_samples_leaf[0],
            upper=self.config.hp_min_samples_leaf[1],
        )

        # Min impurity decrease
        min_impurity_decrease = UniformFloatHyperparameter(
            "min_impurity_decrease",
            lower=self.config.hp_min_impurity_decrease[0],
            upper=self.config.hp_min_impurity_decrease[1],
        )

        # Criterion
        criterion = CategoricalHyperparameter(
            "criterion", choices=self.config.hp_criterion
        )

        # Splitter
        splitter = CategoricalHyperparameter(
            "splitter", choices=self.config.hp_splitter
        )

        # Max features (convert None to string for SMAC)
        max_features_choices = [
            str(x) if x is not None else "None" for x in self.config.hp_max_features
        ]
        max_features = CategoricalHyperparameter(
            "max_features", choices=max_features_choices
        )

        # Class weight
        class_weight_choices = [
            str(x) if x is not None else "None" for x in self.config.hp_class_weight
        ]
        class_weight = CategoricalHyperparameter(
            "class_weight", choices=class_weight_choices
        )

        cs.add(
            [
                max_depth,
                min_samples_split,
                min_samples_leaf,
                min_impurity_decrease,
                criterion,
                splitter,
                max_features,
                class_weight,
            ]
        )

        return cs

    def train_and_evaluate(
        self,
        config: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        seed: int = 0,
    ) -> float:
        """
        Train model with given hyperparameters and return CV score.
        This is the objective function for SMAC3.
        """
        # Convert string representations back to proper types
        params = dict(config)
        params["max_features"] = (
            None if params["max_features"] == "None" else params["max_features"]
        )
        params["class_weight"] = (
            None if params["class_weight"] == "None" else params["class_weight"]
        )
        params["random_state"] = self.config.random_state

        # Create model
        model = DecisionTreeClassifier(**params)

        # Cross-validation
        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )

        scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring=self.config.cv_scoring, n_jobs=1
        )

        # Return negative mean score (SMAC minimizes)
        return -scores.mean()

    def optimize_hyperparameters(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Dict[str, Any]:
        """Run SMAC3 optimization to find best hyperparameters"""
        if self.config.verbose > 0:
            print("=" * 80)
            print("HYPERPARAMETER OPTIMIZATION WITH SMAC3")
            print("=" * 80)
            print(f"Optimization settings:")
            print(f"  Max trials: {self.config.smac_n_trials}")
            print(f"  Walltime limit: {self.config.smac_walltime_limit}s")
            print(f"  CV folds: {self.config.cv_folds}")
            print(f"  Scoring: {self.config.cv_scoring}")
            print(f"  Random state: {self.config.random_state}\n")

        # Create configuration space
        configspace = self.create_configspace()

        # Create scenario
        scenario = Scenario(
            configspace=configspace,
            deterministic=True,
            n_trials=self.config.smac_n_trials,
            walltime_limit=self.config.smac_walltime_limit,
            n_workers=self.config.smac_n_workers,
        )

        # Create SMAC optimizer
        smac = HyperparameterOptimizationFacade(
            scenario=scenario,
            target_function=lambda config, seed: self.train_and_evaluate(
                config, X_train, y_train, seed
            ),
            overwrite=True,
        )

        # Run optimization
        if self.config.verbose > 0:
            print("Starting SMAC3 optimization...\n")

        incumbent = smac.optimize()

        # Get best parameters
        best_params = dict(incumbent)
        best_params["max_features"] = (
            None
            if best_params["max_features"] == "None"
            else best_params["max_features"]
        )
        best_params["class_weight"] = (
            None
            if best_params["class_weight"] == "None"
            else best_params["class_weight"]
        )
        best_params["random_state"] = self.config.random_state

        # Get best score (negate back since we minimized negative score)
        incumbent_cost = smac.runhistory.get_cost(incumbent)
        best_score = -incumbent_cost

        if self.config.verbose > 0:
            print("\n" + "=" * 80)
            print("OPTIMIZATION COMPLETE")
            print("=" * 80)
            print(f"Best {self.config.cv_scoring} score: {best_score:.4f}\n")
            print("Best hyperparameters:")
            for param, value in best_params.items():
                if param != "random_state":
                    print(f"  {param}: {value}")
            print("\n" + "=" * 80 + "\n")

        self.best_params = best_params
        self.cv_score = best_score

        return best_params

    def train_final_model(
        self, X_train: pd.DataFrame, y_train: pd.Series, feature_names: list
    ):
        """Train final model with best hyperparameters on full training set"""
        if self.config.verbose > 0:
            print("Training final model with best hyperparameters...")

        self.feature_names = feature_names
        self.model = DecisionTreeClassifier(**self.best_params)
        self.model.fit(X_train, y_train)

        if self.config.verbose > 0:
            print(f"✓ Model trained successfully")
            print(f"  Tree depth: {self.model.get_depth()}")
            print(f"  Number of leaves: {self.model.get_n_leaves()}")
            print(f"  Number of features: {self.model.n_features_in_}\n")

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model on test set"""
        if self.config.verbose > 0:
            print("=" * 80)
            print("MODEL EVALUATION")
            print("=" * 80)

        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
        }

        self.test_metrics = metrics

        if self.config.verbose > 0:
            print(f"Test Set Performance:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1']:.4f}")
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}\n")

        # Confusion matrix
        if self.config.compute_confusion_matrix:
            cm = confusion_matrix(y_test, y_pred)
            if self.config.verbose > 0:
                print("Confusion Matrix:")
                print("                Predicted")
                print("              Agree  Disagree")
                print(f"Actual Agree     {cm[0,0]:5d}  {cm[0,1]:5d}")
                print(f"     Disagree    {cm[1,0]:5d}  {cm[1,1]:5d}\n")

        # Classification report
        if self.config.compute_classification_report:
            report = classification_report(
                y_test, y_pred, target_names=["Agreement", "Disagreement"], digits=4
            )
            if self.config.verbose > 0:
                print("Classification Report:")
                print(report)

        if self.config.verbose > 0:
            print("=" * 80 + "\n")

        return metrics

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from trained model"""
        importances = self.model.feature_importances_
        importance_df = pd.DataFrame(
            {"feature": self.feature_names, "importance": importances}
        ).sort_values("importance", ascending=False)

        return importance_df

    def load_model(self, filepath: Optional[str] = None):
        """Load trained model from disk"""
        if filepath is None:
            filepath = Path(self.config.output_dir) / self.config.model_filename

        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.best_params = model_data["best_params"]
        self.cv_score = model_data["cv_score"]
        self.test_metrics = model_data.get("test_metrics", {})
        self.feature_names = model_data["feature_names"]

        if self.config.verbose > 0:
            print(f"✓ Model loaded from {filepath}")
            print(f"  Tree depth: {self.model.get_depth()}")
            print(f"  Number of leaves: {self.model.get_n_leaves()}")
            print(f"  Number of features: {len(self.feature_names)}")
            print(f"  CV score: {self.cv_score:.4f}\n")

        return model_data

    def save_model(self, filepath: Optional[str] = None):
        """Save trained model to disk"""
        if filepath is None:
            filepath = Path(self.config.output_dir) / self.config.model_filename

        model_data = {
            "model": self.model,
            "best_params": self.best_params,
            "cv_score": self.cv_score,
            "test_metrics": self.test_metrics,
            "feature_names": self.feature_names,
            "config": self.config.to_dict(),
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        if self.config.verbose > 0:
            print(f"✓ Model saved to {filepath}")

    def save_results(self):
        """Save comprehensive results to text file"""
        results_file = (
            Path(self.config.output_dir) / f"{self.config.results_prefix}_results.txt"
        )

        with open(results_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("DISAGREEMENT PREDICTION MODEL - RESULTS SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Configuration
            f.write("CONFIGURATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Data path: {self.config.data_path}\n")
            f.write(f"Feature mode: {self.config.feature_mode}\n")
            f.write(f"Missing strategy: {self.config.missing_strategy}\n")
            f.write(f"Test size: {self.config.test_size}\n")
            f.write(f"Random state: {self.config.random_state}\n")
            f.write(f"SMAC trials: {self.config.smac_n_trials}\n")
            f.write(f"CV folds: {self.config.cv_folds}\n")
            f.write(f"CV scoring: {self.config.cv_scoring}\n\n")

            # Best hyperparameters
            f.write("BEST HYPERPARAMETERS\n")
            f.write("-" * 80 + "\n")
            for param, value in self.best_params.items():
                f.write(f"{param}: {value}\n")
            f.write(
                f"\nCross-validation {self.config.cv_scoring}: {self.cv_score:.4f}\n\n"
            )

            # Model architecture
            f.write("MODEL ARCHITECTURE\n")
            f.write("-" * 80 + "\n")
            f.write(f"Tree depth: {self.model.get_depth()}\n")
            f.write(f"Number of leaves: {self.model.get_n_leaves()}\n")
            f.write(f"Number of features: {len(self.feature_names)}\n\n")

            # Test performance
            f.write("TEST SET PERFORMANCE\n")
            f.write("-" * 80 + "\n")
            for metric, value in self.test_metrics.items():
                f.write(f"{metric.upper()}: {value:.4f}\n")
            f.write("\n")

            # Feature importance
            f.write("TOP 20 MOST IMPORTANT FEATURES\n")
            f.write("-" * 80 + "\n")
            importance_df = self.get_feature_importance()
            for i, row in importance_df.head(20).iterrows():
                f.write(f"{row['feature']:50s} {row['importance']:.6f}\n")
            f.write("\n")

        if self.config.verbose > 0:
            print(f"✓ Results saved to {results_file}")

        # Save metrics as JSON
        metrics_file = (
            Path(self.config.output_dir) / f"{self.config.results_prefix}_metrics.json"
        )
        with open(metrics_file, "w") as f:
            json.dump(
                {
                    "config": self.config.to_dict(),
                    "best_params": self.best_params,
                    "cv_score": self.cv_score,
                    "test_metrics": self.test_metrics,
                },
                f,
                indent=2,
            )

        if self.config.verbose > 0:
            print(f"✓ Metrics saved to {metrics_file}\n")


def main():
    """Main training pipeline"""
    # Load configuration
    config = ModelConfig()

    print("=" * 80)
    print("DISAGREEMENT PREDICTION MODEL TRAINING")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Prepare data
    print("STEP 1: DATA PREPARATION")
    print("-" * 80)
    preparator = DataPreparator(config)
    X_train, y_train, X_test, y_test, feature_names = preparator.prepare_data()

    # Initialize model
    print("\nSTEP 2: MODEL TRAINING")
    print("-" * 80)
    model = DisagreementModel(config)

    # Optimize hyperparameters
    best_params = model.optimize_hyperparameters(X_train, y_train)

    # Train final model
    model.train_final_model(X_train, y_train, feature_names)

    # Evaluate
    print("\nSTEP 3: EVALUATION")
    print("-" * 80)
    metrics = model.evaluate(X_test, y_test)

    # Save model and results
    print("\nSTEP 4: SAVING RESULTS")
    print("-" * 80)
    if config.save_model:
        model.save_model()
    model.save_results()

    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Test F1-Score: {metrics['f1']:.4f}")
    print(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
    print("=" * 80)

    return model


if __name__ == "__main__":
    model = main()
