"""
Ablation study framework for systematic feature group analysis.

Runs baseline model with SMAC3 optimization, then evaluates ablation experiments
using the same hyperparameters to isolate feature group contributions.
Supports parallel execution with proper signal handling and McNemar's test.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import pickle
from datetime import datetime
import signal
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from functools import partial
import traceback

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# Try to import McNemar's test, fallback to custom implementation
try:
    from statsmodels.stats.contingency_tables import mcnemar

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    # Will use custom implementation below

from .model_config import ModelConfig
from .ablation_config import AblationConfig
from .disagreement_model import DisagreementModel
from data_preparation import DataPreparator


def _custom_mcnemar_test(table, exact=True):
    """
    Custom implementation of McNemar's test when statsmodels is not available.

    Args:
        table: 2x2 contingency table [[n11, n12], [n21, n22]]
        exact: Use exact binomial test (always True for simplicity)

    Returns:
        Object with statistic and pvalue attributes
    """
    from scipy.stats import binom

    n11, n12 = table[0]
    n21, n22 = table[1]

    # McNemar's test statistic focuses on discordant pairs (n12 and n21)
    n = n12 + n21  # Total discordant pairs

    if n == 0:
        # No disagreement between models
        class Result:
            statistic = 0.0
            pvalue = 1.0

        return Result()

    # Under null hypothesis, both models equally likely to be correct
    # So discordant pairs should split 50-50
    # Two-tailed binomial test
    k = min(n12, n21)  # More extreme outcome
    pvalue = 2 * binom.cdf(k, n, 0.5)
    pvalue = min(pvalue, 1.0)  # Cap at 1.0

    # Chi-square statistic (with continuity correction for small n)
    if n > 25:
        statistic = (abs(n12 - n21) - 1) ** 2 / (n12 + n21)
    else:
        statistic = (n12 - n21) ** 2 / (n12 + n21) if (n12 + n21) > 0 else 0

    class Result:
        pass

    result = Result()
    result.statistic = statistic
    result.pvalue = pvalue

    return result


class AblationStudy:
    """Manages ablation study experiments with parallel execution"""

    def __init__(self, config: ModelConfig, ablation_config: AblationConfig):
        self.config = config
        self.ablation_config = ablation_config
        self.results = {}
        self.baseline_params = None
        self.baseline_predictions = None
        self.baseline_metrics = None

        # Create output directory
        Path(config.ablation_output_dir).mkdir(parents=True, exist_ok=True)

        # Signal handling for Ctrl+C
        self.interrupted = False
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        print("\n\n" + "=" * 80)
        print("ABLATION STUDY INTERRUPTED BY USER (Ctrl+C)")
        print("=" * 80)
        print("Saving partial results...")
        self.interrupted = True

        # Save partial results
        if self.results:
            self._save_partial_results()

        print("Partial results saved. Exiting...")
        sys.exit(0)

    def run_baseline(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_names: List[str],
    ) -> Dict[str, Any]:
        """
        Run baseline model with full SMAC3 optimization.
        This provides the hyperparameters to reuse for all ablation experiments.
        """
        if self.config.verbose > 0:
            print("\n" + "=" * 80)
            print("BASELINE MODEL (Full Feature Set)")
            print("=" * 80)
            print(f"Number of features: {len(feature_names)}")
            print(f"Training samples: {len(X_train)}")
            print(f"Test samples: {len(X_test)}")
            print()

        # Initialize model
        model = DisagreementModel(self.config)

        # Run SMAC3 optimization
        best_params = model.optimize_hyperparameters(X_train, y_train)

        # Train final model
        model.train_final_model(X_train, y_train, feature_names)

        # Evaluate
        metrics = model.evaluate(X_test, y_test)

        # Get predictions for McNemar's test
        y_pred = model.model.predict(X_test)
        y_pred_proba = model.model.predict_proba(X_test)[:, 1]

        # Store baseline info
        self.baseline_params = best_params
        self.baseline_predictions = {
            "y_true": y_test.values,
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba,
        }
        self.baseline_metrics = metrics

        # Get feature importance
        feature_importance = model.get_feature_importance()

        result = {
            "experiment": "baseline",
            "n_features": len(feature_names),
            "features": feature_names,
            "best_params": best_params,
            "cv_score": model.cv_score,
            "test_metrics": metrics,
            "tree_depth": model.model.get_depth(),
            "tree_leaves": model.model.get_n_leaves(),
            "feature_importance": feature_importance.to_dict("records"),
            "predictions": self.baseline_predictions,
        }

        # Save baseline model and params
        baseline_path = Path(self.config.ablation_output_dir) / "baseline_model.pkl"
        model.save_model(baseline_path)

        params_path = Path(self.config.ablation_output_dir) / "baseline_params.json"
        with open(params_path, "w") as f:
            json.dump(best_params, f, indent=2)

        if self.config.verbose > 0:
            print(f"\n✓ Baseline model saved to {baseline_path}")
            print(f"✓ Baseline params saved to {params_path}\n")

        self.results["baseline"] = result
        return result

    def run_single_ablation(
        self,
        experiment_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        all_feature_names: List[str],
        hyperparameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run a single ablation experiment with given hyperparameters.

        Args:
            experiment_name: Name of the ablation experiment
            X_train, y_train, X_test, y_test: Full datasets
            all_feature_names: Complete list of available features
            hyperparameters: Pre-optimized hyperparameters to use

        Returns:
            Dictionary with experiment results
        """
        try:
            # Get features for this experiment
            selected_features = self.ablation_config.get_features_for_experiment(
                experiment_name, all_feature_names
            )

            if self.config.verbose > 0:
                print(f"\n{'=' * 80}")
                print(f"ABLATION EXPERIMENT: {experiment_name}")
                print(f"{'=' * 80}")
                print(
                    f"Features selected: {len(selected_features)} / {len(all_feature_names)}"
                )
                print(
                    f"Features removed: {len(all_feature_names) - len(selected_features)}"
                )

            # Filter datasets to selected features
            X_train_filtered = X_train[selected_features]
            X_test_filtered = X_test[selected_features]

            # Create model with baseline hyperparameters
            model = DecisionTreeClassifier(**hyperparameters)

            # Train model
            model.fit(X_train_filtered, y_train)

            # Predictions
            y_pred = model.predict(X_test_filtered)
            y_pred_proba = model.predict_proba(X_test_filtered)[:, 1]

            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_pred_proba),
            }

            # Calculate performance delta from baseline
            metrics_delta = {}
            if self.baseline_metrics:
                for metric_name, value in metrics.items():
                    baseline_value = self.baseline_metrics[metric_name]
                    metrics_delta[f"{metric_name}_delta"] = value - baseline_value

            # McNemar's test (if enabled)
            mcnemar_result = None
            if self.config.run_mcnemar_test and self.baseline_predictions is not None:
                mcnemar_result = self._compute_mcnemar_test(
                    self.baseline_predictions["y_true"],
                    self.baseline_predictions["y_pred"],
                    y_pred,
                )

            # Feature importance
            importances = model.feature_importances_
            feature_importance = pd.DataFrame(
                {"feature": selected_features, "importance": importances}
            ).sort_values("importance", ascending=False)

            # Compile results
            result = {
                "experiment": experiment_name,
                "n_features": len(selected_features),
                "features": selected_features,
                "features_removed": list(
                    set(all_feature_names) - set(selected_features)
                ),
                "hyperparameters": hyperparameters,
                "test_metrics": metrics,
                "metrics_delta": metrics_delta,
                "tree_depth": model.get_depth(),
                "tree_leaves": model.get_n_leaves(),
                "feature_importance": feature_importance.to_dict("records"),
                "mcnemar_test": mcnemar_result,
                "predictions": {
                    "y_true": y_test.values,
                    "y_pred": y_pred,
                    "y_pred_proba": y_pred_proba,
                },
            }

            if self.config.verbose > 0:
                print(f"\nResults:")
                print(
                    f"  Accuracy:  {metrics['accuracy']:.4f} (Δ {metrics_delta.get('accuracy_delta', 0):.4f})"
                )
                print(
                    f"  F1-Score:  {metrics['f1']:.4f} (Δ {metrics_delta.get('f1_delta', 0):.4f})"
                )
                print(
                    f"  ROC-AUC:   {metrics['roc_auc']:.4f} (Δ {metrics_delta.get('roc_auc_delta', 0):.4f})"
                )
                print(
                    f"  Tree depth: {model.get_depth()}, Leaves: {model.get_n_leaves()}"
                )

                if mcnemar_result:
                    print(
                        f"  McNemar's test: p-value = {mcnemar_result['pvalue']:.4f}",
                        end="",
                    )
                    if mcnemar_result["significant"]:
                        print(" (SIGNIFICANT difference from baseline)")
                    else:
                        print(" (not significant)")

            return result

        except Exception as e:
            print(f"\n✗ ERROR in experiment '{experiment_name}': {str(e)}")
            traceback.print_exc()
            return {
                "experiment": experiment_name,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    def _compute_mcnemar_test(
        self,
        y_true: np.ndarray,
        y_pred_baseline: np.ndarray,
        y_pred_ablation: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Compute McNemar's test to compare two models' predictions.

        Tests null hypothesis: both models have same error rate
        """
        # Create contingency table
        # [baseline_correct, ablation_correct]
        baseline_correct = y_pred_baseline == y_true
        ablation_correct = y_pred_ablation == y_true

        # McNemar's table:
        #                  Ablation Correct | Ablation Wrong
        # Baseline Correct      n11         |      n12
        # Baseline Wrong        n21         |      n22

        n11 = np.sum(baseline_correct & ablation_correct)
        n12 = np.sum(baseline_correct & ~ablation_correct)
        n21 = np.sum(~baseline_correct & ablation_correct)
        n22 = np.sum(~baseline_correct & ~ablation_correct)

        # McNemar's test focuses on n12 and n21 (disagreements)
        contingency_table = np.array([[n11, n12], [n21, n22]])

        # Perform McNemar's test (use statsmodels if available, else custom implementation)
        if HAS_STATSMODELS:
            result = mcnemar(
                contingency_table, exact=True if (n12 + n21) < 25 else False
            )
        else:
            result = _custom_mcnemar_test(contingency_table, exact=True)

        return {
            "statistic": float(result.statistic),
            "pvalue": float(result.pvalue),
            "significant": result.pvalue < self.config.mcnemar_alpha,
            "alpha": self.config.mcnemar_alpha,
            "contingency_table": {
                "both_correct": int(n11),
                "baseline_correct_ablation_wrong": int(n12),
                "baseline_wrong_ablation_correct": int(n21),
                "both_wrong": int(n22),
            },
        }

    def run_ablation_experiments(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_names: List[str],
    ) -> Dict[str, Any]:
        """
        Run all ablation experiments (parallel or sequential based on config).
        """
        if self.baseline_params is None:
            raise ValueError("Must run baseline first to get hyperparameters")

        # Get experiment names to run
        if self.config.ablation_experiments:
            experiment_names = self.config.ablation_experiments
        else:
            # Run all experiments
            experiment_names = self.ablation_config.get_all_experiment_names(
                include_leave_one_out=True
            )
            # Remove baseline since we already ran it
            experiment_names = [e for e in experiment_names if e != "baseline"]

        if self.config.verbose > 0:
            print("\n" + "=" * 80)
            print(f"RUNNING {len(experiment_names)} ABLATION EXPERIMENTS")
            print("=" * 80)
            print(f"Parallel execution: {self.config.ablation_n_jobs} jobs")
            print(
                f"Timeout per experiment: {self.config.ablation_timeout_per_experiment}s"
            )
            print(f"Using baseline hyperparameters: {self.config.use_baseline_params}")
            print()

        if self.config.ablation_n_jobs == 1:
            # Sequential execution
            results = self._run_sequential(
                experiment_names, X_train, y_train, X_test, y_test, feature_names
            )
        else:
            # Parallel execution
            results = self._run_parallel(
                experiment_names, X_train, y_train, X_test, y_test, feature_names
            )

        self.results.update(results)
        return results

    def _run_sequential(
        self,
        experiment_names: List[str],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_names: List[str],
    ) -> Dict[str, Any]:
        """Run experiments sequentially"""
        results = {}

        for i, exp_name in enumerate(experiment_names, 1):
            if self.interrupted:
                break

            if self.config.verbose > 0:
                print(f"\nProgress: {i}/{len(experiment_names)}")

            result = self.run_single_ablation(
                exp_name,
                X_train,
                y_train,
                X_test,
                y_test,
                feature_names,
                self.baseline_params,
            )
            results[exp_name] = result

        return results

    def _run_parallel(
        self,
        experiment_names: List[str],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_names: List[str],
    ) -> Dict[str, Any]:
        """Run experiments in parallel with proper timeout and Ctrl+C handling"""
        results = {}
        n_jobs = self.config.ablation_n_jobs
        if n_jobs == -1:
            n_jobs = None  # Use all available cores

        # Partial function with fixed arguments
        run_fn = partial(
            self.run_single_ablation,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            all_feature_names=feature_names,
            hyperparameters=self.baseline_params,
        )

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Submit all jobs
            future_to_exp = {
                executor.submit(run_fn, exp_name): exp_name
                for exp_name in experiment_names
            }

            # Collect results as they complete
            completed = 0
            total = len(experiment_names)

            try:
                for future in as_completed(
                    future_to_exp, timeout=self.config.ablation_timeout_per_experiment
                ):
                    if self.interrupted:
                        # Cancel remaining futures
                        for f in future_to_exp:
                            f.cancel()
                        break

                    exp_name = future_to_exp[future]
                    completed += 1

                    try:
                        result = future.result(
                            timeout=self.config.ablation_timeout_per_experiment
                        )
                        results[exp_name] = result

                        if self.config.verbose > 0:
                            print(f"\n✓ Completed {completed}/{total}: {exp_name}")

                    except TimeoutError:
                        print(f"\n✗ Timeout for experiment: {exp_name}")
                        results[exp_name] = {
                            "experiment": exp_name,
                            "error": "Timeout exceeded",
                        }

                    except Exception as e:
                        print(f"\n✗ Error in experiment {exp_name}: {str(e)}")
                        results[exp_name] = {"experiment": exp_name, "error": str(e)}

            except KeyboardInterrupt:
                print("\n\nReceived Ctrl+C, shutting down parallel workers...")
                executor.shutdown(wait=False, cancel_futures=True)
                self.interrupted = True

        return results

    def compare_results(self) -> pd.DataFrame:
        """Create comparison table of all experiments"""
        comparison_data = []

        for exp_name, result in self.results.items():
            if "error" in result:
                continue

            row = {
                "experiment": exp_name,
                "n_features": result["n_features"],
                "accuracy": result["test_metrics"]["accuracy"],
                "precision": result["test_metrics"]["precision"],
                "recall": result["test_metrics"]["recall"],
                "f1": result["test_metrics"]["f1"],
                "roc_auc": result["test_metrics"]["roc_auc"],
                "tree_depth": result["tree_depth"],
                "tree_leaves": result["tree_leaves"],
            }

            # Add deltas if available
            if "metrics_delta" in result:
                row["f1_delta"] = result["metrics_delta"].get("f1_delta", 0)
                row["roc_auc_delta"] = result["metrics_delta"].get("roc_auc_delta", 0)

            # Add McNemar's test p-value if available
            if result.get("mcnemar_test"):
                row["mcnemar_p"] = result["mcnemar_test"]["pvalue"]
                row["significant"] = result["mcnemar_test"]["significant"]

            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        # Sort by F1 score descending
        if "f1" in df.columns:
            df = df.sort_values("f1", ascending=False)

        return df

    def save_results(self):
        """Save comprehensive ablation study results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self.config.ablation_output_dir)

        # Save full results as pickle
        results_pkl = output_dir / f"ablation_results_{timestamp}.pkl"
        with open(results_pkl, "wb") as f:
            pickle.dump(self.results, f)

        # Save comparison table as CSV
        comparison_df = self.compare_results()
        comparison_csv = output_dir / f"ablation_comparison_{timestamp}.csv"
        comparison_df.to_csv(comparison_csv, index=False)

        # Save detailed text report
        self._save_text_report(output_dir / f"ablation_report_{timestamp}.txt")

        # Save JSON summary
        self._save_json_summary(output_dir / f"ablation_summary_{timestamp}.json")

        if self.config.verbose > 0:
            print(f"\n{'=' * 80}")
            print("RESULTS SAVED")
            print(f"{'=' * 80}")
            print(f"✓ Full results: {results_pkl}")
            print(f"✓ Comparison table: {comparison_csv}")
            print(
                f"✓ Detailed report: {output_dir / f'ablation_report_{timestamp}.txt'}"
            )
            print(
                f"✓ JSON summary: {output_dir / f'ablation_summary_{timestamp}.json'}"
            )

    def _save_partial_results(self):
        """Save partial results when interrupted"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self.config.ablation_output_dir)

        results_pkl = output_dir / f"ablation_results_PARTIAL_{timestamp}.pkl"
        with open(results_pkl, "wb") as f:
            pickle.dump(self.results, f)

        print(f"✓ Partial results saved to: {results_pkl}")

    def _save_text_report(self, filepath: Path):
        """Save detailed text report"""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("ABLATION STUDY COMPREHENSIVE REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Configuration
            f.write("CONFIGURATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Data path: {self.config.data_path}\n")
            f.write(f"Test size: {self.config.test_size}\n")
            f.write(f"Random state: {self.config.random_state}\n")
            f.write(f"Baseline optimization: {self.config.use_baseline_params}\n")
            f.write(f"Parallel jobs: {self.config.ablation_n_jobs}\n")
            f.write(f"McNemar's test: {self.config.run_mcnemar_test}\n\n")

            # Baseline hyperparameters
            if self.baseline_params:
                f.write("BASELINE HYPERPARAMETERS\n")
                f.write("-" * 80 + "\n")
                for param, value in self.baseline_params.items():
                    f.write(f"{param}: {value}\n")
                f.write("\n")

            # Comparison table
            f.write("PERFORMANCE COMPARISON\n")
            f.write("-" * 80 + "\n")
            comparison_df = self.compare_results()
            f.write(comparison_df.to_string(index=False))
            f.write("\n\n")

            # Individual experiment details
            for exp_name in sorted(self.results.keys()):
                result = self.results[exp_name]
                if "error" in result:
                    f.write(f"\nEXPERIMENT: {exp_name} (FAILED)\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"Error: {result['error']}\n\n")
                    continue

                f.write(f"\nEXPERIMENT: {exp_name}\n")
                f.write("-" * 80 + "\n")
                f.write(f"Features: {result['n_features']}\n")
                f.write(f"Tree depth: {result['tree_depth']}\n")
                f.write(f"Tree leaves: {result['tree_leaves']}\n\n")

                f.write("Metrics:\n")
                for metric, value in result["test_metrics"].items():
                    f.write(f"  {metric}: {value:.4f}\n")

                if result.get("metrics_delta"):
                    f.write("\nDeltas from baseline:\n")
                    for metric, delta in result["metrics_delta"].items():
                        f.write(f"  {metric}: {delta:+.4f}\n")

                if result.get("mcnemar_test"):
                    mc = result["mcnemar_test"]
                    f.write(f"\nMcNemar's test:\n")
                    f.write(f"  p-value: {mc['pvalue']:.4f}\n")
                    f.write(f"  Significant: {mc['significant']}\n")

                f.write("\nTop 10 features by importance:\n")
                for feat in result["feature_importance"][:10]:
                    f.write(f"  {feat['feature']:50s} {feat['importance']:.6f}\n")

                f.write("\n")

    def _save_json_summary(self, filepath: Path):
        """Save JSON summary (without predictions to keep file size reasonable)"""
        summary = {}

        for exp_name, result in self.results.items():
            if "error" in result:
                summary[exp_name] = {"error": result["error"]}
                continue

            # Convert mcnemar_test to JSON-safe dict
            mcnemar_safe = None
            if result.get("mcnemar_test"):
                mc = result["mcnemar_test"]
                mcnemar_safe = {
                    "statistic": float(mc.get("statistic", 0)),
                    "pvalue": float(mc.get("pvalue", 1.0)),
                    "significant": bool(mc.get("significant", False)),
                    "alpha": float(mc.get("alpha", 0.05)),
                }
                if "contingency_table" in mc:
                    mcnemar_safe["contingency_table"] = {
                        k: int(v) for k, v in mc["contingency_table"].items()
                    }

            # Exclude predictions to reduce file size
            summary[exp_name] = {
                "n_features": int(result["n_features"]),
                "test_metrics": {
                    k: float(v) for k, v in result["test_metrics"].items()
                },
                "metrics_delta": {
                    k: float(v) for k, v in result.get("metrics_delta", {}).items()
                },
                "tree_depth": int(result["tree_depth"]),
                "tree_leaves": int(result["tree_leaves"]),
                "mcnemar_test": mcnemar_safe,
                "top_10_features": [
                    {"feature": str(f["feature"]), "importance": float(f["importance"])}
                    for f in result["feature_importance"][:10]
                ],
            }

        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2)


def run_full_ablation_study(config: ModelConfig) -> AblationStudy:
    """
    Main entry point for ablation study.

    Args:
        config: Model configuration with ablation settings

    Returns:
        AblationStudy instance with results
    """
    print("=" * 80)
    print("DISAGREEMENT MODEL - ABLATION STUDY")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Prepare data
    print("STEP 1: DATA PREPARATION")
    print("-" * 80)
    preparator = DataPreparator(config)
    X_train, y_train, X_test, y_test, feature_names = preparator.prepare_data()

    # Initialize ablation study
    print("\nSTEP 2: INITIALIZE ABLATION STUDY")
    print("-" * 80)
    ablation_config = AblationConfig()
    study = AblationStudy(config, ablation_config)

    if config.verbose > 1:
        ablation_config.print_summary()

    # Run baseline
    print("\nSTEP 3: RUN BASELINE MODEL")
    print("-" * 80)
    study.run_baseline(X_train, y_train, X_test, y_test, feature_names)

    # Run ablation experiments
    print("\nSTEP 4: RUN ABLATION EXPERIMENTS")
    print("-" * 80)
    study.run_ablation_experiments(X_train, y_train, X_test, y_test, feature_names)

    # Save results
    print("\nSTEP 5: SAVE RESULTS")
    print("-" * 80)
    study.save_results()

    # Print summary
    print("\n" + "=" * 80)
    print("ABLATION STUDY COMPLETE")
    print("=" * 80)
    comparison_df = study.compare_results()
    print("\nTop 5 Performing Experiments (by F1-score):")
    print(
        comparison_df.head(5)[["experiment", "n_features", "f1", "roc_auc"]].to_string(
            index=False
        )
    )
    print("\n" + "=" * 80)

    return study
