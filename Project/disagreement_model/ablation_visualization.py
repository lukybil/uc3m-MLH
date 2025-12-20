"""
Visualization tools for ablation study results.

Creates comparison plots: performance metrics, feature importance shifts,
performance deltas, and tree complexity comparisons.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Any
import pickle

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 10


class AblationVisualizer:
    """Create visualizations for ablation study results"""

    def __init__(self, results_dict: Dict[str, Any], output_dir: str):
        """
        Args:
            results_dict: Dictionary of experiment results from AblationStudy
            output_dir: Directory to save visualization files
        """
        self.results = results_dict
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Extract baseline for comparisons
        self.baseline = results_dict.get("baseline", {})

    def plot_performance_comparison(
        self,
        metrics: List[str] = ["f1", "roc_auc", "accuracy"],
        figsize: tuple = (14, 10),
        save: bool = True,
    ):
        """
        Create bar chart comparing performance metrics across experiments.

        Args:
            metrics: List of metrics to plot
            figsize: Figure size
            save: Whether to save the plot
        """
        # Prepare data
        data = []
        for exp_name, result in self.results.items():
            if "error" in result:
                continue

            row = {"experiment": exp_name}
            for metric in metrics:
                row[metric] = result["test_metrics"].get(metric, np.nan)
            data.append(row)

        df = pd.DataFrame(data)

        # Sort by first metric
        df = df.sort_values(metrics[0], ascending=False)

        # Create subplots
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=figsize)
        if n_metrics == 1:
            axes = [axes]

        # Plot each metric
        for i, metric in enumerate(metrics):
            ax = axes[i]

            # Create bar plot
            bars = ax.barh(df["experiment"], df[metric])

            # Color baseline differently
            for j, exp in enumerate(df["experiment"]):
                if exp == "baseline":
                    bars[j].set_color("darkgreen")
                    bars[j].set_alpha(0.8)
                else:
                    # Color by performance relative to baseline
                    if self.baseline and metric in self.baseline["test_metrics"]:
                        baseline_val = self.baseline["test_metrics"][metric]
                        if df[metric].iloc[j] >= baseline_val:
                            bars[j].set_color("steelblue")
                        else:
                            bars[j].set_color("coral")

            # Add value labels
            for j, (exp, val) in enumerate(zip(df["experiment"], df[metric])):
                ax.text(val + 0.005, j, f"{val:.3f}", va="center", fontsize=8)

            # Add baseline reference line
            if self.baseline and metric in self.baseline["test_metrics"]:
                baseline_val = self.baseline["test_metrics"][metric]
                ax.axvline(
                    baseline_val,
                    color="darkgreen",
                    linestyle="--",
                    alpha=0.7,
                    linewidth=1.5,
                    label="Baseline",
                )

            ax.set_xlabel(metric.upper())
            ax.set_title(f"{metric.upper()} Comparison Across Ablation Experiments")
            ax.legend()
            ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()

        if save:
            filepath = self.output_dir / "ablation_performance_comparison.png"
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"✓ Saved performance comparison to {filepath}")

        return fig

    def plot_performance_deltas(
        self,
        metrics: List[str] = ["f1", "roc_auc"],
        figsize: tuple = (12, 8),
        save: bool = True,
    ):
        """
        Create waterfall plot showing performance delta from baseline.

        Args:
            metrics: Metrics to plot
            figsize: Figure size
            save: Whether to save the plot
        """
        if not self.baseline:
            print("Warning: No baseline found, cannot plot deltas")
            return None

        # Prepare data
        data = []
        for exp_name, result in self.results.items():
            if exp_name == "baseline" or "error" in result:
                continue

            row = {"experiment": exp_name}
            for metric in metrics:
                current_val = result["test_metrics"].get(metric, np.nan)
                baseline_val = self.baseline["test_metrics"].get(metric, np.nan)
                row[f"{metric}_delta"] = current_val - baseline_val
            data.append(row)

        df = pd.DataFrame(data)

        # Sort by first metric delta
        df = df.sort_values(f"{metrics[0]}_delta", ascending=True)

        # Create plot
        fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
        if len(metrics) == 1:
            axes = [axes]

        for i, metric in enumerate(metrics):
            ax = axes[i]
            delta_col = f"{metric}_delta"

            # Color code: positive = blue, negative = red
            colors = ["steelblue" if x >= 0 else "coral" for x in df[delta_col]]

            bars = ax.barh(df["experiment"], df[delta_col], color=colors)

            # Add value labels
            for j, (exp, val) in enumerate(zip(df["experiment"], df[delta_col])):
                x_pos = val + (0.002 if val >= 0 else -0.002)
                ha = "left" if val >= 0 else "right"
                ax.text(x_pos, j, f"{val:+.3f}", va="center", ha=ha, fontsize=8)

            # Reference line at 0
            ax.axvline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)

            ax.set_xlabel(f"Δ {metric.upper()} (from baseline)")
            ax.set_title(f"{metric.upper()} Change from Baseline")
            ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()

        if save:
            filepath = self.output_dir / "ablation_performance_deltas.png"
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"✓ Saved performance deltas to {filepath}")

        return fig

    def plot_tree_complexity(self, figsize: tuple = (12, 6), save: bool = True):
        """
        Plot tree complexity metrics (depth and number of leaves).

        Args:
            figsize: Figure size
            save: Whether to save the plot
        """
        # Prepare data
        data = []
        for exp_name, result in self.results.items():
            if "error" in result:
                continue

            data.append(
                {
                    "experiment": exp_name,
                    "depth": result.get("tree_depth", np.nan),
                    "leaves": result.get("tree_leaves", np.nan),
                    "n_features": result.get("n_features", np.nan),
                }
            )

        df = pd.DataFrame(data)
        df = df.sort_values("depth", ascending=False)

        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot depth
        colors = [
            "darkgreen" if exp == "baseline" else "steelblue"
            for exp in df["experiment"]
        ]
        bars1 = ax1.barh(df["experiment"], df["depth"], color=colors)

        for j, (exp, val) in enumerate(zip(df["experiment"], df["depth"])):
            ax1.text(val + 0.2, j, f"{int(val)}", va="center", fontsize=8)

        ax1.set_xlabel("Tree Depth")
        ax1.set_title("Decision Tree Depth by Experiment")
        ax1.grid(axis="x", alpha=0.3)

        # Plot leaves
        bars2 = ax2.barh(df["experiment"], df["leaves"], color=colors)

        for j, (exp, val) in enumerate(zip(df["experiment"], df["leaves"])):
            ax2.text(val + 2, j, f"{int(val)}", va="center", fontsize=8)

        ax2.set_xlabel("Number of Leaves")
        ax2.set_title("Decision Tree Leaves by Experiment")
        ax2.grid(axis="x", alpha=0.3)

        plt.tight_layout()

        if save:
            filepath = self.output_dir / "ablation_tree_complexity.png"
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"✓ Saved tree complexity plot to {filepath}")

        return fig

    def plot_feature_importance_heatmap(
        self,
        top_n: int = 20,
        experiments: Optional[List[str]] = None,
        figsize: tuple = (14, 10),
        save: bool = True,
    ):
        """
        Create heatmap showing feature importance shifts across experiments.

        Args:
            top_n: Number of top features to show
            experiments: List of experiments to include (None = all)
            figsize: Figure size
            save: Whether to save the plot
        """
        # Get top features from baseline
        if not self.baseline or "feature_importance" not in self.baseline:
            print("Warning: No baseline feature importance found")
            return None

        baseline_fi = pd.DataFrame(self.baseline["feature_importance"])
        top_features = baseline_fi.head(top_n)["feature"].tolist()

        # Build importance matrix
        importance_matrix = []
        exp_names = []

        for exp_name, result in self.results.items():
            if "error" in result or "feature_importance" not in result:
                continue

            if experiments and exp_name not in experiments:
                continue

            fi_df = pd.DataFrame(result["feature_importance"])
            fi_dict = dict(zip(fi_df["feature"], fi_df["importance"]))

            # Get importance for top features (0 if not present)
            importance_row = [fi_dict.get(feat, 0) for feat in top_features]
            importance_matrix.append(importance_row)
            exp_names.append(exp_name)

        # Create dataframe
        importance_df = pd.DataFrame(
            importance_matrix, index=exp_names, columns=top_features
        )

        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            importance_df.T,  # Transpose so features are rows
            annot=False,
            cmap="YlOrRd",
            cbar_kws={"label": "Feature Importance"},
            ax=ax,
        )

        ax.set_title(f"Top {top_n} Feature Importance Across Ablation Experiments")
        ax.set_xlabel("Experiment")
        ax.set_ylabel("Feature")

        plt.tight_layout()

        if save:
            filepath = self.output_dir / "ablation_feature_importance_heatmap.png"
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"✓ Saved feature importance heatmap to {filepath}")

        return fig

    def plot_mcnemar_significance(
        self, alpha: float = 0.05, figsize: tuple = (10, 8), save: bool = True
    ):
        """
        Plot McNemar's test p-values to show which experiments differ significantly.

        Args:
            alpha: Significance threshold
            figsize: Figure size
            save: Whether to save the plot
        """
        # Collect McNemar's test results
        data = []
        for exp_name, result in self.results.items():
            if exp_name == "baseline" or "error" in result:
                continue

            if "mcnemar_test" in result and result["mcnemar_test"]:
                mc = result["mcnemar_test"]
                data.append(
                    {
                        "experiment": exp_name,
                        "p_value": mc["pvalue"],
                        "significant": mc["significant"],
                    }
                )

        if not data:
            print("Warning: No McNemar's test results found")
            return None

        df = pd.DataFrame(data)
        df = df.sort_values("p_value", ascending=True)

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        # Color code: significant vs not significant
        colors = ["coral" if sig else "steelblue" for sig in df["significant"]]

        bars = ax.barh(df["experiment"], df["p_value"], color=colors)

        # Add p-value labels
        for j, (exp, val) in enumerate(zip(df["experiment"], df["p_value"])):
            ax.text(val + 0.005, j, f"{val:.4f}", va="center", fontsize=8)

        # Add significance threshold line
        ax.axvline(
            alpha,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"α = {alpha}",
            alpha=0.7,
        )

        ax.set_xlabel("McNemar's Test p-value")
        ax.set_title("Statistical Significance of Ablation Experiments vs Baseline")
        ax.set_xlim(0, max(df["p_value"].max() * 1.1, alpha * 2))
        ax.legend()
        ax.grid(axis="x", alpha=0.3)

        # Add text annotation
        n_significant = df["significant"].sum()
        ax.text(
            0.95,
            0.95,
            f"{n_significant}/{len(df)} experiments\nsignificantly different\nfrom baseline",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            fontsize=10,
        )

        plt.tight_layout()

        if save:
            filepath = self.output_dir / "ablation_mcnemar_significance.png"
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"✓ Saved McNemar's significance plot to {filepath}")

        return fig

    def create_all_plots(self):
        """Generate all visualization plots"""
        print("\n" + "=" * 80)
        print("GENERATING ABLATION STUDY VISUALIZATIONS")
        print("=" * 80 + "\n")

        self.plot_performance_comparison()
        self.plot_performance_deltas()
        self.plot_tree_complexity()
        self.plot_feature_importance_heatmap()

        # Only create McNemar plot if data exists
        has_mcnemar = any(
            "mcnemar_test" in result and result["mcnemar_test"]
            for result in self.results.values()
            if "error" not in result
        )
        if has_mcnemar:
            self.plot_mcnemar_significance()

        print("\n" + "=" * 80)
        print("✓ All visualizations complete")
        print("=" * 80 + "\n")


def visualize_ablation_results(results_pkl_path: str, output_dir: Optional[str] = None):
    """
    Load ablation results from pickle file and create visualizations.

    Args:
        results_pkl_path: Path to ablation results pickle file
        output_dir: Output directory for plots (default: same as results file)
    """
    # Load results
    with open(results_pkl_path, "rb") as f:
        results = pickle.load(f)

    # Set output directory
    if output_dir is None:
        output_dir = Path(results_pkl_path).parent / "visualizations"

    # Create visualizer
    visualizer = AblationVisualizer(results, output_dir)
    visualizer.create_all_plots()

    return visualizer


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ablation_visualization.py <results_pkl_path> [output_dir]")
        sys.exit(1)

    results_path = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else None

    visualize_ablation_results(results_path, output)
