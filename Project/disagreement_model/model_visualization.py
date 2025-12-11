"""
Visualization utilities for decision tree model.
Creates tree plots, feature importance charts, and interactive visualizations.
"""

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for server environments
import seaborn as sns
from sklearn.tree import plot_tree, export_text
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
import sys

from .model_config import ModelConfig


class ModelVisualizer:
    """Handles all visualization tasks for the decision tree model"""

    def __init__(self, model, feature_names: List[str], config: ModelConfig):
        self.model = model
        self.feature_names = feature_names
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams["figure.dpi"] = config.viz_dpi
        plt.rcParams["savefig.dpi"] = config.viz_dpi
        plt.rcParams["font.size"] = 10

    def plot_tree_structure(
        self, filename: Optional[str] = None, max_depth: Optional[int] = None
    ):
        """
        Plot the decision tree structure.

        Args:
            filename: Output filename (without extension)
            max_depth: Maximum depth to display (None for full tree)
        """
        if filename is None:
            filename = f"{self.config.results_prefix}_tree_structure"

        if max_depth is None:
            max_depth = self.config.viz_max_depth

        if self.config.verbose > 0:
            print(f"Generating tree structure plot...")

        # Create figure
        fig, ax = plt.subplots(figsize=self.config.viz_figsize)

        # Plot tree
        plot_tree(
            self.model,
            feature_names=self.feature_names,
            class_names=["Agreement", "Disagreement"],
            filled=True,
            rounded=True,
            fontsize=8,
            max_depth=max_depth,
            ax=ax,
        )

        plt.title(
            f"Decision Tree for Disagreement Prediction\n(Max depth shown: {max_depth})",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        plt.tight_layout()

        # Save as PNG
        output_path = self.output_dir / f"{filename}.png"
        plt.savefig(output_path, bbox_inches="tight", dpi=self.config.viz_dpi)
        plt.close()

        if self.config.verbose > 0:
            print(f"  ✓ Saved to {output_path}")

        # Also save as PDF for better quality
        output_path_pdf = self.output_dir / f"{filename}.pdf"
        fig, ax = plt.subplots(figsize=self.config.viz_figsize)
        plot_tree(
            self.model,
            feature_names=self.feature_names,
            class_names=["Agreement", "Disagreement"],
            filled=True,
            rounded=True,
            fontsize=8,
            max_depth=max_depth,
            ax=ax,
        )
        plt.title(
            f"Decision Tree for Disagreement Prediction\n(Max depth shown: {max_depth})",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        plt.tight_layout()
        plt.savefig(output_path_pdf, bbox_inches="tight", format="pdf")
        plt.close()

        if self.config.verbose > 0:
            print(f"  ✓ Saved to {output_path_pdf}")

    def plot_feature_importance(
        self, filename: Optional[str] = None, top_n: Optional[int] = None
    ):
        """
        Plot feature importance bar chart.

        Args:
            filename: Output filename (without extension)
            top_n: Number of top features to display
        """
        if filename is None:
            filename = f"{self.config.results_prefix}_feature_importance"

        if top_n is None:
            top_n = self.config.viz_feature_importance_top_n

        if self.config.verbose > 0:
            print(f"Generating feature importance plot (top {top_n})...")

        # Get feature importances
        importances = self.model.feature_importances_
        importance_df = pd.DataFrame(
            {"feature": self.feature_names, "importance": importances}
        ).sort_values("importance", ascending=False)

        # Select top N
        top_features = importance_df.head(top_n)

        # Create plot
        fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.4)))

        # Horizontal bar plot
        y_pos = np.arange(len(top_features))
        bars = ax.barh(
            y_pos, top_features["importance"].values, color="steelblue", alpha=0.8
        )

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, top_features["importance"].values)):
            ax.text(val, i, f" {val:.4f}", va="center", fontsize=9)

        # Formatting
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features["feature"].values, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("Importance (Gini Decrease)", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Top {top_n} Most Important Features\nfor Disagreement Prediction",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()

        # Save
        output_path = self.output_dir / f"{filename}.png"
        plt.savefig(output_path, bbox_inches="tight", dpi=self.config.viz_dpi)
        plt.close()

        if self.config.verbose > 0:
            print(f"  ✓ Saved to {output_path}")

    def plot_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray, filename: Optional[str] = None
    ):
        """
        Plot confusion matrix heatmap.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            filename: Output filename (without extension)
        """
        if filename is None:
            filename = f"{self.config.results_prefix}_confusion_matrix"

        if self.config.verbose > 0:
            print(f"Generating confusion matrix plot...")

        from sklearn.metrics import confusion_matrix

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Agreement", "Disagreement"],
            yticklabels=["Agreement", "Disagreement"],
            cbar_kws={"label": "Count"},
            ax=ax,
            square=True,
            linewidths=1,
            linecolor="black",
            annot_kws={"fontsize": 14, "fontweight": "bold"},
        )

        ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
        ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
        ax.set_title(
            "Confusion Matrix\nDisagreement Prediction",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        plt.tight_layout()

        # Save
        output_path = self.output_dir / f"{filename}.png"
        plt.savefig(output_path, bbox_inches="tight", dpi=self.config.viz_dpi)
        plt.close()

        if self.config.verbose > 0:
            print(f"  ✓ Saved to {output_path}")

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        filename: Optional[str] = None,
    ):
        """
        Plot ROC curve.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities for positive class
            filename: Output filename (without extension)
        """
        if filename is None:
            filename = f"{self.config.results_prefix}_roc_curve"

        if self.config.verbose > 0:
            print(f"Generating ROC curve plot...")

        from sklearn.metrics import roc_curve, auc

        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        # Create plot
        fig, ax = plt.subplots(figsize=(8, 8))

        # ROC curve
        ax.plot(
            fpr, tpr, color="darkblue", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})"
        )

        # Diagonal reference line
        ax.plot(
            [0, 1],
            [0, 1],
            color="gray",
            lw=1,
            linestyle="--",
            label="Random classifier",
        )

        # Formatting
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
        ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
        ax.set_title(
            "ROC Curve - Disagreement Prediction",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_aspect("equal")

        plt.tight_layout()

        # Save
        output_path = self.output_dir / f"{filename}.png"
        plt.savefig(output_path, bbox_inches="tight", dpi=self.config.viz_dpi)
        plt.close()

        if self.config.verbose > 0:
            print(f"  ✓ Saved to {output_path}")

    def export_text_representation(
        self, filename: Optional[str] = None, max_depth: Optional[int] = None
    ):
        """
        Export text representation of the tree.

        Args:
            filename: Output filename (without extension)
            max_depth: Maximum depth to display
        """
        if filename is None:
            filename = f"{self.config.results_prefix}_tree_text"

        if max_depth is None:
            max_depth = self.config.viz_max_depth

        if self.config.verbose > 0:
            print(f"Exporting text tree representation...")

        # Generate text representation
        tree_text = export_text(
            self.model,
            feature_names=self.feature_names,
            max_depth=max_depth,
            show_weights=True,
        )

        # Save to file
        output_path = self.output_dir / f"{filename}.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("DECISION TREE TEXT REPRESENTATION\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Max depth shown: {max_depth}\n")
            f.write(f"Total tree depth: {self.model.get_depth()}\n")
            f.write(f"Total leaves: {self.model.get_n_leaves()}\n\n")
            f.write("=" * 80 + "\n\n")
            f.write(tree_text)

        if self.config.verbose > 0:
            print(f"  ✓ Saved to {output_path}")

    def create_interactive_tree(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Create interactive HTML tree visualization using dtreeviz.

        Args:
            X_train: Training features
            y_train: Training labels
        """
        if not self.config.generate_interactive_tree:
            return

        try:
            import dtreeviz

            if self.config.verbose > 0:
                print(f"Generating interactive tree visualization...")

            # Create visualization
            viz = dtreeviz.model(
                self.model,
                X_train=X_train,
                y_train=y_train,
                feature_names=self.feature_names,
                target_name="disagreement",
                class_names=["Agreement", "Disagreement"],
            )

            # Save as SVG and HTML
            output_path_svg = (
                self.output_dir / f"{self.config.results_prefix}_interactive_tree.svg"
            )
            viz.save(str(output_path_svg))

            if self.config.verbose > 0:
                print(f"  ✓ Saved to {output_path_svg}")

        except ImportError:
            if self.config.verbose > 0:
                print(
                    "  ⚠ dtreeviz not installed. Skipping interactive tree visualization."
                )
                print("    Install with: pip install dtreeviz")
        except Exception as e:
            if self.config.verbose > 0:
                print(f"  ⚠ Error creating interactive tree: {e}")

    def generate_all_visualizations(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ):
        """Generate all visualizations"""
        if self.config.verbose > 0:
            print("\n" + "=" * 80)
            print("GENERATING VISUALIZATIONS")
            print("=" * 80 + "\n")

        # Tree structure
        self.plot_tree_structure()

        # Feature importance
        if self.config.compute_feature_importance:
            self.plot_feature_importance()

        # Confusion matrix
        if self.config.compute_confusion_matrix:
            y_pred = self.model.predict(X_test)
            self.plot_confusion_matrix(y_test, y_pred)

        # ROC curve
        if self.config.compute_roc_curve:
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            self.plot_roc_curve(y_test, y_pred_proba)

        # Text representation
        self.export_text_representation()

        # Interactive tree (optional)
        self.create_interactive_tree(X_train, y_train)

        if self.config.verbose > 0:
            print("\n" + "=" * 80)
            print("VISUALIZATION COMPLETE")
            print("=" * 80 + "\n")


def visualize_from_saved_model(
    model_path: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    """
    Load a saved model and generate visualizations.

    Args:
        model_path: Path to saved model pickle file
        X_train, y_train: Training data
        X_test, y_test: Test data
    """
    import pickle

    # Load model
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    model = model_data["model"]
    feature_names = model_data["feature_names"]

    # Recreate config from saved data
    from model_config import ModelConfig

    config = ModelConfig()

    # Create visualizer
    visualizer = ModelVisualizer(model, feature_names, config)

    # Generate all visualizations
    visualizer.generate_all_visualizations(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    print("This module provides visualization utilities.")
    print("Import and use ModelVisualizer class or run disagreement_model.py")
