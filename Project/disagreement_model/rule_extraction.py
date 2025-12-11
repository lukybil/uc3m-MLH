"""
Extract and export decision rules from trained decision tree.
Provides human-readable rules in multiple formats (TXT, CSV, LaTeX, JSON).
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import sys

from .model_config import ModelConfig


class RuleExtractor:
    """Extract and format decision rules from decision tree"""

    def __init__(
        self,
        model: DecisionTreeClassifier,
        feature_names: List[str],
        config: ModelConfig,
    ):
        self.model = model
        self.feature_names = feature_names
        self.config = config
        self.rules: List[Dict[str, Any]] = []
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_rules(self) -> List[Dict[str, Any]]:
        """
        Extract all decision rules from the tree.

        Returns:
            List of rule dictionaries containing conditions, prediction, support, and confidence
        """
        if self.config.verbose > 0:
            print("Extracting decision rules from tree...")

        tree = self.model.tree_
        rules = []

        def recurse(node: int, conditions: List[str], depth: int = 0):
            """Recursively traverse tree and extract rules"""
            # Check if leaf node (more reliable than checking feature == -2)
            if tree.children_left[node] == tree.children_right[node]:  # Leaf node
                # Get prediction
                values = tree.value[node][0]
                predicted_class = np.argmax(values)

                # Use actual sample count for support, not weighted sum
                n_samples = tree.n_node_samples[node]

                # Use weighted values for confidence
                weighted_total = np.sum(values)
                class_samples = values[predicted_class]
                confidence = class_samples / weighted_total if weighted_total > 0 else 0

                # Only include rules with sufficient support
                if n_samples >= self.config.min_rule_support:
                    rule = {
                        "conditions": conditions.copy(),
                        "prediction": (
                            "Disagreement" if predicted_class == 1 else "Agreement"
                        ),
                        "prediction_class": int(predicted_class),
                        "support": int(n_samples),
                        "confidence": float(confidence),
                        "samples_per_class": {
                            "agreement": float(values[0]),
                            "disagreement": float(values[1]),
                        },
                        "depth": depth,
                    }
                    rules.append(rule)
            else:
                # Internal node - get split condition
                feature_name = self.feature_names[tree.feature[node]]
                threshold = tree.threshold[node]

                # Left child (<=)
                left_condition = f"{feature_name} <= {threshold:.4f}"
                recurse(
                    tree.children_left[node], conditions + [left_condition], depth + 1
                )

                # Right child (>)
                right_condition = f"{feature_name} > {threshold:.4f}"
                recurse(
                    tree.children_right[node], conditions + [right_condition], depth + 1
                )

        # Start recursion from root
        recurse(0, [])

        self.rules = rules

        if self.config.verbose > 0:
            print(
                f"  ✓ Extracted {len(rules)} rules with support >= {self.config.min_rule_support}"
            )

        return rules

    def format_rule_text(self, rule: Dict[str, Any], rule_number: int) -> str:
        """Format a single rule as human-readable text"""
        lines = []
        lines.append(f"Rule #{rule_number}")
        lines.append("-" * 60)
        lines.append("IF:")

        for i, condition in enumerate(rule["conditions"], 1):
            lines.append(f"  {i}. {condition}")

        lines.append(f"\nTHEN:")
        lines.append(f"  Prediction: {rule['prediction']}")
        lines.append(f"  Confidence: {rule['confidence']*100:.2f}%")
        lines.append(f"  Support: {rule['support']} samples")
        lines.append(
            f"  (Agreement: {rule['samples_per_class']['agreement']}, "
            f"Disagreement: {rule['samples_per_class']['disagreement']})"
        )
        lines.append(f"  Depth: {rule['depth']}")
        lines.append("")

        return "\n".join(lines)

    def export_rules_text(self, filename: Optional[str] = None):
        """Export rules as formatted text file"""
        if filename is None:
            filename = f"{self.config.results_prefix}_rules.txt"

        output_path = self.output_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("DECISION RULES FOR DISAGREEMENT PREDICTION\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Total rules extracted: {len(self.rules)}\n")
            f.write(
                f"Minimum support threshold: {self.config.min_rule_support} samples\n\n"
            )

            # Sort by confidence and support
            sorted_rules = sorted(
                self.rules,
                key=lambda x: (x["prediction_class"], -x["confidence"], -x["support"]),
            )

            # Group by prediction
            f.write("=" * 80 + "\n")
            f.write("RULES PREDICTING AGREEMENT\n")
            f.write("=" * 80 + "\n\n")

            agreement_rules = [
                r for r in sorted_rules if r["prediction"] == "Agreement"
            ]
            for i, rule in enumerate(agreement_rules, 1):
                f.write(self.format_rule_text(rule, i))
                f.write("\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("RULES PREDICTING DISAGREEMENT\n")
            f.write("=" * 80 + "\n\n")

            disagreement_rules = [
                r for r in sorted_rules if r["prediction"] == "Disagreement"
            ]
            for i, rule in enumerate(disagreement_rules, 1):
                f.write(self.format_rule_text(rule, i))
                f.write("\n")

        if self.config.verbose > 0:
            print(f"  ✓ Saved text rules to {output_path}")

    def export_rules_csv(self, filename: Optional[str] = None):
        """Export rules as CSV file"""
        if filename is None:
            filename = f"{self.config.results_prefix}_rules.csv"

        output_path = self.output_dir / filename

        # Convert rules to DataFrame
        rows = []
        for i, rule in enumerate(self.rules, 1):
            row = {
                "rule_id": i,
                "conditions": " AND ".join(rule["conditions"]),
                "num_conditions": len(rule["conditions"]),
                "prediction": rule["prediction"],
                "confidence": rule["confidence"],
                "support": rule["support"],
                "agreement_samples": rule["samples_per_class"]["agreement"],
                "disagreement_samples": rule["samples_per_class"]["disagreement"],
                "depth": rule["depth"],
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)

        if self.config.verbose > 0:
            print(f"  ✓ Saved CSV rules to {output_path}")

    def export_rules_latex(self, filename: Optional[str] = None):
        """Export rules as LaTeX formatted file"""
        if filename is None:
            filename = f"{self.config.results_prefix}_rules.tex"

        output_path = self.output_dir / filename

        # Sort rules by prediction and confidence
        sorted_rules = sorted(
            self.rules,
            key=lambda x: (x["prediction_class"], -x["confidence"], -x["support"]),
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("% Decision Rules for Disagreement Prediction\n")
            f.write("% Auto-generated LaTeX file\n\n")

            f.write("\\section{Decision Rules}\n\n")

            # Agreement rules
            f.write("\\subsection{Rules Predicting Agreement}\n\n")
            agreement_rules = [
                r for r in sorted_rules if r["prediction"] == "Agreement"
            ]

            for i, rule in enumerate(agreement_rules, 1):
                f.write("\\begin{itemize}\n")
                f.write(f"  \\item \\textbf{{Rule {i}:}}\n")
                f.write("  \\begin{enumerate}\n")
                for condition in rule["conditions"]:
                    # Escape special LaTeX characters
                    escaped_condition = condition.replace("_", "\\_").replace(
                        "%", "\\%"
                    )
                    f.write(f"    \\item {escaped_condition}\n")
                f.write("  \\end{enumerate}\n")
                f.write(f"  \\textbf{{Prediction:}} {rule['prediction']} \\\\\n")
                f.write(
                    f"  \\textbf{{Confidence:}} {rule['confidence']*100:.2f}\\% \\\\\n"
                )
                f.write(f"  \\textbf{{Support:}} {rule['support']} samples\n")
                f.write("\\end{itemize}\n\n")

            # Disagreement rules
            f.write("\\subsection{Rules Predicting Disagreement}\n\n")
            disagreement_rules = [
                r for r in sorted_rules if r["prediction"] == "Disagreement"
            ]

            for i, rule in enumerate(disagreement_rules, 1):
                f.write("\\begin{itemize}\n")
                f.write(f"  \\item \\textbf{{Rule {i}:}}\n")
                f.write("  \\begin{enumerate}\n")
                for condition in rule["conditions"]:
                    escaped_condition = condition.replace("_", "\\_").replace(
                        "%", "\\%"
                    )
                    f.write(f"    \\item {escaped_condition}\n")
                f.write("  \\end{enumerate}\n")
                f.write(f"  \\textbf{{Prediction:}} {rule['prediction']} \\\\\n")
                f.write(
                    f"  \\textbf{{Confidence:}} {rule['confidence']*100:.2f}\\% \\\\\n"
                )
                f.write(f"  \\textbf{{Support:}} {rule['support']} samples\n")
                f.write("\\end{itemize}\n\n")

        if self.config.verbose > 0:
            print(f"  ✓ Saved LaTeX rules to {output_path}")

    def export_rules_json(self, filename: Optional[str] = None):
        """Export rules as JSON file"""
        if filename is None:
            filename = f"{self.config.results_prefix}_rules.json"

        output_path = self.output_dir / filename

        # Convert numpy types to native Python types for JSON serialization
        rules_json = []
        for rule in self.rules:
            rule_copy = rule.copy()
            # Ensure all values are JSON serializable
            rule_copy["prediction_class"] = int(rule_copy["prediction_class"])
            rule_copy["support"] = int(rule_copy["support"])
            rule_copy["confidence"] = float(rule_copy["confidence"])
            rule_copy["depth"] = int(rule_copy["depth"])
            rule_copy["samples_per_class"]["agreement"] = int(
                rule_copy["samples_per_class"]["agreement"]
            )
            rule_copy["samples_per_class"]["disagreement"] = int(
                rule_copy["samples_per_class"]["disagreement"]
            )
            rules_json.append(rule_copy)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_rules": len(self.rules),
                    "min_support": self.config.min_rule_support,
                    "rules": rules_json,
                },
                f,
                indent=2,
            )

        if self.config.verbose > 0:
            print(f"  ✓ Saved JSON rules to {output_path}")

    def get_top_rules(
        self, n: int = 10, by: str = "confidence", prediction: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get top N rules sorted by specified metric.

        Args:
            n: Number of rules to return
            by: Metric to sort by ('confidence', 'support', 'depth')
            prediction: Filter by prediction ('Agreement', 'Disagreement', or None for all)

        Returns:
            List of top rules
        """
        filtered_rules = self.rules

        if prediction:
            filtered_rules = [
                r for r in filtered_rules if r["prediction"] == prediction
            ]

        if by == "confidence":
            sorted_rules = sorted(filtered_rules, key=lambda x: -x["confidence"])
        elif by == "support":
            sorted_rules = sorted(filtered_rules, key=lambda x: -x["support"])
        elif by == "depth":
            sorted_rules = sorted(filtered_rules, key=lambda x: x["depth"])
        else:
            sorted_rules = filtered_rules

        return sorted_rules[:n]

    def analyze_rule_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in the extracted rules"""
        if self.config.verbose > 0:
            print("Analyzing rule patterns...")

        # Feature frequency in rules
        feature_counts = {}
        for rule in self.rules:
            for condition in rule["conditions"]:
                feature = condition.split(" ")[0]
                feature_counts[feature] = feature_counts.get(feature, 0) + 1

        # Sort by frequency
        sorted_features = sorted(feature_counts.items(), key=lambda x: -x[1])

        # Rule complexity
        rule_lengths = [len(r["conditions"]) for r in self.rules]

        # Prediction distribution
        prediction_dist = {
            "Agreement": len([r for r in self.rules if r["prediction"] == "Agreement"]),
            "Disagreement": len(
                [r for r in self.rules if r["prediction"] == "Disagreement"]
            ),
        }

        # Average confidence by prediction
        avg_confidence = {}
        for pred in ["Agreement", "Disagreement"]:
            pred_rules = [r for r in self.rules if r["prediction"] == pred]
            if pred_rules:
                avg_confidence[pred] = np.mean([r["confidence"] for r in pred_rules])
            else:
                avg_confidence[pred] = 0.0

        # Handle empty rules case
        if len(rule_lengths) == 0:
            rule_complexity = {
                "mean_conditions": 0.0,
                "median_conditions": 0.0,
                "min_conditions": 0,
                "max_conditions": 0,
            }
        else:
            rule_complexity = {
                "mean_conditions": float(np.mean(rule_lengths)),
                "median_conditions": float(np.median(rule_lengths)),
                "min_conditions": int(np.min(rule_lengths)),
                "max_conditions": int(np.max(rule_lengths)),
            }

        analysis = {
            "total_rules": len(self.rules),
            "feature_frequency": dict(sorted_features[:20]),  # Top 20
            "rule_complexity": rule_complexity,
            "prediction_distribution": prediction_dist,
            "average_confidence": avg_confidence,
        }

        if self.config.verbose > 0:
            print(f"  ✓ Analysis complete")

        return analysis

    def export_all(self):
        """Export rules in all configured formats"""
        if self.config.verbose > 0:
            print("\n" + "=" * 80)
            print("EXPORTING DECISION RULES")
            print("=" * 80 + "\n")

        # Extract rules if not already done
        if not self.rules:
            self.extract_rules()

        # Export in all formats
        if self.config.export_rules_txt:
            self.export_rules_text()

        if self.config.export_rules_csv:
            self.export_rules_csv()

        if self.config.export_rules_latex:
            self.export_rules_latex()

        if self.config.export_rules_json:
            self.export_rules_json()

        # Analyze patterns
        analysis = self.analyze_rule_patterns()

        # Save analysis
        analysis_path = (
            self.output_dir / f"{self.config.results_prefix}_rule_analysis.json"
        )
        with open(analysis_path, "w") as f:
            json.dump(analysis, f, indent=2)

        if self.config.verbose > 0:
            print(f"  ✓ Saved rule analysis to {analysis_path}")
            print("\n" + "=" * 80)
            print("RULE EXPORT COMPLETE")
            print("=" * 80)
            print(f"Total rules: {len(self.rules)}")
            print(
                f"Agreement rules: {analysis['prediction_distribution']['Agreement']}"
            )
            print(
                f"Disagreement rules: {analysis['prediction_distribution']['Disagreement']}"
            )
            print(
                f"Average rule complexity: {analysis['rule_complexity']['mean_conditions']:.2f} conditions"
            )
            print("=" * 80 + "\n")


def extract_rules_from_saved_model(model_path: str):
    """
    Load a saved model and extract rules.

    Args:
        model_path: Path to saved model pickle file
    """
    import pickle

    # Load model
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    model = model_data["model"]
    feature_names = model_data["feature_names"]

    # Recreate config
    from model_config import ModelConfig

    config = ModelConfig()

    # Extract rules
    extractor = RuleExtractor(model, feature_names, config)
    extractor.extract_rules()
    extractor.export_all()

    return extractor


if __name__ == "__main__":
    print("This module provides rule extraction utilities.")
    print("Import and use RuleExtractor class or run disagreement_model.py")
