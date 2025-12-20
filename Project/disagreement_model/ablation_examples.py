"""
Example script demonstrating ablation study usage.

This script shows different ways to run ablation studies:
1. Quick start with defaults
2. Custom experiment selection
3. Parallel execution
4. Programmatic access to results
"""

from disagreement_model import (
    ModelConfig,
    AblationConfig,
    run_full_ablation_study,
    visualize_ablation_results,
)
from pathlib import Path


def example_1_quick_start():
    """Example 1: Quick start with default settings"""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Quick Start - Run All Experiments")
    print("=" * 80 + "\n")

    # Create configuration
    config = ModelConfig()
    config.run_ablation_study = True
    config.verbose = 1

    # Run ablation study (this will run all experiments)
    study = run_full_ablation_study(config)

    # Print summary
    print("\nResults summary:")
    print(
        study.compare_results()[["experiment", "n_features", "f1", "roc_auc"]].head(10)
    )

    return study


def example_2_custom_experiments():
    """Example 2: Run only specific experiments"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Custom Experiments - Strategic Ablations Only")
    print("=" * 80 + "\n")

    # Create configuration
    config = ModelConfig()
    config.run_ablation_study = True
    config.verbose = 1

    # Select specific experiments
    config.ablation_experiments = [
        "baseline",
        "clinical_only",
        "no_administrative",
        "core_clinical",
        "scores_only",
    ]

    # Run
    study = run_full_ablation_study(config)

    # Access specific results
    print("\nClinical-only experiment results:")
    if "clinical_only" in study.results:
        result = study.results["clinical_only"]
        print(f"  Features: {result['n_features']}")
        print(f"  F1-Score: {result['test_metrics']['f1']:.4f}")
        print(f"  ROC-AUC: {result['test_metrics']['roc_auc']:.4f}")

        if result.get("mcnemar_test"):
            mc = result["mcnemar_test"]
            print(f"  McNemar's p-value: {mc['p_value']:.4f}")
            print(f"  Significant difference: {mc['significant']}")

    return study


def example_3_parallel_execution():
    """Example 3: Parallel execution for faster results"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Parallel Execution - Leave-One-Out Experiments")
    print("=" * 80 + "\n")

    # Create configuration
    config = ModelConfig()
    config.run_ablation_study = True
    config.verbose = 1

    # Enable parallel execution
    config.ablation_n_jobs = 4  # Use 4 cores

    # Run leave-one-out experiments for core clinical groups
    ablation_config = AblationConfig()
    leave_one_out = [
        f"without_{group}"
        for group in [
            "depression",
            "anxiety",
            "suicide_risk",
            "substance_use",
            "psychosis",
        ]
    ]
    config.ablation_experiments = ["baseline"] + leave_one_out

    # Run
    study = run_full_ablation_study(config)

    # Compare results
    comparison = study.compare_results()
    comparison = comparison.sort_values("f1_delta", ascending=True)

    print("\nFeature groups ranked by impact (worst removal first):")
    print(
        comparison[["experiment", "n_features", "f1_delta", "significant"]].to_string(
            index=False
        )
    )

    return study


def example_4_custom_feature_groups():
    """Example 4: Add custom data-driven feature groups"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Custom Feature Groups - Data-Driven Analysis")
    print("=" * 80 + "\n")

    from disagreement_model.ablation_config import FeatureGroup

    # Create custom ablation config
    ablation_config = AblationConfig()

    # Add custom data-driven group (example: top 10 features from baseline)
    top_features_group = FeatureGroup(
        name="top_10_features",
        description="Top 10 most important features from baseline model",
        category="data_driven",
        features=[
            "phq9_severity_group_label",
            "service",
            "gad7_severity_group_label",
            "cssrs_total_score_1",
            "evaluating_user",
            "combined_depression_anxiety_score",
            "auditc_risk_group_label",
            "requesting_service_missing",
            "phq9_severity_group_code",
            "psychosis_3_something_very_strange_happening",
        ],
    )
    ablation_config.add_data_driven_group(top_features_group)

    # You can now use this in experiments
    # Note: This requires modifying the ablation study code to accept custom configs
    print("Custom feature group added: 'top_10_features'")
    print(f"  Features: {len(top_features_group.features)}")
    print(f"  Category: {top_features_group.category}")


def example_5_visualize_results():
    """Example 5: Generate visualizations from saved results"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Visualize Results")
    print("=" * 80 + "\n")

    # Find most recent results file
    results_dir = Path("disagreement_model/results/ablation")

    if results_dir.exists():
        pkl_files = list(results_dir.glob("ablation_results_*.pkl"))
        if pkl_files:
            # Get most recent
            latest_results = max(pkl_files, key=lambda p: p.stat().st_mtime)

            print(f"Loading results from: {latest_results}")

            # Generate visualizations
            visualizer = visualize_ablation_results(
                str(latest_results), output_dir=str(results_dir / "visualizations")
            )

            print("\n✓ Visualizations created successfully")
            print(f"  Output directory: {results_dir / 'visualizations'}")
        else:
            print("No ablation results found. Run an ablation study first.")
    else:
        print("Results directory does not exist. Run an ablation study first.")


def example_6_analyze_feature_importance():
    """Example 6: Analyze feature importance shifts across experiments"""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Feature Importance Analysis")
    print("=" * 80 + "\n")

    # Run quick ablation with key experiments
    config = ModelConfig()
    config.run_ablation_study = True
    config.verbose = 0  # Quiet mode
    config.ablation_experiments = [
        "baseline",
        "clinical_only",
        "without_depression",
        "without_anxiety",
    ]

    study = run_full_ablation_study(config)

    # Analyze feature importance shifts
    print("\nTop 5 features by experiment:")
    print("-" * 80)

    for exp_name in config.ablation_experiments:
        if exp_name in study.results:
            result = study.results[exp_name]
            if "feature_importance" in result:
                top_5 = result["feature_importance"][:5]
                print(f"\n{exp_name}:")
                for i, feat in enumerate(top_5, 1):
                    print(f"  {i}. {feat['feature']:45s} {feat['importance']:.6f}")


if __name__ == "__main__":
    import sys

    examples = {
        "1": ("Quick Start", example_1_quick_start),
        "2": ("Custom Experiments", example_2_custom_experiments),
        "3": ("Parallel Execution", example_3_parallel_execution),
        "4": ("Custom Feature Groups", example_4_custom_feature_groups),
        "5": ("Visualize Results", example_5_visualize_results),
        "6": ("Feature Importance Analysis", example_6_analyze_feature_importance),
    }

    if len(sys.argv) > 1 and sys.argv[1] in examples:
        # Run specific example
        name, func = examples[sys.argv[1]]
        print(f"\nRunning Example {sys.argv[1]}: {name}")
        func()
    else:
        # Print menu
        print("\n" + "=" * 80)
        print("ABLATION STUDY EXAMPLES")
        print("=" * 80)
        print("\nAvailable examples:")
        for num, (name, _) in examples.items():
            print(f"  {num}. {name}")

        print("\nUsage:")
        print("  python ablation_examples.py <example_number>")
        print("\nExample:")
        print("  python ablation_examples.py 1")
        print("\n" + "=" * 80 + "\n")
