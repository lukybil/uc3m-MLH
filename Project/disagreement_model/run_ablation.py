"""
Main entry point for running ablation study on disagreement model.

Usage:
    python -m disagreement_model.run_ablation
    python -m disagreement_model.run_ablation --experiments clinical_only no_administrative
    python -m disagreement_model.run_ablation --parallel 4
    python -m disagreement_model.run_ablation --no-mcnemar
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

from disagreement_model.model_config import ModelConfig
from disagreement_model.ablation_study import run_full_ablation_study
from disagreement_model.ablation_config import AblationConfig


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run ablation study for disagreement prediction model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data arguments
    parser.add_argument(
        "--data",
        type=str,
        default="data/merged.csv",
        help="Path to input data CSV file",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="disagreement_model/results/ablation",
        help="Output directory for ablation results",
    )

    # Experiment selection
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="Specific experiments to run (default: all). Options: baseline, clinical_only, "
        "no_administrative, no_temporal, scores_only, core_clinical, no_personality, "
        "without_<group_name> for leave-one-out",
    )

    parser.add_argument(
        "--list-experiments",
        action="store_true",
        help="List all available experiments and exit",
    )

    # Execution settings
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel jobs (1=sequential, -1=all cores)",
    )

    parser.add_argument(
        "--timeout", type=int, default=1800, help="Timeout per experiment in seconds"
    )

    # Optimization settings
    parser.add_argument(
        "--smac-trials",
        type=int,
        default=100,
        help="Number of SMAC3 optimization trials for baseline",
    )

    parser.add_argument(
        "--no-baseline-params",
        action="store_true",
        help="Run SMAC3 optimization for each ablation experiment (not recommended)",
    )

    # Statistical testing
    parser.add_argument(
        "--no-mcnemar",
        action="store_true",
        help="Disable McNemar's test for comparing with baseline",
    )

    parser.add_argument(
        "--mcnemar-alpha",
        type=float,
        default=0.05,
        help="Significance level for McNemar's test",
    )

    # Other settings
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random state for reproducibility"
    )

    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Proportion of data for test set"
    )

    parser.add_argument(
        "--cv-folds", type=int, default=5, help="Number of cross-validation folds"
    )

    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbosity level (0=silent, 1=progress, 2=detailed)",
    )

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    # If listing experiments, print and exit
    if args.list_experiments:
        print("\n" + "=" * 80)
        print("AVAILABLE ABLATION EXPERIMENTS")
        print("=" * 80 + "\n")

        config = AblationConfig()
        config.print_summary()
        return 0

    # Create configuration
    config = ModelConfig()

    # Update configuration from arguments
    config.data_path = args.data
    config.ablation_output_dir = args.output_dir
    config.random_state = args.random_state
    config.test_size = args.test_size
    config.cv_folds = args.cv_folds
    config.verbose = args.verbose

    # Ablation-specific settings
    config.run_ablation_study = True
    config.ablation_experiments = args.experiments or []
    config.use_baseline_params = not args.no_baseline_params
    config.ablation_n_jobs = args.parallel
    config.ablation_timeout_per_experiment = args.timeout
    config.run_mcnemar_test = not args.no_mcnemar
    config.mcnemar_alpha = args.mcnemar_alpha
    config.smac_n_trials = args.smac_trials

    # Print configuration
    print("\n" + "=" * 80)
    print("ABLATION STUDY CONFIGURATION")
    print("=" * 80)
    print(f"Data path: {config.data_path}")
    print(f"Output directory: {config.ablation_output_dir}")
    print(f"Random state: {config.random_state}")
    print(f"Test size: {config.test_size}")
    print(f"CV folds: {config.cv_folds}")
    print(
        f"\nExperiments to run: {len(args.experiments) if args.experiments else 'ALL'}"
    )
    if args.experiments:
        for exp in args.experiments:
            print(f"  - {exp}")
    print(f"\nOptimization:")
    print(f"  SMAC3 trials (baseline): {config.smac_n_trials}")
    print(f"  Reuse baseline params: {config.use_baseline_params}")
    print(f"\nExecution:")
    print(f"  Parallel jobs: {config.ablation_n_jobs}")
    print(f"  Timeout per experiment: {config.ablation_timeout_per_experiment}s")
    print(f"\nStatistical testing:")
    print(f"  McNemar's test: {config.run_mcnemar_test}")
    if config.run_mcnemar_test:
        print(f"  Significance level: {config.mcnemar_alpha}")
    print("=" * 80 + "\n")

    # Confirmation prompt
    if config.ablation_n_jobs == 1:
        exp_count = len(args.experiments) if args.experiments else "~20"
        est_time = int(exp_count) * 2 if isinstance(exp_count, int) else 40
        print(f"Estimated time: ~{est_time} minutes (sequential execution)")
    else:
        print(f"Running with {config.ablation_n_jobs} parallel jobs")

    response = input("\nProceed with ablation study? [y/N]: ")
    if response.lower() not in ["y", "yes"]:
        print("Ablation study cancelled.")
        return 0

    # Run ablation study
    try:
        study = run_full_ablation_study(config)

        print("\n" + "=" * 80)
        print("✓ ABLATION STUDY COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Results saved to: {config.ablation_output_dir}")
        print(f"Experiments completed: {len(study.results)}")
        print("\nSee the following files for details:")
        print(f"  - ablation_comparison_*.csv (summary table)")
        print(f"  - ablation_report_*.txt (detailed report)")
        print(f"  - ablation_results_*.pkl (full results)")
        print("=" * 80 + "\n")

        return 0

    except KeyboardInterrupt:
        print("\n\nAblation study interrupted by user.")
        return 1

    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
