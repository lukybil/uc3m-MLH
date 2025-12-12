"""
Main runner script for the disagreement prediction model.
Orchestrates data preparation, model training, visualization, and rule extraction.
"""

import sys
from pathlib import Path
from datetime import datetime

# Fix Windows encoding once at entry point
from .encoding_utils import fix_windows_encoding

fix_windows_encoding()

from .model_config import ModelConfig
from data_preparation import DataPreparator
from .disagreement_model import DisagreementModel
from .model_visualization import ModelVisualizer
from .rule_extraction import RuleExtractor


def run_full_pipeline(config: ModelConfig = None):
    """
    Run the complete model pipeline:
    1. Data preparation
    2. Hyperparameter optimization
    3. Model training
    4. Evaluation
    5. Visualization
    6. Rule extraction
    """
    if config is None:
        config = ModelConfig()

    start_time = datetime.now()

    print("=" * 80)
    print("DISAGREEMENT PREDICTION MODEL - COMPLETE PIPELINE")
    print("=" * 80)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # =========================================================================
    # STEP 1: DATA PREPARATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: DATA PREPARATION")
    print("=" * 80 + "\n")

    preparator = DataPreparator(config)
    X_train, y_train, X_test, y_test, feature_names = preparator.prepare_data()

    # =========================================================================
    # STEP 2: MODEL TRAINING WITH SMAC3 OPTIMIZATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: MODEL TRAINING")
    print("=" * 80 + "\n")

    model_trainer = DisagreementModel(config)

    # Optimize hyperparameters
    best_params = model_trainer.optimize_hyperparameters(X_train, y_train)

    # Train final model
    model_trainer.train_final_model(X_train, y_train, feature_names)

    # =========================================================================
    # STEP 3: EVALUATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: MODEL EVALUATION")
    print("=" * 80 + "\n")

    metrics = model_trainer.evaluate(X_test, y_test)

    # =========================================================================
    # STEP 4: SAVE MODEL AND RESULTS
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: SAVING MODEL AND RESULTS")
    print("=" * 80 + "\n")

    if config.save_model:
        model_trainer.save_model()
    model_trainer.save_results()

    # =========================================================================
    # STEP 5: VISUALIZATIONS
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: GENERATING VISUALIZATIONS")
    print("=" * 80 + "\n")

    visualizer = ModelVisualizer(model_trainer.model, feature_names, config)
    visualizer.generate_all_visualizations(X_train, y_train, X_test, y_test)

    # =========================================================================
    # STEP 6: RULE EXTRACTION
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: EXTRACTING DECISION RULES")
    print("=" * 80 + "\n")

    rule_extractor = RuleExtractor(model_trainer.model, feature_names, config)
    rule_extractor.extract_rules()
    rule_extractor.export_all()

    # Print top rules summary
    print("\n" + "=" * 80)
    print("TOP 5 RULES FOR EACH PREDICTION")
    print("=" * 80 + "\n")

    print("Top 5 rules predicting DISAGREEMENT (by confidence):")
    print("-" * 80)
    top_disagree = rule_extractor.get_top_rules(
        n=5, by="confidence", prediction="Disagreement"
    )
    for i, rule in enumerate(top_disagree, 1):
        print(
            f"\n{i}. Confidence: {rule['confidence']*100:.1f}%, Support: {rule['support']} samples"
        )
        print(f"   Conditions: {len(rule['conditions'])}")
        for cond in rule["conditions"][:3]:  # Show first 3 conditions
            print(f"   - {cond}")
        if len(rule["conditions"]) > 3:
            print(f"   ... and {len(rule['conditions'])-3} more")

    print("\n" + "=" * 80)
    print("Top 5 rules predicting AGREEMENT (by confidence):")
    print("-" * 80)
    top_agree = rule_extractor.get_top_rules(
        n=5, by="confidence", prediction="Agreement"
    )
    for i, rule in enumerate(top_agree, 1):
        print(
            f"\n{i}. Confidence: {rule['confidence']*100:.1f}%, Support: {rule['support']} samples"
        )
        print(f"   Conditions: {len(rule['conditions'])}")
        for cond in rule["conditions"][:3]:
            print(f"   - {cond}")
        if len(rule["conditions"]) > 3:
            print(f"   ... and {len(rule['conditions'])-3} more")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {duration}")
    print(f"\nModel Performance:")
    print(f"  Test Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Test F1-Score:  {metrics['f1']:.4f}")
    print(f"  Test ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"\nOutputs saved to: {config.output_dir}/")
    print(f"  - Model: {config.model_filename}")
    print(f"  - Results: {config.results_prefix}_results.txt")
    print(f"  - Metrics: {config.results_prefix}_metrics.json")
    print(f"  - Rules: {config.results_prefix}_rules.*")
    print(f"  - Visualizations: {config.results_prefix}_*.png")
    print("=" * 80 + "\n")

    return {
        "model": model_trainer,
        "visualizer": visualizer,
        "rule_extractor": rule_extractor,
        "metrics": metrics,
        "duration": duration,
    }


def run_quick_test():
    """Run a quick test with reduced trials for fast iteration"""
    print("Running quick test configuration (faster, less comprehensive)...\n")

    config = ModelConfig()
    config.smac_n_trials = 20  # Fewer trials
    config.smac_walltime_limit = 600  # 10 minutes max
    config.cv_folds = 3  # Fewer folds
    config.verbose = 1

    return run_full_pipeline(config)


def run_evaluation_only(model_path: str = None, config: ModelConfig = None):
    """
    Run evaluation only using a pre-trained model.
    Skips training and optimization, only performs evaluation and visualization.

    Args:
        model_path: Path to the saved model file (optional, uses config default if not provided)
        config: ModelConfig instance (optional, creates default if not provided)
    """
    if config is None:
        config = ModelConfig()

    start_time = datetime.now()

    print("=" * 80)
    print("DISAGREEMENT PREDICTION MODEL - EVALUATION ONLY MODE")
    print("=" * 80)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # =========================================================================
    # STEP 1: DATA PREPARATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: DATA PREPARATION")
    print("=" * 80 + "\n")

    preparator = DataPreparator(config)
    X_train, y_train, X_test, y_test, feature_names = preparator.prepare_data()

    # =========================================================================
    # STEP 2: LOAD PRE-TRAINED MODEL
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: LOADING PRE-TRAINED MODEL")
    print("=" * 80 + "\n")

    model_trainer = DisagreementModel(config)
    model_data = model_trainer.load_model(model_path)

    # =========================================================================
    # STEP 3: EVALUATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: MODEL EVALUATION")
    print("=" * 80 + "\n")

    metrics = model_trainer.evaluate(X_test, y_test)

    # =========================================================================
    # STEP 4: VISUALIZATIONS
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: GENERATING VISUALIZATIONS")
    print("=" * 80 + "\n")

    visualizer = ModelVisualizer(model_trainer.model, feature_names, config)
    visualizer.generate_all_visualizations(X_train, y_train, X_test, y_test)

    # =========================================================================
    # STEP 5: RULE EXTRACTION
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: EXTRACTING DECISION RULES")
    print("=" * 80 + "\n")

    rule_extractor = RuleExtractor(model_trainer.model, feature_names, config)
    rule_extractor.extract_rules()
    rule_extractor.export_all()

    # Print top rules summary
    print("\n" + "=" * 80)
    print("TOP 5 RULES FOR EACH PREDICTION")
    print("=" * 80 + "\n")

    print("Top 5 rules predicting DISAGREEMENT (by confidence):")
    print("-" * 80)
    top_disagree = rule_extractor.get_top_rules(
        n=5, by="confidence", prediction="Disagreement"
    )
    for i, rule in enumerate(top_disagree, 1):
        print(
            f"\n{i}. Confidence: {rule['confidence']*100:.1f}%, Support: {rule['support']} samples"
        )
        print(f"   Conditions: {len(rule['conditions'])}")
        for cond in rule["conditions"][:3]:  # Show first 3 conditions
            print(f"   - {cond}")
        if len(rule["conditions"]) > 3:
            print(f"   ... and {len(rule['conditions'])-3} more")

    print("\n" + "=" * 80)
    print("Top 5 rules predicting AGREEMENT (by confidence):")
    print("-" * 80)
    top_agree = rule_extractor.get_top_rules(
        n=5, by="confidence", prediction="Agreement"
    )
    for i, rule in enumerate(top_agree, 1):
        print(
            f"\n{i}. Confidence: {rule['confidence']*100:.1f}%, Support: {rule['support']} samples"
        )
        print(f"   Conditions: {len(rule['conditions'])}")
        for cond in rule["conditions"][:3]:
            print(f"   - {cond}")
        if len(rule["conditions"]) > 3:
            print(f"   ... and {len(rule['conditions'])-3} more")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {duration}")
    print(f"\nModel Performance:")
    print(f"  Test Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Test F1-Score:  {metrics['f1']:.4f}")
    print(f"  Test ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"\nOutputs saved to: {config.output_dir}/")
    print(f"  - Rules: {config.results_prefix}_rules.*")
    print(f"  - Visualizations: {config.results_prefix}_*.png")
    print("=" * 80 + "\n")

    return {
        "model": model_trainer,
        "visualizer": visualizer,
        "rule_extractor": rule_extractor,
        "metrics": metrics,
        "duration": duration,
    }


def run_with_custom_config(
    feature_mode: str = "all", missing_strategy: str = "indicator", n_trials: int = 100
):
    """
    Run pipeline with custom configuration.

    Args:
        feature_mode: 'all', 'clinical', 'clinical_plus', 'custom'
        missing_strategy: 'simple', 'indicator', 'iterative', 'drop'
        n_trials: Number of SMAC3 optimization trials
    """
    config = ModelConfig()
    config.feature_mode = feature_mode
    config.missing_strategy = missing_strategy
    config.smac_n_trials = n_trials

    print(f"Running with custom configuration:")
    print(f"  Feature mode: {feature_mode}")
    print(f"  Missing strategy: {missing_strategy}")
    print(f"  SMAC trials: {n_trials}\n")

    return run_full_pipeline(config)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train disagreement prediction model with SMAC3 optimization"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "quick", "custom", "eval"],
        default="full",
        help="Run mode: full (default), quick (test), custom, or eval (evaluation only)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to pre-trained model file (for eval mode)",
    )
    parser.add_argument(
        "--features",
        choices=["all", "clinical", "clinical_plus", "custom"],
        default="all",
        help="Feature selection mode (for custom mode)",
    )
    parser.add_argument(
        "--missing",
        choices=["simple", "indicator", "iterative", "drop"],
        default="indicator",
        help="Missing data strategy (for custom mode)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Number of SMAC3 trials (for custom mode)",
    )

    args = parser.parse_args()

    if args.mode == "full":
        results = run_full_pipeline()
    elif args.mode == "quick":
        results = run_quick_test()
    elif args.mode == "custom":
        results = run_with_custom_config(
            feature_mode=args.features,
            missing_strategy=args.missing,
            n_trials=args.trials,
        )
    elif args.mode == "eval":
        results = run_evaluation_only(model_path=args.model_path)


if __name__ == "__main__":
    main()
