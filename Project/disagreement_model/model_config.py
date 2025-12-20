"""
Configuration file for the disagreement prediction model.
Centralizes all hyperparameters, feature selection, and model settings.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for the decision tree disagreement model"""

    # ============================================================================
    # DATA CONFIGURATION
    # ============================================================================

    data_path: str = "data/merged.csv"
    output_dir: str = "disagreement_model/results"
    random_state: int = 42
    test_size: float = 0.2

    # ============================================================================
    # FEATURE SELECTION
    # ============================================================================

    # Feature selection mode: 'all', 'clinical', 'clinical_plus', 'custom'
    feature_mode: str = "all"

    # Custom feature list (used only when feature_mode='custom')
    custom_features: Optional[List[str]] = None

    # Feature groups for easy configuration
    clinical_core_features: List[str] = field(
        default_factory=lambda: [
            # Depression (PHQ-9)
            "phq9_total_score",
            "phq9_severity_group_code",
            # Anxiety (GAD-7)
            "gad7_total_score",
            "gad7_severity_group_code",
            # Well-being (WHO-5)
            "who5_total_score",
            # Suicide risk (C-SSRS)
            "cssrs_total_score_1",
            "cssrs_total_score_2",
            "cssrs_total_score_3",
            "cssrs_group_2a_code",
            # Alcohol & drugs
            "auditc_total_score",
            "dast_total_score",
        ]
    )

    clinical_plus_features: List[str] = field(
        default_factory=lambda: [
            # Demographics
            "sex",
            "center_id",
            "location_of_birth",
            # Personality (BFI-10)
            "bfi10_extraversion_score",
            "bfi10_agreeableness_score",
            "bfi10_conscientiousness_score",
            "bfi10_emotional_stability_score",
            "bfi10_openness_to_experience_score",
            # Psychosis symptoms
            "psychosis_1_thoughts_being_interfered_with_or_controlled",
            "psychosis_2_group_conspiring_to_harm_you",
            "psychosis_3_something_very_strange_happening",
            "psychosis_4_heard_voices_with_no_one_there",
            # Self-harm
            "nssi_ever_intentional_self_injury_without_suicidal_intent",
            "nssi_last_month_frequency",
            # Medication & behavioral
            "current_psychotropic_medication",
            "current_smoker",
            "cigarettes_per_day",
            # Service info
            "service",
            "care_circuit",
            "response_type",
            "evaluating_user",
        ]
    )

    # Columns to always exclude (target, IDs, etc.)
    exclude_columns: List[str] = field(
        default_factory=lambda: [
            # Target variables
            "professional_recommendation",
            "algorithm_recommendation",
            "professional_recommendation_clinician",
            "algorithm_professional_agreement",
            # IDs and administrative
            "patient_id",
            "episode_id",
            "assessment_id",
            "record_id",
            # Text fields (too high cardinality)
            "risk_assessment_algorithm",
            "risk_assessment_algorithm_2",
            "stratification_algorithm_raw",
            "referral",
            "risk_assessment_psychiatrist",
            # Status fields (likely data leakage)
            "record_status",
            "form_status",
            "package_evaluated",
            # Death status (very imbalanced and potentially sensitive)
            "death_status",
        ]
    )

    # Date columns for feature engineering
    date_columns: List[str] = field(
        default_factory=lambda: [
            "date_of_birth",
            "assessment_date",
            "episode_date",
            "request_date",
            "publication_date",
            "report_date",
            "from_date",
            "to_date",
            "completion_date",
            "results_date",
            "evaluation_date",
        ]
    )

    # ============================================================================
    # MISSING DATA HANDLING
    # ============================================================================

    # Missing data strategy: 'simple', 'indicator', 'iterative', 'drop'
    missing_strategy: str = "indicator"

    # For 'simple' strategy
    numeric_imputation: str = "median"  # 'mean', 'median', 'most_frequent'
    categorical_imputation: str = "most_frequent"  # 'most_frequent', 'constant'

    # For 'indicator' strategy (simple + adds missingness indicators)
    add_missingness_indicators: bool = True

    # For 'drop' strategy
    missing_threshold: float = 0.2  # Drop features with >20% missing

    # Specific columns to handle differently (e.g., important clinical scores)
    special_imputation: Dict[str, str] = field(
        default_factory=lambda: {
            "auditc_total_score": "median",  # High missingness but clinically relevant
            "who5_total_score": "median",
            "nssi_last_month_frequency": "constant_0",
        }
    )

    # ============================================================================
    # SMAC3 HYPERPARAMETER OPTIMIZATION
    # ============================================================================

    # SMAC3 settings
    smac_n_trials: int = 100  # Number of configurations to evaluate
    smac_walltime_limit: int = 3600  # Maximum time in seconds (1 hour)
    smac_n_workers: int = 1  # Parallel workers (set to 1 for reproducibility)

    # Cross-validation settings
    cv_folds: int = 5
    cv_scoring: str = "f1"  # 'accuracy', 'f1', 'roc_auc', 'balanced_accuracy'

    # Hyperparameter search space for Decision Tree
    hp_max_depth: tuple = (3, 15)  # (min, max)
    hp_min_samples_split: tuple = (50, 500)
    hp_min_samples_leaf: tuple = (25, 250)
    hp_min_impurity_decrease: tuple = (0.0, 0.01)
    hp_criterion: List[str] = field(default_factory=lambda: ["gini", "entropy"])
    hp_splitter: List[str] = field(default_factory=lambda: ["best", "random"])
    hp_max_features: List[str] = field(
        default_factory=lambda: ["sqrt", "log2", None]  # None means use all features
    )
    hp_class_weight: List[str] = field(default_factory=lambda: ["balanced", None])

    # ============================================================================
    # MODEL SETTINGS
    # ============================================================================

    # Target variable definition
    target_column: str = "disagreement"  # Will be created from recommendations

    # Create ordinal disagreement magnitude? (|algorithm - professional|)
    create_magnitude_target: bool = True
    magnitude_target_column: str = "disagreement_magnitude"

    # Minimum support for decision tree rules
    min_rule_support: int = 10  # Minimum samples in leaf for rule export

    # ============================================================================
    # VISUALIZATION SETTINGS
    # ============================================================================

    # Tree visualization
    viz_max_depth: Optional[int] = 5  # Max depth to show in visualizations
    viz_feature_importance_top_n: int = 20
    viz_figsize: tuple = (20, 10)
    viz_dpi: int = 300

    # Rule export format
    export_rules_txt: bool = True
    export_rules_csv: bool = True
    export_rules_latex: bool = True
    export_rules_json: bool = True

    # Generate interactive HTML tree (requires dtreeviz)
    generate_interactive_tree: bool = True

    # ============================================================================
    # EVALUATION SETTINGS
    # ============================================================================

    # Metrics to compute
    compute_confusion_matrix: bool = True
    compute_classification_report: bool = True
    compute_roc_curve: bool = True
    compute_feature_importance: bool = True

    # Stratification variables for analysis
    stratify_by: List[str] = field(
        default_factory=lambda: [
            "evaluating_user",
            "center_id",
            "service",
        ]
    )

    # ============================================================================
    # OUTPUT SETTINGS
    # ============================================================================

    # Model output files
    save_model: bool = True
    model_filename: str = "disagreement_tree_model.pkl"

    # Results files
    results_prefix: str = "disagreement_model"

    # Verbose output
    verbose: int = 1  # 0: silent, 1: progress, 2: detailed

    # ============================================================================
    # ABLATION STUDY SETTINGS
    # ============================================================================

    # Run ablation study instead of regular training
    run_ablation_study: bool = False

    # Ablation experiment names to run (empty list = run all)
    # Options: 'baseline', 'clinical_only', 'no_administrative', 'no_temporal',
    #          'scores_only', 'core_clinical', 'no_personality',
    #          'without_<group_name>' for leave-one-out experiments
    ablation_experiments: List[str] = field(default_factory=list)

    # Use baseline hyperparameters for all ablation experiments
    # If True, only runs SMAC3 optimization for baseline, then reuses params
    # If False, optimizes hyperparameters for each ablation experiment
    use_baseline_params: bool = True

    # Path to baseline model/params (auto-generated if None)
    baseline_params_path: Optional[str] = None

    # Parallel execution settings for ablation experiments
    ablation_n_jobs: int = 1  # Number of parallel jobs (1 = sequential, -1 = all cores)
    ablation_timeout_per_experiment: int = 1800  # Max seconds per experiment (30 min)

    # Statistical significance testing
    run_mcnemar_test: bool = (
        True  # Compare predictions with baseline using McNemar's test
    )
    mcnemar_alpha: float = 0.05  # Significance level

    # Ablation output directory
    ablation_output_dir: str = "disagreement_model/results/ablation"

    def get_features(self) -> List[str]:
        """Get the list of features based on feature_mode"""
        if self.feature_mode == "all":
            return "all"  # Will use all columns except excluded ones
        elif self.feature_mode == "clinical":
            return self.clinical_core_features
        elif self.feature_mode == "clinical_plus":
            return self.clinical_core_features + self.clinical_plus_features
        elif self.feature_mode == "custom":
            if self.custom_features is None:
                raise ValueError(
                    "custom_features must be provided when feature_mode='custom'"
                )
            return self.custom_features
        else:
            raise ValueError(f"Unknown feature_mode: {self.feature_mode}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging"""
        return {
            "data_path": self.data_path,
            "feature_mode": self.feature_mode,
            "missing_strategy": self.missing_strategy,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "smac_n_trials": self.smac_n_trials,
            "cv_folds": self.cv_folds,
            "cv_scoring": self.cv_scoring,
        }


# Create default configuration instance
default_config = ModelConfig()
