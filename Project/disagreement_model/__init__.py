"""
Decision Tree Disagreement Prediction Model Package

Main modules:
- disagreement_model: Core model training with SMAC3 optimization
- ablation_study: Systematic feature group ablation analysis
- ablation_config: Feature group definitions and experiment configurations
- ablation_visualization: Visualization tools for ablation results
- model_config: Centralized configuration for all settings
- run_model: Main entry point for standard training
- run_ablation: Main entry point for ablation study
"""

__version__ = "1.1.0"

# Core components
from .model_config import ModelConfig
from .disagreement_model import DisagreementModel

# Ablation study components
from .ablation_config import AblationConfig, FeatureGroup
from .ablation_study import AblationStudy, run_full_ablation_study
from .ablation_visualization import AblationVisualizer, visualize_ablation_results

__all__ = [
    "ModelConfig",
    "DisagreementModel",
    "AblationConfig",
    "FeatureGroup",
    "AblationStudy",
    "run_full_ablation_study",
    "AblationVisualizer",
    "visualize_ablation_results",
]
