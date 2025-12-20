# Disagreement Model Ablation Study

Comprehensive ablation study framework for systematic analysis of feature group contributions to disagreement prediction.

## Overview

The ablation study systematically evaluates the impact of different feature groups on model performance by:
1. Training a **baseline model** with all features using SMAC3 hyperparameter optimization
2. Running **ablation experiments** that remove specific feature groups using the same hyperparameters
3. Comparing performance metrics and statistical significance

## Key Features

- ✅ **Semantic Feature Groups**: Clinically meaningful groupings (Depression, Anxiety, Suicide Risk, etc.)
- ✅ **Baseline Hyperparameter Reuse**: SMAC3 optimization only for baseline, ensuring fair comparison
- ✅ **Parallel Execution**: Multi-core support with proper Ctrl+C handling
- ✅ **McNemar's Test**: Statistical significance testing for performance differences
- ✅ **Comprehensive Visualizations**: Performance charts, feature importance heatmaps, complexity plots
- ✅ **Extensible Design**: Easy to add data-driven correlation groups later

## Quick Start

### List Available Experiments

```bash
python -m disagreement_model.run_ablation --list-experiments
```

### Run All Experiments (Sequential)

```bash
python -m disagreement_model.run_ablation
```

### Run Specific Experiments

```bash
python -m disagreement_model.run_ablation --experiments clinical_only no_administrative core_clinical
```

### Run with Parallel Execution

```bash
# Use 4 cores
python -m disagreement_model.run_ablation --parallel 4

# Use all available cores
python -m disagreement_model.run_ablation --parallel -1
```

### Custom Configuration

```bash
python -m disagreement_model.run_ablation \
    --data data/merged.csv \
    --output-dir results/my_ablation \
    --smac-trials 150 \
    --parallel 4 \
    --timeout 2400 \
    --mcnemar-alpha 0.01 \
    --verbose 2
```

## Feature Groups

### Core Clinical Groups

1. **Depression** (`depression`): PHQ-9 total scores, severity groups, individual items
2. **Anxiety** (`anxiety`): GAD-7 total scores, severity groups, individual items  
3. **Suicide Risk** (`suicide_risk`): C-SSRS total scores, groups, individual items
4. **Substance Use** (`substance_use`): AUDIT-C (alcohol) + DAST (drugs)
5. **Psychosis** (`psychosis`): 4 psychosis screening items + symptom count
6. **Well-being & Other** (`wellbeing_other`): WHO-5, self-harm, medication, smoking

### Contextual Groups

7. **Administrative** (`administrative`): Service, center, evaluating user, patient ID
8. **Demographics** (`demographics`): Age, sex, location of birth
9. **Temporal** (`temporal`): Time delays between assessment stages
10. **Personality** (`personality`): BFI-10 Big Five personality traits
11. **Engineered Composites** (`engineered_composites`): Combined scores

## Ablation Experiments

### Strategic Experiments

- **`baseline`**: All features (reference model)
- **`clinical_only`**: Only clinical features (remove administrative, demographics, temporal)
- **`no_administrative`**: Remove potentially leaky administrative features
- **`no_temporal`**: Remove time-based features
- **`scores_only`**: Only summary scores (remove individual items)
- **`core_clinical`**: Only PHQ-9 + GAD-7 + C-SSRS
- **`no_personality`**: Remove BFI-10 personality traits

### Leave-One-Out Experiments

Remove one semantic group at a time:
- `without_depression`
- `without_anxiety`
- `without_suicide_risk`
- `without_substance_use`
- `without_psychosis`
- `without_wellbeing_other`
- `without_administrative`
- `without_demographics`
- `without_temporal`
- `without_personality`
- `without_engineered_composites`

## Output Files

All results are saved to `disagreement_model/results/ablation/` (configurable):

### Summary Files
- **`ablation_comparison_TIMESTAMP.csv`**: Performance metrics table for all experiments
- **`ablation_report_TIMESTAMP.txt`**: Detailed text report with configuration and results
- **`ablation_summary_TIMESTAMP.json`**: JSON summary (without predictions for size)

### Data Files
- **`ablation_results_TIMESTAMP.pkl`**: Full results with predictions (for reanalysis)
- **`baseline_model.pkl`**: Trained baseline model
- **`baseline_params.json`**: Optimized hyperparameters

### Visualizations
- **`ablation_performance_comparison.png`**: Bar charts comparing F1, ROC-AUC, accuracy
- **`ablation_performance_deltas.png`**: Waterfall plot showing Δ from baseline
- **`ablation_tree_complexity.png`**: Tree depth and leaf count comparison
- **`ablation_feature_importance_heatmap.png`**: Top features across experiments
- **`ablation_mcnemar_significance.png`**: Statistical significance p-values

## Programmatic Usage

### Basic Usage

```python
from disagreement_model.model_config import ModelConfig
from disagreement_model.ablation_study import run_full_ablation_study

# Configure
config = ModelConfig()
config.run_ablation_study = True
config.ablation_n_jobs = 4
config.run_mcnemar_test = True

# Run
study = run_full_ablation_study(config)

# Access results
print(study.compare_results())
```

### Custom Experiments

```python
from disagreement_model.ablation_config import AblationConfig, FeatureGroup

# Load default config
ablation_config = AblationConfig()

# Add custom data-driven group
custom_group = FeatureGroup(
    name='high_correlation_cluster',
    description='Features with correlation > 0.7',
    category='data_driven',
    features=['phq9_total_score', 'gad7_total_score', 'combined_depression_anxiety_score']
)
ablation_config.add_data_driven_group(custom_group)

# Run specific experiments
config.ablation_experiments = ['baseline', 'without_high_correlation_cluster']
```

### Generate Visualizations

```python
from disagreement_model.ablation_visualization import visualize_ablation_results

# From saved results
visualize_ablation_results(
    results_pkl_path='results/ablation/ablation_results_20231220_153045.pkl',
    output_dir='results/ablation/visualizations'
)
```

## Methodology

### Baseline Training

1. Load and preprocess data (same pipeline as regular training)
2. Run SMAC3 hyperparameter optimization (100 trials by default)
3. Train final model with best hyperparameters
4. Evaluate on test set and save predictions

### Ablation Experiments

1. For each experiment:
   - Filter dataset to selected feature groups
   - Train model using **baseline hyperparameters** (no re-optimization)
   - Evaluate on same test set
   - Compare predictions with baseline using McNemar's test

### Statistical Testing

**McNemar's Test** compares paired predictions:
- Null hypothesis: Both models have same error rate
- Tests if performance difference is statistically significant
- Exact test for small samples, asymptotic for large samples

## Configuration Options

See [model_config.py](model_config.py) for all options:

```python
# Ablation settings
run_ablation_study: bool = False
ablation_experiments: List[str] = []  # Empty = run all
use_baseline_params: bool = True
ablation_n_jobs: int = 1
ablation_timeout_per_experiment: int = 1800
run_mcnemar_test: bool = True
mcnemar_alpha: float = 0.05
ablation_output_dir: str = "disagreement_model/results/ablation"
```

## Extending the Framework

### Adding Data-Driven Groups

```python
from disagreement_model.ablation_config import AblationConfig, FeatureGroup

# After correlation analysis
ablation_config = AblationConfig()

correlation_group = FeatureGroup(
    name='correlation_cluster_1',
    description='Highly correlated clinical scores (r > 0.8)',
    category='data_driven',
    features=['phq9_total_score', 'phq9_severity_group_label', 'combined_depression_anxiety_score']
)

ablation_config.add_data_driven_group(correlation_group)
```

### Adding Custom Experiments

Edit `ablation_config.py`:

```python
experiments['custom_minimal'] = {
    'description': 'Minimal viable feature set',
    'include_groups': ['depression', 'anxiety'],
    'exclude_groups': []
}
```

## Performance Considerations

- **Sequential execution**: ~2-5 minutes per experiment (no SMAC3 optimization)
- **Parallel execution**: Linear speedup with number of cores
- **Memory usage**: ~500MB per parallel job
- **Estimated total time**: 
  - All experiments (sequential): ~40-60 minutes
  - All experiments (4 cores): ~10-15 minutes
  - All experiments (8 cores): ~5-8 minutes

## Troubleshooting

### Out of Memory

Reduce parallel jobs:
```bash
python -m disagreement_model.run_ablation --parallel 2
```

### Timeout Issues

Increase timeout per experiment:
```bash
python -m disagreement_model.run_ablation --timeout 3600
```

### Interrupted Study

Partial results are automatically saved as `ablation_results_PARTIAL_*.pkl`. Press Ctrl+C to gracefully interrupt.

## References

- **SMAC3**: Sequential Model-Based Algorithm Configuration ([docs](https://automl.github.io/SMAC3/))
- **McNemar's Test**: Statistical comparison of paired predictions ([statsmodels](https://www.statsmodels.org/))
- **Ablation Studies**: Systematic feature importance analysis methodology

## Citation

If you use this ablation study framework in your research, please cite:

```bibtex
@software{disagreement_model_ablation,
  title = {Ablation Study Framework for Disagreement Prediction Model},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/lukybil/uc3m-MLH}
}
```
