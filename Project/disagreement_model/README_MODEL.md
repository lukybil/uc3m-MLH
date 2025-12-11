# Disagreement Prediction Model

An explainable decision tree model with SMAC3 hyperparameter optimization to predict when algorithm and professional recommendations disagree in mental health assessment.

## Overview

This project builds an interpretable machine learning model to understand the 43.88% disagreement rate between algorithmic and professional clinical recommendations. The model uses decision trees for maximum interpretability, with comprehensive visualization and rule extraction utilities.

## Features

- ✅ **Flexible Feature Selection**: Use all features, clinical scores only, or custom sets
- ✅ **Multiple Missing Data Strategies**: Simple imputation, indicators, iterative (MICE), or drop
- ✅ **SMAC3 Hyperparameter Optimization**: Efficient Bayesian optimization with 100+ trials
- ✅ **Comprehensive Visualizations**: Tree plots, feature importance, ROC curves, confusion matrices
- ✅ **Rule Extraction**: Export decision rules in TXT, CSV, LaTeX, and JSON formats
- ✅ **Easy Configuration**: Centralized config file for all settings

## Project Structure

```
Project/
├── model_config.py           # Configuration settings
├── data_preparation.py       # Data loading and preprocessing
├── disagreement_model.py     # Model training with SMAC3
├── model_visualization.py    # Visualization utilities
├── rule_extraction.py        # Decision rule extraction
├── run_model.py             # Main runner script
├── requirements.txt         # Python dependencies
├── data/
│   └── merged.csv          # Input data
└── results/                # Output directory (auto-created)
    ├── *.pkl               # Trained model
    ├── *_results.txt       # Results summary
    ├── *_metrics.json      # Performance metrics
    ├── *_rules.*          # Decision rules (TXT/CSV/LaTeX/JSON)
    └── *_*.png            # Visualizations
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install Dependencies

```powershell
# Install all required packages
pip install -r requirements.txt

# Optional: Install dtreeviz for interactive tree visualization
pip install dtreeviz
```

## Quick Start

### Basic Usage

Run the complete pipeline with default settings:

```powershell
python run_model.py
```

This will:
1. Load and preprocess data from `data/merged.csv`
2. Optimize hyperparameters using SMAC3 (100 trials)
3. Train the final decision tree model
4. Generate visualizations
5. Extract and export decision rules
6. Save all results to `results/`

### Quick Test Mode

Run a faster version for testing (20 trials, 3-fold CV):

```powershell
python run_model.py --mode quick
```

### Custom Configuration

Run with specific settings:

```powershell
# Use only clinical features with iterative imputation
python run_model.py --mode custom --features clinical --missing iterative --trials 50

# Options:
#   --features: all, clinical, clinical_plus, custom
#   --missing: simple, indicator, iterative, drop
#   --trials: number of SMAC3 optimization trials
```

## Configuration

Edit `model_config.py` to customize:

### Feature Selection

```python
config = ModelConfig()

# Use all features (default)
config.feature_mode = 'all'

# Use only clinical scores (PHQ-9, GAD-7, WHO-5, C-SSRS, etc.)
config.feature_mode = 'clinical'

# Use clinical + demographics + personality traits
config.feature_mode = 'clinical_plus'

# Use custom feature list
config.feature_mode = 'custom'
config.custom_features = ['phq9_total_score', 'gad7_total_score', 'age']
```

### Missing Data Handling

```python
# Simple imputation: median for numeric, mode for categorical
config.missing_strategy = 'simple'

# Add binary indicators for missingness (recommended)
config.missing_strategy = 'indicator'

# Iterative imputation (MICE algorithm)
config.missing_strategy = 'iterative'

# Drop features with >20% missing
config.missing_strategy = 'drop'
config.missing_threshold = 0.2
```

### SMAC3 Optimization

```python
# Number of configurations to evaluate
config.smac_n_trials = 100

# Maximum time in seconds
config.smac_walltime_limit = 3600  # 1 hour

# Cross-validation settings
config.cv_folds = 5
config.cv_scoring = 'f1'  # or 'accuracy', 'roc_auc', 'balanced_accuracy'
```

### Hyperparameter Search Space

```python
# Decision tree hyperparameters
config.hp_max_depth = (3, 15)
config.hp_min_samples_split = (50, 500)
config.hp_min_samples_leaf = (25, 250)
config.hp_min_impurity_decrease = (0.0, 0.01)
config.hp_criterion = ['gini', 'entropy']
config.hp_splitter = ['best', 'random']
config.hp_max_features = ['sqrt', 'log2', None]
config.hp_class_weight = ['balanced', None]
```

## Output Files

### Model Files
- `disagreement_tree_model.pkl`: Trained model with metadata

### Results
- `disagreement_model_results.txt`: Comprehensive text summary
- `disagreement_model_metrics.json`: Performance metrics in JSON

### Decision Rules
- `disagreement_model_rules.txt`: Human-readable rules
- `disagreement_model_rules.csv`: Rules in tabular format
- `disagreement_model_rules.tex`: LaTeX-formatted rules for papers
- `disagreement_model_rules.json`: Rules in JSON format
- `disagreement_model_rule_analysis.json`: Rule pattern analysis

### Visualizations
- `disagreement_model_tree_structure.png/.pdf`: Full tree visualization
- `disagreement_model_feature_importance.png`: Top 20 features bar chart
- `disagreement_model_confusion_matrix.png`: Confusion matrix heatmap
- `disagreement_model_roc_curve.png`: ROC curve with AUC
- `disagreement_model_tree_text.txt`: Text representation of tree
- `disagreement_model_interactive_tree.svg`: Interactive tree (if dtreeviz installed)

## Understanding the Output

### Decision Rules Format

Each rule includes:
- **Conditions**: IF-THEN statements showing decision path
- **Prediction**: Agreement or Disagreement
- **Confidence**: Purity of the leaf node (0-100%)
- **Support**: Number of training samples matching the rule
- **Sample breakdown**: How many samples of each class

Example rule:
```
Rule #1
IF:
  1. phq9_total_score <= 12.5000
  2. gad7_total_score <= 10.5000
  3. cssrs_total_score_1 <= 0.5000
THEN:
  Prediction: Agreement
  Confidence: 87.34%
  Support: 1245 samples
  (Agreement: 1088, Disagreement: 157)
```

### Feature Importance

Features are ranked by Gini importance (total reduction in node impurity). Higher values indicate more important features for prediction.

### Performance Metrics

- **Accuracy**: Overall correctness
- **Precision**: Of predicted disagreements, how many were correct
- **Recall**: Of actual disagreements, how many were caught
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve (discrimination ability)

## Advanced Usage

### Programmatic Access

```python
from model_config import ModelConfig
from run_model import run_full_pipeline

# Create custom configuration
config = ModelConfig()
config.feature_mode = 'clinical_plus'
config.missing_strategy = 'indicator'
config.smac_n_trials = 150

# Run pipeline
results = run_full_pipeline(config)

# Access components
model = results['model']
visualizer = results['visualizer']
rule_extractor = results['rule_extractor']
metrics = results['metrics']

# Get specific rules
top_disagree_rules = rule_extractor.get_top_rules(
    n=10, 
    by='confidence', 
    prediction='Disagreement'
)
```

### Loading a Saved Model

```python
import pickle
from pathlib import Path

# Load model
model_path = Path('results/disagreement_tree_model.pkl')
with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
feature_names = model_data['feature_names']
best_params = model_data['best_params']
test_metrics = model_data['test_metrics']

# Make predictions
import pandas as pd
X_new = pd.DataFrame(...)  # Your new data
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)
```

### Re-generate Visualizations

```python
from model_visualization import visualize_from_saved_model

visualize_from_saved_model(
    'results/disagreement_tree_model.pkl',
    X_train, y_train,
    X_test, y_test
)
```

### Extract Rules from Saved Model

```python
from rule_extraction import extract_rules_from_saved_model

extractor = extract_rules_from_saved_model('results/disagreement_tree_model.pkl')
rules = extractor.rules
```

## Data Requirements

### Input Data Format

The model expects `data/merged.csv` with the following structure:

#### Required Columns
- `algorithm_recommendation` (1.0-5.0): Algorithm's care recommendation
- `professional_recommendation_clinician` (1.0-5.0): Clinician's recommendation

#### Clinical Scores
- PHQ-9: Depression (phq9_total_score, phq9_1 through phq9_9)
- GAD-7: Anxiety (gad7_total_score, gad7_1 through gad7_7)
- WHO-5: Well-being (who5_total_score)
- C-SSRS: Suicide risk (cssrs_total_score_1/2/3, cssrs_group_2a_code)
- AUDIT-C: Alcohol use (auditc_total_score)
- DAST: Drug abuse (dast_total_score)

#### Personality Traits (BFI-10)
- bfi10_extraversion_score
- bfi10_agreeableness_score
- bfi10_conscientiousness_score
- bfi10_emotional_stability_score
- bfi10_openness_to_experience_score

#### Demographics
- sex, age (or date_of_birth), center_id, location_of_birth

#### Additional Features
- Psychosis symptoms, self-harm indicators, medication status, service information

## Troubleshooting

### Common Issues

**1. SMAC3 import errors**
```powershell
pip install smac>=2.0.0 ConfigSpace>=0.7.0
```

**2. Memory errors with large datasets**
- Reduce `smac_n_trials` in config
- Use `feature_mode='clinical'` instead of `'all'`
- Increase `missing_threshold` to drop sparse features

**3. Visualization errors**
```powershell
pip install matplotlib seaborn
```

**4. Slow optimization**
- Use `--mode quick` for testing
- Reduce `smac_n_trials` (default: 100)
- Reduce `cv_folds` (default: 5)
- Decrease `smac_walltime_limit`

## Citation

If you use this code in your research, please cite:

```bibtex
@software{disagreement_model_2025,
  title={Explainable Decision Tree for Clinical Recommendation Disagreement},
  author={[Your Name]},
  year={2025},
  institution={UC3M}
}
```

## License

This project is for academic research purposes.

## Contact

For questions or issues, please open an issue in the repository or contact [your contact info].

---

**Version**: 1.0.0  
**Last Updated**: December 2025
