# Ablation Study - Quick Setup Guide

## Prerequisites

Ensure you have all required dependencies installed:

```bash
pip install -r requirements.txt
```

### Key Dependencies
- scikit-learn >= 1.0
- smac >= 2.0
- pandas >= 1.5
- numpy >= 1.20
- matplotlib >= 3.5
- seaborn >= 0.12
- scipy >= 1.9

## Verify Installation

Test that all modules can be imported:

```python
python -c "from disagreement_model import AblationConfig, run_full_ablation_study; print('✓ Installation successful')"
```

## Quick Test Run

### 1. List Available Experiments

```bash
python -m disagreement_model.run_ablation --list-experiments
```

Expected output: Shows 11 semantic feature groups and ~18 available experiments.

### 2. Run Mini Ablation Study (Fast Test)

Test with just 3 experiments and reduced SMAC trials:

```bash
python -m disagreement_model.run_ablation \
    --experiments baseline clinical_only without_depression \
    --smac-trials 20 \
    --timeout 600 \
    --verbose 2
```

**Expected time:** ~5-10 minutes
**Expected output:** 
- Baseline model training with SMAC3
- 2 ablation experiments
- Results saved to `disagreement_model/results/ablation/`

### 3. Verify Output Files

Check that files were created:

```bash
ls disagreement_model/results/ablation/
```

You should see:
- `baseline_model.pkl`
- `baseline_params.json`
- `ablation_results_*.pkl`
- `ablation_comparison_*.csv`
- `ablation_report_*.txt`

### 4. Generate Visualizations

```bash
python -m disagreement_model.ablation_examples 5
```

This will create visualizations in `disagreement_model/results/ablation/visualizations/`.

## Full Production Run

Once verified, run the complete ablation study:

```bash
python -m disagreement_model.run_ablation \
    --parallel 4 \
    --smac-trials 100 \
    --timeout 1800 \
    --verbose 1
```

**Expected time:** ~15-30 minutes (with 4 cores)
**Total experiments:** ~19 (baseline + 18 ablations)

## Troubleshooting

### Import Errors

If you get import errors:

```bash
# Ensure you're in the Project directory
cd d:\Dev\UC3M\MLH\Project

# Verify PYTHONPATH includes current directory
python -c "import sys; print('\n'.join(sys.path))"
```

### SMAC3 Installation Issues

If SMAC3 fails to install:

```bash
pip install --upgrade pip
pip install smac==2.0.2
```

### Memory Errors

If you run out of memory with parallel execution:

```bash
# Reduce parallel jobs
python -m disagreement_model.run_ablation --parallel 2
```

### Timeout Issues

If experiments timeout:

```bash
# Increase timeout
python -m disagreement_model.run_ablation --timeout 3600
```

## File Structure Verification

Ensure your directory structure matches:

```
Project/
├── data/
│   └── merged.csv                           # Your data file
├── disagreement_model/
│   ├── __init__.py                          # ✓ Updated
│   ├── model_config.py                      # ✓ Updated
│   ├── disagreement_model.py                # Existing
│   ├── ablation_config.py                   # ✓ NEW
│   ├── ablation_study.py                    # ✓ NEW
│   ├── ablation_visualization.py            # ✓ NEW
│   ├── run_ablation.py                      # ✓ NEW
│   ├── ablation_examples.py                 # ✓ NEW
│   ├── ABLATION_README.md                   # ✓ NEW
│   ├── IMPLEMENTATION_SUMMARY.md            # ✓ NEW
│   └── results/
│       └── ablation/                        # Will be created
├── data_preparation.py                       # ✓ Updated
└── requirements.txt
```

## Next Steps

1. **Review Documentation:**
   - Read `ABLATION_README.md` for detailed usage
   - Check `IMPLEMENTATION_SUMMARY.md` for technical details

2. **Run Examples:**
   ```bash
   python -m disagreement_model.ablation_examples
   ```

3. **Customize:**
   - Modify `ablation_config.py` to add feature groups
   - Adjust `model_config.py` for different settings

4. **Analyze Results:**
   - Open `ablation_comparison_*.csv` in Excel/pandas
   - Review visualizations in `results/ablation/visualizations/`
   - Read `ablation_report_*.txt` for detailed findings

## Getting Help

- **Configuration issues:** Check `model_config.py` settings
- **Experiment definitions:** Review `ablation_config.py`
- **Usage examples:** See `ablation_examples.py`
- **API reference:** Read docstrings in `ablation_study.py`
- **Visualization options:** Check `ablation_visualization.py`

## Success Indicators

You'll know everything is working if:

✅ `python -m disagreement_model.run_ablation --list-experiments` shows all experiments  
✅ Mini test run completes without errors  
✅ Output files are created in `results/ablation/`  
✅ Visualizations are generated successfully  
✅ Comparison CSV contains expected metrics  

---

**Ready to go! 🚀**

For detailed usage and API documentation, see `ABLATION_README.md`.
