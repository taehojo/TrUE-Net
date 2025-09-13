# TrUE-Net: Uncertainty-Aware Genomic Classification of Alzheimer's Disease

[![Web Demo](https://img.shields.io/badge/Web%20Demo-Available-brightgreen)](https://www.jolab.ai/truenet)
[![Python](https://img.shields.io/badge/Python-3.12.9-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-orange)](https://pytorch.org/)

This repository contains the implementation of **TrUE-Net** (**Tr**ansformer-based, **U**ncertainty-aware **E**nsemble **Net**work), a deep learning framework that combines transformer models with Monte Carlo Dropout for uncertainty-aware genomic classification of Alzheimer's Disease.

## Live Demo

Try our interactive web demo: **[www.jolab.ai/truenet](https://www.jolab.ai/truenet)**

## Key Features

- **Uncertainty Quantification**: Monte Carlo Dropout for prediction reliability assessment
- **Selective Classification**: Identifies certain (24.6%) vs uncertain (75.4%) predictions
- **High Performance on Certain Subset**: 72.9% accuracy, F1-score 0.821
- **Comprehensive Baseline Comparison**: Validated against 11 ML models with statistical testing
- **Real Genomic Data**: APOE 50kb region (14,094 SNPs, 1,050 samples)

## Project Structure

```
TrUE-Net/
├── README.md
├── requirements.txt
├── CLAUDE.md                     # Project guidelines for AI assistance
├── data/
│   ├── APOE_50kb-1050.raw      # Genomic data (14,094 SNPs)
│   └── DX-1050.txt              # Diagnosis labels
├── src/
│   ├── dataset.py               # Data loading and preprocessing
│   ├── model.py                 # TransformerClassifier with MC-Dropout
│   ├── training.py              # Training utilities
│   ├── evaluation.py            # Metrics and uncertainty analysis
│   ├── main.py                  # Main experiment runner
│   ├── baseline_models.py       # 11 ML baseline implementations
│   ├── statistical_tests.py     # Bootstrap CI & McNemar tests
│   └── run_baseline_comparison.py # Baseline comparison script
├── result/
│   ├── demo_test_details.csv    # TrUE-Net predictions (525 samples)
│   ├── baseline/                # Baseline model predictions
│   └── statistical_validation.csv # Bootstrap confidence intervals
└── paper/
    └── Table2_Academic_Clean_NoAdaBoost.txt # Final results table
```

## Environment Setup

### System Requirements
- **CPU**: Intel Xeon or equivalent (128GB RAM recommended)
- **OS**: Linux/Unix environment preferred
- **Python**: 3.12.9

### Python Dependencies
```bash
# Core requirements
Python==3.12.9
PyTorch==2.7.1+cu126  # CPU version also available
scikit-learn==1.6.1
NumPy==1.26.4
pandas==2.2.3
matplotlib==3.10.0
seaborn==0.13.2
scipy==1.14.1
```

### Installation
```bash
# Clone repository
git clone https://github.com/taehojo/TrUE-Net.git
cd TrUE-Net

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Reproducing Results

### 1. Run TrUE-Net Experiment
```bash
python src/main.py data/APOE_50kb-1050.raw data/DX-1050.txt demo
```

This generates:
- `result/demo_test_details.csv`: Individual predictions with uncertainty
- `result/demo_test_summary.csv`: Performance metrics by subset
- `result/demo_analysis_plots.pdf`: Visualization plots

### 2. Run Baseline Comparison (11 ML Models)
```bash
python run_complete_ml_experiments.py
```

This generates:
- Complete comparison with 11 baseline models
- Bootstrap confidence intervals (1,000 iterations)
- McNemar's test p-values
- Table 2 for publication

### 3. Statistical Validation Only
```bash
python src/run_statistical_validation.py
```

## Baseline Models Included

**Ensemble Methods** (3):
- Gradient Boosting
- Random Forest
- XGBoost

**Support Vector Machines** (3):
- SVM-Linear
- SVM-RBF
- SVM-Polynomial

**Traditional Methods** (5):
- Logistic Regression
- K-Nearest Neighbors
- Naive Bayes
- Decision Tree
- Neural Network (MLP)

## Key Results

### McNemar's Test Results
TrUE-Net significantly outperforms (p<0.05):
- Decision Tree (p=0.021)
- Logistic Regression (p=0.002)
- Naive Bayes (p=0.007)
- SVM-Linear (p=0.002)
- SVM-Polynomial (p=0.015)

Comparable performance with:
- Gradient Boosting (p=0.614)
- Random Forest (p=0.193)
- XGBoost (p=0.113)

## Citation

If you use TrUE-Net in your research, please cite:

```bibtex
@article{truenet2025,
  title={Uncertainty-Aware Genomic Classification of Alzheimer's Disease Using Transformer-Based Deep Learning},
  author={Jo, Taeho and others},
  journal={Manuscript under review},
  year={2025}
}
```

## License
© Dr. Jo's Medical AI Research lab, IUSM | www.jolab.ai