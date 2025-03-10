# Uncertainty-Aware Genomic Classification of Alzheimer’s Disease

This repository demonstrates how to implement **TrUE-Net** (**Tr**ansformer-based, **U**ncertainty-aware **E**nsemble **Net**work) by combining a Transformer model with MC-Dropout and a RandomForest classifier. We use out-of-fold (OOF) predictions to derive an ensemble weight that maximizes AUC, and then apply variance thresholding to split samples into Uncertain and Certain groups.

## Folder Structure

```
TrUE-Net/
├── README.md
├── requirements.txt
├── data/
│   ├── APOE_50kb-1050.raw
│   └── DX-1050.txt
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── training.py
│   ├── evaluation.py
│   └── main.py
└── result/
```
- `dataset.py`: dataset utilities (`load_data`, `split_into_windows_as_sequence`, `SequenceSNPDataset`)
- `model.py`: `TransformerClassifier` with MC-Dropout
- `training.py`: functions to train/evaluate MC-Dropout
- `evaluation.py`: metrics, subgroup analysis, plotting
- `main.py`: main script, orchestrates cross-validation, alpha/threshold selection, and final testing

## Requirements

- Python 3.8+
- PyTorch >= 1.7
- scikit-learn
- pandas
- seaborn
- matplotlib

pip install -r requirements.txt

## Usage

python src/main.py [raw_file.csv] [dx_file.csv] [prefix]

Example:

python src/main.py data/APOE_50kb-1050.raw data/DX-1050.txt apoe-run

It will create outputs in the `result/` folder, such as:

- `_test_details.csv`, `_test_summary.csv`: final test metrics (All/Uncertain/Certain)
- `_analysis_plots.pdf`: final test plots

## License
This code is maintained by the **Taeho Jo AI Research Lab** at Indiana University School of Medicine.  
For more information, visit our lab website: [JoLab.AI](https://www.jolab.ai).

All Rights Reserved © 2025 Taeho Jo AI Research Lab 
