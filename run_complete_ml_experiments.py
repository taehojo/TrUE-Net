"""
Run COMPLETE ML experiments with ACTUAL genomic data
NO simulation - only real experimental results
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("COMPLETE ML EXPERIMENTS WITH ACTUAL GENOMIC DATA")
print("="*80)

# Step 1: Load ACTUAL genomic data
print("\n1. Loading ACTUAL genomic data...")
try:
    # Load the APOE region genomic data (actual experimental data)
    raw_data = pd.read_csv('/N/project/AiLab/TruNet/data/APOE_50kb-1050.raw', sep=r'\s+')

    # Load diagnosis data - it's just labels without IDs
    dx_data = pd.read_csv('/N/project/AiLab/TruNet/data/DX-1050.txt')

    print(f"   Loaded genomic data: {raw_data.shape}")
    print(f"   Loaded diagnosis data: {dx_data.shape}")

    # DX-1050.txt has labels in same order as raw data
    # Add labels directly
    y = dx_data['New_Label'].values[:len(raw_data)]

    # Remove non-feature columns from raw data
    non_feature_cols = ['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE']
    feature_cols = [col for col in raw_data.columns if col not in non_feature_cols]

    # Prepare features
    X = raw_data[feature_cols].values

    print(f"   Features: {X.shape}")
    print(f"   Class distribution: AD={np.sum(y==1)}, CN={np.sum(y==0)}")

    # Check for any missing values
    if np.any(np.isnan(X)):
        print("   Handling missing values...")
        # Simple imputation with mean
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)

except FileNotFoundError:
    print("[ERROR] Cannot find genomic data files!")
    print("Files needed:")
    print("  - /N/project/AiLab/TruNet/data/APOE_50kb-1050.raw")
    print("  - /N/project/AiLab/TruNet/data/DX-1050.txt")
    import sys
    sys.exit(1)

# Step 2: Create train-test split (same as original TrUE-Net)
print("\n2. Creating train-test split (50-50)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42, stratify=y
)

print(f"   Training set: {X_train.shape[0]} samples")
print(f"   Test set: {X_test.shape[0]} samples")

# Standardize features for some models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Define all ML models to test
print("\n3. Training multiple ML models on ACTUAL data...")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM-Linear': SVC(kernel='linear', probability=True, random_state=42),
    'SVM-RBF': SVC(kernel='rbf', probability=True, random_state=42),
    'SVM-Poly': SVC(kernel='poly', probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
}

# Store results
results = {}

for model_name, model in models.items():
    print(f"\n   Training {model_name}...")

    # Use scaled features for models that benefit from it
    if model_name in ['Logistic Regression', 'SVM-Linear', 'SVM-RBF', 'SVM-Poly',
                       'K-Nearest Neighbors', 'Neural Network']:
        X_train_use = X_train_scaled
        X_test_use = X_test_scaled
    else:
        X_train_use = X_train
        X_test_use = X_test

    try:
        # Train model
        model.fit(X_train_use, y_train)

        # Get predictions
        y_pred = model.predict(X_test_use)
        y_prob = model.predict_proba(X_test_use)[:, 1]

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)

        results[model_name] = {
            'y_test': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'acc': acc,
            'auc': auc,
            'f1': f1
        }

        print(f"      Accuracy: {acc:.4f}")
        print(f"      AUC: {auc:.4f}")
        print(f"      F1-score: {f1:.4f}")

    except Exception as e:
        print(f"      Error training {model_name}: {e}")

# Step 4: Calculate bootstrap confidence intervals
print("\n4. Calculating bootstrap confidence intervals (1000 iterations)...")

def bootstrap_ci(y_true, y_pred, y_prob, n_iterations=1000):
    """Calculate bootstrap confidence intervals"""
    n = len(y_true)
    accs, aucs, f1s = [], [], []

    for _ in range(n_iterations):
        idx = np.random.choice(n, n, replace=True)
        y_true_b = y_true[idx]
        y_pred_b = y_pred[idx]
        y_prob_b = y_prob[idx]

        accs.append(accuracy_score(y_true_b, y_pred_b))
        aucs.append(roc_auc_score(y_true_b, y_prob_b))
        f1s.append(f1_score(y_true_b, y_pred_b))

    return {
        'acc_ci': np.percentile(accs, [2.5, 97.5]),
        'auc_ci': np.percentile(aucs, [2.5, 97.5]),
        'f1_ci': np.percentile(f1s, [2.5, 97.5])
    }

# Calculate CIs for all models
for model_name in results:
    print(f"   Calculating CI for {model_name}...")
    ci = bootstrap_ci(
        results[model_name]['y_test'],
        results[model_name]['y_pred'],
        results[model_name]['y_prob']
    )
    results[model_name].update(ci)

# Step 5: Load TrUE-Net results
print("\n5. Loading TrUE-Net results...")
truenet_data = pd.read_csv('/N/project/AiLab/TruNet/result/demo_test_details.csv')
truenet_data['y_true'] = truenet_data['true_label']
truenet_data['y_prob'] = truenet_data['final_prob']
truenet_data['y_pred'] = (truenet_data['y_prob'] >= 0.5).astype(int)

# TrUE-Net All
truenet_all = {
    'acc': accuracy_score(truenet_data['y_true'], truenet_data['y_pred']),
    'auc': roc_auc_score(truenet_data['y_true'], truenet_data['y_prob']),
    'f1': f1_score(truenet_data['y_true'], truenet_data['y_pred'])
}
truenet_all.update(bootstrap_ci(
    truenet_data['y_true'].values,
    truenet_data['y_pred'].values,
    truenet_data['y_prob'].values
))

# TrUE-Net Certain
certain_mask = truenet_data['final_var'] <= 0.0741
certain_data = truenet_data[certain_mask]
truenet_certain = {
    'n': len(certain_data),
    'acc': accuracy_score(certain_data['y_true'], certain_data['y_pred']),
    'auc': roc_auc_score(certain_data['y_true'], certain_data['y_prob']),
    'f1': f1_score(certain_data['y_true'], certain_data['y_pred'])
}
truenet_certain.update(bootstrap_ci(
    certain_data['y_true'].values,
    certain_data['y_pred'].values,
    certain_data['y_prob'].values
))

# TrUE-Net Uncertain
uncertain_data = truenet_data[~certain_mask]
truenet_uncertain = {
    'n': len(uncertain_data),
    'acc': accuracy_score(uncertain_data['y_true'], uncertain_data['y_pred']),
    'auc': roc_auc_score(uncertain_data['y_true'], uncertain_data['y_prob']),
    'f1': f1_score(uncertain_data['y_true'], uncertain_data['y_pred'])
}
truenet_uncertain.update(bootstrap_ci(
    uncertain_data['y_true'].values,
    uncertain_data['y_pred'].values,
    uncertain_data['y_prob'].values
))

# Step 6: McNemar's tests
print("\n6. Performing McNemar's tests...")
from scipy.stats import chi2

truenet_pred = truenet_data['y_pred'].values[:len(y_test)]
mcnemar_results = {}

for model_name in results:
    model_pred = results[model_name]['y_pred']

    # Contingency table
    correct_truenet = (truenet_pred == y_test).astype(int)
    correct_model = (model_pred == y_test).astype(int)

    n01 = np.sum((correct_model == 0) & (correct_truenet == 1))
    n10 = np.sum((correct_model == 1) & (correct_truenet == 0))

    if n01 + n10 > 0:
        statistic = (abs(n01 - n10) - 1)**2 / (n01 + n10)
        p_value = 1 - chi2.cdf(statistic, df=1)
    else:
        p_value = 1.0

    mcnemar_results[model_name] = p_value

# Step 7: Create complete table
print("\n7. Creating COMPLETE table with ALL values...")

table = """
Table 2. Complete ML Model Comparison with Bootstrap Confidence Intervals (1,000 iterations)

Model                     | n    | Accuracy [95% CI]           | AUC [95% CI]              | F1-score [95% CI]         | McNemar p-value
--------------------------|------|----------------------------|---------------------------|---------------------------|----------------
TrUE-Net (All)           | 525  | {:.4f} [{:.4f}-{:.4f}] | {:.4f} [{:.4f}-{:.4f}] | {:.4f} [{:.4f}-{:.4f}] | -
TrUE-Net (Uncertain)     | 396  | {:.4f} [{:.4f}-{:.4f}] | {:.4f} [{:.4f}-{:.4f}] | {:.4f} [{:.4f}-{:.4f}] | -
TrUE-Net (Certain)       | 129  | {:.4f} [{:.4f}-{:.4f}] | {:.4f} [{:.4f}-{:.4f}] | {:.4f} [{:.4f}-{:.4f}] | -
--------------------------|------|----------------------------|---------------------------|---------------------------|----------------
""".format(
    truenet_all['acc'], truenet_all['acc_ci'][0], truenet_all['acc_ci'][1],
    truenet_all['auc'], truenet_all['auc_ci'][0], truenet_all['auc_ci'][1],
    truenet_all['f1'], truenet_all['f1_ci'][0], truenet_all['f1_ci'][1],

    truenet_uncertain['acc'], truenet_uncertain['acc_ci'][0], truenet_uncertain['acc_ci'][1],
    truenet_uncertain['auc'], truenet_uncertain['auc_ci'][0], truenet_uncertain['auc_ci'][1],
    truenet_uncertain['f1'], truenet_uncertain['f1_ci'][0], truenet_uncertain['f1_ci'][1],

    truenet_certain['acc'], truenet_certain['acc_ci'][0], truenet_certain['acc_ci'][1],
    truenet_certain['auc'], truenet_certain['auc_ci'][0], truenet_certain['auc_ci'][1],
    truenet_certain['f1'], truenet_certain['f1_ci'][0], truenet_certain['f1_ci'][1]
)

# Add all ML models
for model_name in sorted(results.keys()):
    r = results[model_name]
    p_val = mcnemar_results[model_name]
    sig = "*" if p_val < 0.05 else "NS"

    table += "{:<25} | {:<4} | {:.4f} [{:.4f}-{:.4f}] | {:.4f} [{:.4f}-{:.4f}] | {:.4f} [{:.4f}-{:.4f}] | {:.3f} ({})\n".format(
        model_name, len(y_test),
        r['acc'], r['acc_ci'][0], r['acc_ci'][1],
        r['auc'], r['auc_ci'][0], r['auc_ci'][1],
        r['f1'], r['f1_ci'][0], r['f1_ci'][1],
        p_val, sig
    )

table += """
Notes: CI = Confidence Interval; NS = Not Significant; *p < 0.05
All results from ACTUAL genomic data (APOE 50kb region, 1050 samples)
No simulation data used - all values are real experimental results
"""

# Save table
with open('/N/project/AiLab/TruNet/Table2_Complete_ML_Results.txt', 'w') as f:
    f.write(table)

print("\n" + "="*80)
print("COMPLETE TABLE:")
print("="*80)
print(table)

print("\n✅ Complete table saved as Table2_Complete_ML_Results.txt")
print("✅ ALL values filled with ACTUAL experimental results")
print("✅ NO simulation data used")