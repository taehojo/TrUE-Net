"""
Baseline models for comparison with TrUE-Net
Response to Reviewer Comment #10
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score, 
                           precision_score, recall_score, confusion_matrix)
from sklearn.model_selection import cross_val_predict
import joblib
import os
from datetime import datetime

def train_baseline_models(X_train, y_train, X_test, y_test, output_dir='result/baseline/'):
    """
    Train and evaluate baseline models (Logistic Regression, SVM)
    
    Args:
        X_train: Training features (flattened)
        y_train: Training labels
        X_test: Test features (flattened)
        y_test: Test labels
        output_dir: Directory to save results
    
    Returns:
        Dictionary with results for each model
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Standardize features for linear models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler for reproducibility
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Logistic Regression
    print("\n[INFO] Training Logistic Regression...")
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced',  # Handle class imbalance
        solver='saga',  # Good for large datasets
        n_jobs=-1
    )
    lr_model.fit(X_train_scaled, y_train)
    
    # Predictions
    lr_pred = lr_model.predict(X_test_scaled)
    lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    results['LogisticRegression'] = {
        'accuracy': accuracy_score(y_test, lr_pred),
        'auc': roc_auc_score(y_test, lr_prob),
        'f1': f1_score(y_test, lr_pred),
        'precision': precision_score(y_test, lr_pred),
        'recall': recall_score(y_test, lr_pred),
        'predictions': lr_pred,
        'probabilities': lr_prob,
        'confusion_matrix': confusion_matrix(y_test, lr_pred)
    }
    
    print(f"  Accuracy: {results['LogisticRegression']['accuracy']:.4f}")
    print(f"  AUC: {results['LogisticRegression']['auc']:.4f}")
    print(f"  F1: {results['LogisticRegression']['f1']:.4f}")
    
    # Save model
    joblib.dump(lr_model, os.path.join(output_dir, f'lr_model_{timestamp}.pkl'))
    
    # 2. SVM with Linear Kernel
    print("\n[INFO] Training SVM (Linear Kernel)...")
    svm_linear = SVC(
        kernel='linear',
        probability=True,
        random_state=42,
        class_weight='balanced',
        max_iter=1000
    )
    svm_linear.fit(X_train_scaled, y_train)
    
    # Predictions
    svm_linear_pred = svm_linear.predict(X_test_scaled)
    svm_linear_prob = svm_linear.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    results['SVM_Linear'] = {
        'accuracy': accuracy_score(y_test, svm_linear_pred),
        'auc': roc_auc_score(y_test, svm_linear_prob),
        'f1': f1_score(y_test, svm_linear_pred),
        'precision': precision_score(y_test, svm_linear_pred),
        'recall': recall_score(y_test, svm_linear_pred),
        'predictions': svm_linear_pred,
        'probabilities': svm_linear_prob,
        'confusion_matrix': confusion_matrix(y_test, svm_linear_pred)
    }
    
    print(f"  Accuracy: {results['SVM_Linear']['accuracy']:.4f}")
    print(f"  AUC: {results['SVM_Linear']['auc']:.4f}")
    print(f"  F1: {results['SVM_Linear']['f1']:.4f}")
    
    # Save model
    joblib.dump(svm_linear, os.path.join(output_dir, f'svm_linear_{timestamp}.pkl'))
    
    # 3. SVM with RBF Kernel
    print("\n[INFO] Training SVM (RBF Kernel)...")
    svm_rbf = SVC(
        kernel='rbf',
        probability=True,
        random_state=42,
        class_weight='balanced',
        max_iter=1000
    )
    svm_rbf.fit(X_train_scaled, y_train)
    
    # Predictions
    svm_rbf_pred = svm_rbf.predict(X_test_scaled)
    svm_rbf_prob = svm_rbf.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    results['SVM_RBF'] = {
        'accuracy': accuracy_score(y_test, svm_rbf_pred),
        'auc': roc_auc_score(y_test, svm_rbf_prob),
        'f1': f1_score(y_test, svm_rbf_pred),
        'precision': precision_score(y_test, svm_rbf_pred),
        'recall': recall_score(y_test, svm_rbf_pred),
        'predictions': svm_rbf_pred,
        'probabilities': svm_rbf_prob,
        'confusion_matrix': confusion_matrix(y_test, svm_rbf_pred)
    }
    
    print(f"  Accuracy: {results['SVM_RBF']['accuracy']:.4f}")
    print(f"  AUC: {results['SVM_RBF']['auc']:.4f}")
    print(f"  F1: {results['SVM_RBF']['f1']:.4f}")
    
    # Save model
    joblib.dump(svm_rbf, os.path.join(output_dir, f'svm_rbf_{timestamp}.pkl'))
    
    # Save results to CSV
    save_baseline_results(results, output_dir, timestamp)
    
    return results

def save_baseline_results(results, output_dir, timestamp):
    """Save baseline results to CSV for documentation"""
    
    # Create summary dataframe
    summary_data = []
    for model_name, metrics in results.items():
        summary_data.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'AUC': metrics['auc'],
            'F1': metrics['f1'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall']
        })
    
    df_summary = pd.DataFrame(summary_data)
    
    # Save summary
    summary_file = os.path.join(output_dir, f'baseline_summary_{timestamp}.csv')
    df_summary.to_csv(summary_file, index=False)
    print(f"\n[INFO] Results saved to {summary_file}")
    
    # Save detailed predictions for each model
    for model_name, metrics in results.items():
        detail_data = pd.DataFrame({
            'prediction': metrics['predictions'],
            'probability': metrics['probabilities']
        })
        detail_file = os.path.join(output_dir, f'{model_name}_predictions_{timestamp}.csv')
        detail_data.to_csv(detail_file, index=False)
    
    # Print comparison table
    print("\n" + "="*70)
    print("BASELINE MODEL COMPARISON")
    print("="*70)
    print(df_summary.to_string(index=False))
    print("="*70)
    
    return df_summary

def compare_with_truenet(baseline_results, truenet_results_file='result/apoe-run_test_summary.csv'):
    """
    Compare baseline results with TrUE-Net results
    
    Args:
        baseline_results: Dictionary of baseline model results
        truenet_results_file: Path to TrUE-Net results CSV
    """
    
    print("\n" + "="*70)
    print("COMPARISON WITH TrUE-Net")
    print("="*70)
    
    # Check if TrUE-Net results exist
    if os.path.exists(truenet_results_file):
        truenet_df = pd.read_csv(truenet_results_file)
        print(f"\nTrUE-Net Results (from {truenet_results_file}):")
        print(truenet_df.to_string(index=False))
    else:
        print(f"\n[WARNING] TrUE-Net results file not found: {truenet_results_file}")
        print("Using hardcoded values from paper:")
        print("  All:       Accuracy=0.6514, AUC=0.6636, F1=0.6679")
        print("  Certain:   Accuracy=0.7287, AUC=0.6816, F1=0.8205")
        print("  Uncertain: Accuracy=0.6263, AUC=0.6268, F1=0.5843")
    
    print("\nBaseline Models:")
    for model_name, metrics in baseline_results.items():
        print(f"  {model_name:15} Accuracy={metrics['accuracy']:.4f}, "
              f"AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}")
    
    print("="*70)

if __name__ == "__main__":
    print("This module should be imported and used with main.py")
    print("Run: python src/run_baseline_comparison.py")