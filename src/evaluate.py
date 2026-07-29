import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve, classification_report
)
import joblib

from src import config

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix', save_path=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_true, y_prob, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    ax.plot(fpr, tpr, label=f'AUC = {auc_score:.3f}')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_title('ROC Curve')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_precision_recall_curve(y_true, y_prob, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
    ax.plot(recall_vals, precision_vals)
    ax.set_title('Precision-Recall Curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def find_cost_optimal_threshold(y_true, y_prob, cost_fn=500, cost_fp=50):
    thresholds_to_test = np.arange(0.05, 0.95, 0.01)
    costs = []
    
    for thr in thresholds_to_test:
        y_pred_thr = (y_prob >= thr).astype(int)
        cm = confusion_matrix(y_true, y_pred_thr)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
        total_cost = (fn * cost_fn) + (fp * cost_fp)
        costs.append(total_cost)
        
    costs = np.array(costs)
    best_cost_idx = np.argmin(costs)
    best_cost_threshold = thresholds_to_test[best_cost_idx]
    
    return best_cost_threshold, costs[best_cost_idx]

def capacity_recall_analysis(y_true, y_prob, capacities=[200, 300, 500, 700, 1000]):
    risk_df = pd.DataFrame({
        'true_label': y_true,
        'churn_probability': y_prob
    }).sort_values('churn_probability', ascending=False).reset_index(drop=True)
    
    total_churners = y_true.sum()
    capacity_results = []
    
    for cap in capacities:
        top_k_temp = risk_df.head(cap)
        captured = top_k_temp['true_label'].sum()
        capacity_results.append({
            'Capacity': cap,
            'Churners_Captured': int(captured),
            'Total_Churners': int(total_churners),
            'Recall@K': captured / total_churners if total_churners > 0 else 0,
            'Precision@K': captured / cap if cap > 0 else 0
        })
        
    return pd.DataFrame(capacity_results)

def evaluate_model():
    print("--- Evaluating Production Model ---")
    
    # Load model and data
    import tensorflow as tf
    model = tf.keras.models.load_model(config.PRODUCTION_MODEL_PATH)
    
    X_test = pd.read_csv(config.X_TEST_PATH)
    y_test = pd.read_csv(config.Y_TEST_PATH)
    y_true = y_test.values.ravel()
    
    # X_test_processed.csv was produced by notebook-04 which applied StandardScaler
    # before saving — the CSV is already scaled. Do NOT re-scale here.
    y_prob = model.predict(X_test, verbose=0).ravel()
    
    # Cost optimal threshold
    best_threshold, min_cost = find_cost_optimal_threshold(y_true, y_prob)
    print(f"Optimal Threshold: {best_threshold:.3f}")
    joblib.dump(best_threshold, config.BEST_THRESHOLD_PATH)
    
    y_pred = (y_prob >= best_threshold).astype(int)
    
    # Metrics
    metrics_data = [{
        'Model': 'Production Model',
        'Threshold': best_threshold,
        'Accuracy': np.mean(y_true == y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1_Score': f1_score(y_true, y_pred),
        'ROC_AUC': roc_auc_score(y_true, y_prob),
        'Expected_Cost': min_cost
    }]
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(config.METRICS_CSV_PATH, index=False)
    print(f"Metrics saved to {config.METRICS_CSV_PATH}")
    
    # Confusion Matrix CSV
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, columns=['Pred_Retained', 'Pred_Churned'], index=['Actual_Retained', 'Actual_Churned'])
    cm_df.to_csv(config.CONFUSION_MATRIX_CSV_PATH)
    print(f"Confusion matrix saved to {config.CONFUSION_MATRIX_CSV_PATH}")
    
    # Plots
    plot_confusion_matrix(y_true, y_pred, title=f'Confusion Matrix @ {best_threshold:.2f}', 
                          save_path=os.path.join(config.FIGURES_DIR, 'confusion_matrix.png'))
    plot_roc_curve(y_true, y_prob, save_path=os.path.join(config.FIGURES_DIR, 'roc_curve.png'))
    plot_precision_recall_curve(y_true, y_prob, save_path=os.path.join(config.FIGURES_DIR, 'precision_recall_curve.png'))
    
    # Capacity Analysis
    capacity_df = capacity_recall_analysis(y_true, y_prob)
    capacity_csv_path = os.path.join(config.REPORTS_DIR, 'capacity_analysis.csv')
    capacity_df.to_csv(capacity_csv_path, index=False)
    print(f"Capacity analysis saved to {capacity_csv_path}")

if __name__ == "__main__":
    evaluate_model()
