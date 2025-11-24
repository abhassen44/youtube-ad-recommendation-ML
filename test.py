import pandas as pd
import joblib
import os
import numpy as np
from sklearn.metrics import (
    roc_auc_score, 
    classification_report, 
    confusion_matrix, 
    RocCurveDisplay,
    precision_recall_curve,
    log_loss
)
import matplotlib.pyplot as plt
import seaborn as sns

# Create output folder
if not os.path.exists('output'):
    os.makedirs('output')
    print("Created 'output' folder\n")

plt.style.use('seaborn-v0_8-darkgrid')

# --- 1. LOAD MODEL AND HASHER ---
print("Loading model and hasher...")
model = joblib.load('ctr_model.joblib')
hasher = joblib.load('ctr_hasher.joblib')

# --- 2. LOAD A FRESH CHUNK OF DATA FOR VALIDATION ---
# We read the first 100,000 rows as a test set
file_path = 'train.csv'
test_df = pd.read_csv(file_path, nrows=100000)

print("Preparing test data...")
y_true = test_df['click']

# Prepare features for hashing
categorical_cols = [
    'hour', 'C1', 'banner_pos', 'site_id', 'site_domain',
    'site_category', 'app_id', 'app_domain', 'app_category',
    'device_id', 'device_ip', 'device_model', 'device_type'
]
X_test_df = test_df[categorical_cols].astype(str)
X_test_dict = X_test_df.to_dict('records')

# Hash the test data
X_test_hashed = hasher.transform(X_test_dict)

# --- 3. GET PREDICTIONS ---
print("Getting model predictions...")
y_pred_proba = model.predict_proba(X_test_hashed)[:, 1] # For ROC Curve
y_pred_class = model.predict(X_test_hashed)           # For Confusion Matrix

# --- 4. CALCULATE METRICS ---
print("\n=== CALCULATING METRICS ===")
auc = roc_auc_score(y_true, y_pred_proba)
logloss = log_loss(y_true, y_pred_proba)
cm = confusion_matrix(y_true, y_pred_class)

print(f"ROC-AUC Score: {auc:.4f}")
print(f"Log Loss: {logloss:.4f}")
print(f"\nConfusion Matrix:\n{cm}")

# --- 5. SAVE RESULTS TO CSV ---
print("\n=== SAVING RESULTS TO CSV ===")

# 5.1 Model Performance Metrics
metrics_df = pd.DataFrame({
    'Metric': ['ROC-AUC', 'Log Loss', 'Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Value': [
        auc,
        logloss,
        (cm[0,0] + cm[1,1]) / cm.sum(),
        cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0,
        cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0,
        2 * (cm[1,1] / (cm[1,1] + cm[0,1])) * (cm[1,1] / (cm[1,1] + cm[1,0])) / 
        ((cm[1,1] / (cm[1,1] + cm[0,1])) + (cm[1,1] / (cm[1,1] + cm[1,0]))) 
        if (cm[1,1] + cm[0,1]) > 0 and (cm[1,1] + cm[1,0]) > 0 else 0
    ]
})
metrics_df.to_csv('output/model_metrics.csv', index=False)
print("✓ Saved: output/model_metrics.csv")

# 5.2 Confusion Matrix
cm_df = pd.DataFrame(cm, 
    columns=['Predicted No Click', 'Predicted Click'],
    index=['Actual No Click', 'Actual Click']
)
cm_df.to_csv('output/confusion_matrix.csv')
print("✓ Saved: output/confusion_matrix.csv")

# 5.3 Predictions with Probabilities
predictions_df = test_df[['id', 'click']].copy()
predictions_df['predicted_probability'] = y_pred_proba
predictions_df['predicted_class'] = y_pred_class
predictions_df['correct'] = (predictions_df['click'] == predictions_df['predicted_class'])
predictions_df.to_csv('output/predictions.csv', index=False)
print("✓ Saved: output/predictions.csv")

# 5.4 CTR Analysis by Features
ctr_analysis = []
for col in ['device_type', 'banner_pos', 'C1']:
    if col in test_df.columns:
        analysis = test_df.groupby(col).agg({
            'click': ['count', 'mean']
        }).round(4)
        analysis.columns = ['count', 'actual_ctr']
        analysis['feature'] = col
        analysis['value'] = analysis.index
        analysis = analysis.reset_index(drop=True)
        ctr_analysis.append(analysis[['feature', 'value', 'count', 'actual_ctr']])

if ctr_analysis:
    ctr_df = pd.concat(ctr_analysis, ignore_index=True)
    ctr_df.to_csv('output/ctr_by_features.csv', index=False)
    print("✓ Saved: output/ctr_by_features.csv")

# 5.5 Probability Distribution
prob_bins = pd.cut(y_pred_proba, bins=10)
prob_dist = pd.DataFrame({
    'probability_range': prob_bins.value_counts().sort_index().index.astype(str),
    'count': prob_bins.value_counts().sort_index().values
})
prob_dist.to_csv('output/probability_distribution.csv', index=False)
print("✓ Saved: output/probability_distribution.csv")


# --- 6. GENERATE VISUALIZATIONS ---
print("\n=== GENERATING VISUALIZATIONS ===")

# Plot 1: ROC Curve
RocCurveDisplay.from_predictions(y_true, y_pred_proba)
plt.title(f'ROC Curve (AUC = {auc:.4f})', fontsize=14, fontweight='bold')
plt.plot([0, 1], [0, 1], 'r--', label='Random Guess (AUC=0.5)')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.savefig('output/1_roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: output/1_roc_curve.png")

# Plot 2: Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
    xticklabels=['Predicted No Click', 'Predicted Click'],
    yticklabels=['Actual No Click', 'Actual Click'],
    cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('Actual Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.savefig('output/2_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: output/2_confusion_matrix.png")

# Plot 3: Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, 'b-', linewidth=2)
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.savefig('output/3_precision_recall_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: output/3_precision_recall_curve.png")

# Plot 4: Probability Distribution
plt.figure(figsize=(10, 6))
plt.hist(y_pred_proba, bins=50, edgecolor='black', alpha=0.7, color='green')
plt.xlabel('Predicted Click Probability', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Predicted Probabilities', fontsize=14, fontweight='bold')
plt.axvline(y_pred_proba.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean = {y_pred_proba.mean():.4f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('output/4_probability_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: output/4_probability_distribution.png")

# Plot 5: CTR by Device Type
if 'device_type' in test_df.columns:
    plt.figure(figsize=(10, 6))
    device_ctr = test_df.groupby('device_type')['click'].agg(['count', 'mean'])
    device_ctr['mean'].plot(kind='bar', color='green', edgecolor='black')
    plt.xlabel('Device Type', fontsize=12)
    plt.ylabel('Click-Through Rate', fontsize=12)
    plt.title('CTR by Device Type', fontsize=14, fontweight='bold')
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig('output/5_ctr_by_device.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: output/5_ctr_by_device.png")

# Plot 6: CTR by Banner Position
if 'banner_pos' in test_df.columns:
    plt.figure(figsize=(10, 6))
    banner_ctr = test_df.groupby('banner_pos')['click'].agg(['count', 'mean'])
    banner_ctr['mean'].plot(kind='bar', color='darkgreen', edgecolor='black')
    plt.xlabel('Banner Position', fontsize=12)
    plt.ylabel('Click-Through Rate', fontsize=12)
    plt.title('CTR by Banner Position', fontsize=14, fontweight='bold')
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig('output/6_ctr_by_banner.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: output/6_ctr_by_banner.png")

# Plot 7: Model Performance Metrics
plt.figure(figsize=(10, 6))
metrics_plot = metrics_df[metrics_df['Metric'] != 'Log Loss'].copy()
metrics_plot['Value'] = metrics_plot['Value'] * 100
plt.barh(metrics_plot['Metric'], metrics_plot['Value'], color='green', edgecolor='black')
plt.xlabel('Score (%)', fontsize=12)
plt.title('Model Performance Metrics', fontsize=14, fontweight='bold')
plt.xlim(0, 100)
for i, v in enumerate(metrics_plot['Value']):
    plt.text(v + 1, i, f'{v:.2f}%', va='center', fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')
plt.savefig('output/7_performance_metrics.png', dpi=300, bbox_inches='tight')
plt.close()
plt.close()
print("✓ Saved: output/7_performance_metrics.png")

print("\n" + "="*50)
print("✓ REPORT GENERATION COMPLETE")
print("="*50)
print("\nGenerated Files:")
print("  CSV Files: 5")
print("  Visualizations: 7")
print("\nCheck 'output' folder for all results!")
print("\nReport Covers:")
print("  1. ✓ Algorithm Details (SGD + Feature Hashing)")
print("  2. ✓ Results & Analysis (Metrics + CSVs)")
print("  3. ✓ Visualizations (7 plots)")
print("  4. ✓ Interpretation (CTR analysis by features)")
print("="*50)