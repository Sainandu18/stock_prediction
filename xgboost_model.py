import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                            roc_curve, precision_recall_curve, f1_score, accuracy_score)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("XGBOOST MODEL - STOCK PRICE PREDICTION")
print("="*70)

# ==================== STEP 1: LOAD DATA ====================
print("\n[STEP 1] Loading Data...")
df = pd.read_csv('apple_stock_complete.csv')
print(f"✓ Data loaded: {df.shape}")

# ==================== STEP 2: PREPARE DATA ====================
print("\n[STEP 2] Preparing Data...")

# Drop non-feature columns
drop_cols = ['Date', 'Price_Tomorrow', 'Price_Change', 'Price_Change_Pct']
X = df.drop(columns=drop_cols + ['Target'])
y = df['Target']

print(f"Features: {X.shape[1]}")
print(f"Samples: {X.shape[0]}")
print(f"Target distribution - Up: {y.sum()}, Down: {(1-y).sum()}")

# ==================== STEP 3: SPLIT DATA ====================
print("\n[STEP 3] Splitting Data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Testing set: {X_test.shape}")

# ==================== STEP 4: SCALE FEATURES ====================
print("\n[STEP 4] Scaling Features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

print("✓ Features scaled successfully")

# ==================== STEP 5: HYPERPARAMETER TUNING ====================
print("\n[STEP 5] Hyperparameter Tuning...")
print("Testing different hyperparameters...\n")

# Calculate scale_pos_weight for imbalanced data
scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])

# Define parameter grid
param_grid = {
    'max_depth': [4, 5, 6, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

# Create base model
xgb_base = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# Grid search (using subset for speed)
param_grid_subset = {
    'max_depth': [5, 6, 7],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 0.9]
}

grid_search = GridSearchCV(
    xgb_base, 
    param_grid_subset, 
    cv=5, 
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

print(f"\n✓ Best parameters: {grid_search.best_params_}")
print(f"✓ Best cross-validation score: {grid_search.best_score_:.4f}")

# ==================== STEP 6: TRAIN FINAL MODEL ====================
print("\n[STEP 6] Training Final Model...")

xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=grid_search.best_params_['max_depth'],
    learning_rate=grid_search.best_params_['learning_rate'],
    subsample=grid_search.best_params_['subsample'],
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    verbose=0
)

# Train with early stopping
xgb_model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_test_scaled, y_test)],
    verbose=False
)

print("✓ Model trained successfully")

# ==================== STEP 7: PREDICTIONS ====================
print("\n[STEP 7] Making Predictions...")

y_pred_train = xgb_model.predict(X_train_scaled)
y_pred_test = xgb_model.predict(X_test_scaled)
y_pred_proba_train = xgb_model.predict_proba(X_train_scaled)[:, 1]
y_pred_proba_test = xgb_model.predict_proba(X_test_scaled)[:, 1]

print("✓ Predictions completed")

# ==================== STEP 8: EVALUATION ====================
print("\n[STEP 8] Model Evaluation\n")

print("="*70)
print("TRAINING SET METRICS")
print("="*70)
print(f"Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_train, y_pred_proba_train):.4f}")
print(f"F1 Score: {f1_score(y_train, y_pred_train):.4f}")
print(f"\nClassification Report:\n{classification_report(y_train, y_pred_train)}")

print("\n" + "="*70)
print("TESTING SET METRICS")
print("="*70)
print(f"Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba_test):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_test):.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred_test)}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test)
print(f"\nConfusion Matrix:\n{cm}")

# ==================== STEP 9: VISUALIZATIONS ====================
print("\n[STEP 9] Creating Evaluation Visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('XGBoost Model Evaluation', fontsize=16, fontweight='bold')

# 1. Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
axes[0, 0].set_title('Confusion Matrix', fontweight='bold')
axes[0, 0].set_ylabel('True Label')
axes[0, 0].set_xlabel('Predicted Label')

# 2. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_test)
auc = roc_auc_score(y_test, y_pred_proba_test)
axes[0, 1].plot(fpr, tpr, label=f'ROC Curve (AUC={auc:.4f})', linewidth=2, color='blue')
axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
axes[0, 1].set_title('ROC Curve', fontweight='bold')
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba_test)
axes[1, 0].plot(recall, precision, linewidth=2, color='green')
axes[1, 0].set_title('Precision-Recall Curve', fontweight='bold')
axes[1, 0].set_xlabel('Recall')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].grid(True, alpha=0.3)

# 4. Feature Importance
feature_importance = xgb_model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False).head(15)

axes[1, 1].barh(importance_df['Feature'], importance_df['Importance'], color='orange')
axes[1, 1].set_title('Top 15 Feature Importance', fontweight='bold')
axes[1, 1].set_xlabel('Importance')

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
print("✓ Evaluation plots saved as 'model_evaluation.png'")
plt.show()

# ==================== STEP 10: FEATURE IMPORTANCE ====================
print("\n[STEP 10] Feature Importance\n")
importance_df_full = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print(importance_df_full.head(20))

# ==================== STEP 11: SAVE MODEL ====================
print("\n[STEP 11] Saving Model...")
xgb_model.save_model('xgboost_model.json')
print("✓ Model saved as 'xgboost_model.json'")

# Save scaler
import joblib
joblib.dump(scaler, 'scaler.pkl')
print("✓ Scaler saved as 'scaler.pkl'")

# ==================== STEP 12: SUMMARY ====================
print("\n" + "="*70)
print("MODEL SUMMARY")
print("="*70)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
print(f"Test ROC-AUC: {roc_auc_score(y_test, y_pred_proba_test):.4f}")
print(f"Test F1-Score: {f1_score(y_test, y_pred_test):.4f}")
print(f"\nModel Parameters:")
print(f"  • n_estimators: {xgb_model.n_estimators}")
print(f"  • max_depth: {xgb_model.max_depth}")
print(f"  • learning_rate: {xgb_model.learning_rate}")
print(f"  • subsample: {xgb_model.subsample}")
print(f"  • colsample_bytree: {xgb_model.colsample_bytree}")
print("="*70)