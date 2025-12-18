# task6_modeling.py
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, RocCurveDisplay
)
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load Task 4 Features
# -----------------------------
processed_path = r"C:\Users\assef\Desktop\Kifiya AI Mastery\week4\credit-risk-model\data\processed"
task4_csv_path = os.path.join(processed_path, "task4_features.csv")
df = pd.read_csv(task4_csv_path)

# Quick preview
print("Columns in dataset:", df.columns.tolist())
print(df.head())

# -----------------------------
# Define Features and Target
# -----------------------------
X = df.drop(columns=['CustomerId', 'is_high_risk'])
y = df['is_high_risk']

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# -----------------------------
# Scaling (Optional)
# -----------------------------
numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# -----------------------------
# Model 1: Logistic Regression
# -----------------------------
logreg = LogisticRegression(random_state=42)
logreg.fit(X_train, y_train)
y_pred_lr = logreg.predict(X_test)
y_proba_lr = logreg.predict_proba(X_test)[:, 1]

# -----------------------------
# Model 2: Random Forest
# -----------------------------
rf = RandomForestClassifier(random_state=42)
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
}
grid_rf = GridSearchCV(rf, rf_params, cv=3, scoring='roc_auc')
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)
y_proba_rf = best_rf.predict_proba(X_test)[:, 1]

# -----------------------------
# Evaluation Metrics Table
# -----------------------------
def evaluate_model(y_true, y_pred, y_proba):
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, y_proba)
    }
    return metrics

metrics_lr = evaluate_model(y_test, y_pred_lr, y_proba_lr)
metrics_rf = evaluate_model(y_test, y_pred_rf, y_proba_rf)

metrics_df = pd.DataFrame([metrics_lr, metrics_rf], index=['Logistic Regression', 'Random Forest'])
print("\n✅ Model Metrics Table")
print(metrics_df)

# -----------------------------
# Visualization: Confusion Matrices
# -----------------------------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d', cmap='Blues')
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.subplot(1,2,2)
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Greens')
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(processed_path, "task6_confusion_matrices.png"))
plt.show()
print("✅ Confusion matrices saved: task6_confusion_matrices.png")

# -----------------------------
# Visualization: ROC Curve
# -----------------------------
plt.figure(figsize=(8,6))
RocCurveDisplay.from_estimator(logreg, X_test, y_test, name='Logistic Regression')
RocCurveDisplay.from_estimator(best_rf, X_test, y_test, name='Random Forest')
plt.title("ROC Curves")
plt.savefig(os.path.join(processed_path, "task6_roc_curves.png"))
plt.show()
print("✅ ROC curves saved: task6_roc_curves.png")

# -----------------------------
# Save Best Model
# -----------------------------
model_path = os.path.join(processed_path, "best_model_rf.joblib")
joblib.dump(best_rf, model_path)
print(f"✅ Best model saved: {model_path}")

