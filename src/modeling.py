import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)


# ------------------------------------
# LOAD DATA
# ------------------------------------
PROCESSED_PATH = r"C:\Users\assef\Desktop\Kifiya AI Mastery\week4\credit-risk-model\data\processed"

task3_path = os.path.join(PROCESSED_PATH, "task3_features.csv")
task4_path = os.path.join(PROCESSED_PATH, "task4_features.csv")

features_df = pd.read_csv(task3_path)
target_df = pd.read_csv(task4_path)


# ------------------------------------
# MERGE FEATURES + TARGET
# ------------------------------------
df = features_df.merge(
    target_df[['CustomerId', 'is_high_risk']],
    on='CustomerId',
    how='inner'
)

X = df.drop(columns=['CustomerId', 'is_high_risk'])
y = df['is_high_risk']


# ------------------------------------
# PREPROCESSING
# ------------------------------------
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)


# ------------------------------------
# TRAIN-TEST SPLIT
# ------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ------------------------------------
# MODELS
# ------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
}


# ------------------------------------
# TRAIN & EVALUATE
# ------------------------------------
for name, model in models.items():
    print(f"\n{'='*40}")
    print(f"Model: {name}")
    print(f"{'='*40}")

    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1-score :", f1_score(y_test, y_pred))
    print("ROC-AUC  :", roc_auc_score(y_test, y_proba))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
