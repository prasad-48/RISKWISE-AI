import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from xgboost import XGBClassifier

print("=" * 60)
print("RISKWISE-AI : MODEL TRAINING")
print("=" * 60)

# --------------------------------------------------
# Load Data
# --------------------------------------------------
print("\nLoading processed data...")

X = pd.read_csv("X_features.csv")
y = pd.read_csv("y_target.csv").squeeze()

print(f"Features Shape: {X.shape}")
print(f"Target Shape: {y.shape}")

# --------------------------------------------------
# Train Test Split
# --------------------------------------------------
print("\nSplitting train/test data...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")

# --------------------------------------------------
# Models
# --------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
}

results = {}

# --------------------------------------------------
# Training Loop
# --------------------------------------------------
for name, model in models.items():

    print("\n" + "-" * 60)
    print(f"Training {name}")
    print("-" * 60)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUC": auc
    }

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")

# --------------------------------------------------
# Save Models
# --------------------------------------------------
print("\nSaving models...")

joblib.dump(models["Logistic Regression"], "models/lr_model.pkl")
joblib.dump(models["Random Forest"], "models/rf_model.pkl")
joblib.dump(models["XGBoost"], "models/xgb_model.pkl")

print("✓ lr_model.pkl saved")
print("✓ rf_model.pkl saved")
print("✓ xgb_model.pkl saved")

# --------------------------------------------------
# Best Model
# --------------------------------------------------
best_model = max(results, key=lambda x: results[x]["AUC"])

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

for model_name, metrics in results.items():
    print(f"\n{model_name}")
    for metric_name, value in metrics.items():
        print(f"{metric_name:<10}: {value:.4f}")

print("\n" + "=" * 60)
print(f"BEST MODEL: {best_model}")
print(f"AUC SCORE : {results[best_model]['AUC']:.4f}")
print("=" * 60)

print("\nTraining Complete!")
