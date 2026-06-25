# ==========================================
# RiskWise-AI
# 02_feature_engineering.py
# ==========================================

import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from pathlib import Path

print("=" * 50)
print("RISKWISE-AI : FEATURE ENGINEERING")
print("=" * 50)

# ------------------------------------------
# Create required folders
# ------------------------------------------
Path("models").mkdir(exist_ok=True)

# ------------------------------------------
# Load Dataset
# ------------------------------------------
print("\nLoading dataset...")

df = pd.read_csv("../UCI_Credit_Card.csv")

print(f"Dataset Shape: {df.shape}")

# ------------------------------------------
# Rename Target Column
# ------------------------------------------
df.rename(
    columns={
        "default.payment.next.month": "default"
    },
    inplace=True
)

# ------------------------------------------
# Drop ID Column
# ------------------------------------------
if "ID" in df.columns:
    df.drop("ID", axis=1, inplace=True)
    print("ID column removed")

# ------------------------------------------
# Check Missing Values
# ------------------------------------------
missing_values = df.isnull().sum().sum()

print(f"Missing Values: {missing_values}")

if missing_values > 0:
    df.fillna(df.median(numeric_only=True), inplace=True)
    print("Missing values handled")

# ------------------------------------------
# Separate Features & Target
# ------------------------------------------
X = df.drop("default", axis=1)
y = df["default"]

print(f"\nFeatures Shape : {X.shape}")
print(f"Target Shape   : {y.shape}")

# ------------------------------------------
# Feature Scaling
# ------------------------------------------
print("\nApplying StandardScaler...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(
    X_scaled,
    columns=X.columns
)

# ------------------------------------------
# Save Processed Files
# ------------------------------------------
print("\nSaving files...")

X_scaled.to_csv("X_features.csv", index=False)
y.to_csv("y_target.csv", index=False)

with open("models/feature_scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

# ------------------------------------------
# Summary
# ------------------------------------------
print("\n" + "=" * 50)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 50)

print(f"X_features.csv Shape : {X_scaled.shape}")
print(f"y_target.csv Shape   : {y.shape}")

print("\nFiles Created:")
print("✓ X_features.csv")
print("✓ y_target.csv")
print("✓ models/feature_scaler.pkl")

print("\nDone!")