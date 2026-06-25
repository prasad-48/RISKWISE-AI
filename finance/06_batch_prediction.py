import pandas as pd
import joblib

print("=" * 60)
print("RISKWISE-AI : BATCH RISK SCORING")
print("=" * 60)

# --------------------------------------
# Load model
# --------------------------------------
model = joblib.load("models/xgb_model.pkl")

print("✓ Model Loaded")

# --------------------------------------
# Load customer dataset
# --------------------------------------
df = pd.read_csv("../UCI_Credit_Card.csv")

print(f"Dataset Shape: {df.shape}")

# Remove target if present
if "default.payment.next.month" in df.columns:
    df = df.drop(columns=["default.payment.next.month"])

# Remove ID
if "ID" in df.columns:
    df = df.drop(columns=["ID"])

# --------------------------------------
# Predict probabilities
# --------------------------------------
probabilities = model.predict_proba(df)[:, 1]

# --------------------------------------
# Risk classification
# --------------------------------------
def classify_risk(prob):

    if prob < 0.30:
        return "LOW"

    elif prob < 0.60:
        return "MEDIUM"

    else:
        return "HIGH"


results = pd.DataFrame()

results["Default_Probability"] = probabilities
results["Risk_Category"] = results["Default_Probability"].apply(classify_risk)

# --------------------------------------
# Save Results
# --------------------------------------
results.to_csv(
    "outputs/risk_predictions.csv",
    index=False
)

print("\nPrediction Summary")
print("-" * 30)

print(results["Risk_Category"].value_counts())

print("\n✓ File Saved:")
print("outputs/risk_predictions.csv")

print("\nDone!")