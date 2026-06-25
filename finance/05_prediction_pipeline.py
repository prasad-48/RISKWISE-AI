import pandas as pd
import joblib

print("=" * 60)
print("RISKWISE-AI : CUSTOMER RISK PREDICTION")
print("=" * 60)

# --------------------------------------------------
# Load Model
# --------------------------------------------------
print("\nLoading model...")

model = joblib.load("models/xgb_model.pkl")

print("✓ XGBoost model loaded")

# --------------------------------------------------
# Sample Customer Data
# --------------------------------------------------
# Replace these values later with user input
customer_data = {
    "LIMIT_BAL": 50000,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 35,
    "PAY_0": 2,
    "PAY_2": 2,
    "PAY_3": 1,
    "PAY_4": 0,
    "PAY_5": 0,
    "PAY_6": 0,
    "BILL_AMT1": 45000,
    "BILL_AMT2": 42000,
    "BILL_AMT3": 40000,
    "BILL_AMT4": 38000,
    "BILL_AMT5": 35000,
    "BILL_AMT6": 32000,
    "PAY_AMT1": 1000,
    "PAY_AMT2": 1000,
    "PAY_AMT3": 1500,
    "PAY_AMT4": 2000,
    "PAY_AMT5": 2000,
    "PAY_AMT6": 2500
}

# --------------------------------------------------
# Convert to DataFrame
# --------------------------------------------------
customer_df = pd.DataFrame([customer_data])

# --------------------------------------------------
# Predict
# --------------------------------------------------
probability = model.predict_proba(customer_df)[0][1]

# --------------------------------------------------
# Risk Classification
# --------------------------------------------------
if probability < 0.30:
    risk = "LOW"
    recommendation = "Approve Loan"

elif probability < 0.60:
    risk = "MEDIUM"
    recommendation = "Manual Review Required"

else:
    risk = "HIGH"
    recommendation = "Reject Loan Application"

# --------------------------------------------------
# Report
# --------------------------------------------------
print("\n" + "=" * 60)
print("CUSTOMER RISK REPORT")
print("=" * 60)

print(f"Default Probability : {probability*100:.2f}%")
print(f"Risk Category       : {risk}")
print(f"Recommendation      : {recommendation}")

print("=" * 60)