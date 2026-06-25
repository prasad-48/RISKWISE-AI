import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

print("=" * 60)
print("RISKWISE-AI : MODEL EXPLAINABILITY")
print("=" * 60)

# Load model
print("\nLoading XGBoost model...")
model = joblib.load("models/xgb_model.pkl")

# Load features
X = pd.read_csv("X_features.csv")

print(f"Dataset Shape: {X.shape}")

# Create SHAP explainer
print("\nCreating SHAP explainer...")
explainer = shap.TreeExplainer(model)

# Sample for faster computation
X_sample = X.sample(1000, random_state=42)

print("Calculating SHAP values...")
shap_values = explainer.shap_values(X_sample)

# --------------------------------------------------
# Feature Importance Plot
# --------------------------------------------------
print("\nCreating Feature Importance Plot...")

plt.figure(figsize=(10, 6))

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

top10 = feature_importance.head(10)

plt.barh(top10["Feature"], top10["Importance"])
plt.title("Top 10 Most Important Features")
plt.tight_layout()

plt.savefig(
    "outputs/feature_importance.png",
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# SHAP Summary Plot
# --------------------------------------------------
print("Creating SHAP Summary Plot...")

shap.summary_plot(
    shap_values,
    X_sample,
    show=False
)

plt.savefig(
    "outputs/shap_summary.png",
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# Print Top Features
# --------------------------------------------------
print("\nTop Risk Factors:")

for i, row in top10.iterrows():
    print(f"{row['Feature']} --> {row['Importance']:.4f}")

print("\n" + "=" * 60)
print("EXPLAINABILITY COMPLETE")
print("=" * 60)

print("\nFiles Generated:")
print("✓ outputs/feature_importance.png")
print("✓ outputs/shap_summary.png")