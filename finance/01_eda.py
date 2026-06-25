import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("=" * 50)
print("LOAN RISK ANALYSIS - DAY 1: EDA")
print("=" * 50)

# Load dataset
print("\nLoading dataset...")
df = pd.read_csv('../UCI_Credit_Card.csv')

# Check shape
print(f"\n✓ Dataset loaded!")
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")

# Check columns
print(f"\nColumns: {list(df.columns[:5])}... and {len(df.columns)-5} more")

# Check target variable
print("\n" + "="*50)
print("TARGET VARIABLE ANALYSIS")
print("="*50)
target_counts = df['default.payment.next.month'].value_counts()
print(f"\nNo Default (0): {target_counts[0]} ({target_counts[0]/len(df)*100:.1f}%)")
print(f"Default (1): {target_counts[1]} ({target_counts[1]/len(df)*100:.1f}%)")

# Basic stats
print("\n" + "="*50)
print("BASIC STATISTICS")
print("="*50)
print(df.describe())

# Check for missing values
print("\n" + "="*50)
print("MISSING VALUES")
print("="*50)
missing = df.isnull().sum()
if missing.sum() == 0:
    print("✓ No missing values! Dataset is clean.")
else:
    print(f"Missing values found:\n{missing[missing > 0]}")

# First few rows
print("\n" + "="*50)
print("FIRST 5 ROWS")
print("="*50)
print(df.head())

print("\n" + "="*50)
print("✓ EDA COMPLETE!")
print("="*50)

# Create a simple plot
print("\nCreating visualization...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot 1: Target distribution
df['default.payment.next.month'].value_counts().plot(kind='bar', ax=axes[0], color=['green', 'red'])
axes[0].set_title('Default Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Default (0=No, 1=Yes)')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels(['No Default', 'Default'], rotation=0)

# Plot 2: Age distribution
df['AGE'].hist(bins=30, ax=axes[1], color='blue', edgecolor='black')
axes[1].set_title('Age Distribution', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Age')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
print("Saving plot as 'eda_plot.png'...")
plt.savefig('outputs/eda_plot.png', dpi=100, bbox_inches='tight')
print("✓ Plot saved!")

plt.show()