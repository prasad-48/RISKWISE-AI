# 🏦 RiskWise-AI

An end-to-end Machine Learning application that predicts credit card default risk using customer financial and repayment history data.

The project leverages Exploratory Data Analysis (EDA), Feature Engineering, Machine Learning, Explainable AI (SHAP), Batch Risk Scoring, and a Streamlit Dashboard to provide interpretable credit risk predictions.

---

## 📌 Project Overview

Financial institutions face significant losses due to customer defaults.

RiskWise-AI helps estimate the probability that a customer will default on their next payment by analyzing:

- Credit limit
- Demographics
- Repayment history
- Bill statements
- Previous payment amounts

The system provides:

- Default Probability Score
- Risk Classification
- Business Recommendation
- Explainable AI Insights

---

## 🎯 Business Objective

Build a machine learning-powered credit risk assessment system that can:

- Predict customer default risk
- Support loan approval decisions
- Reduce financial losses
- Improve risk management processes
- Provide interpretable model explanations

---

## 📊 Dataset

Dataset: UCI Credit Card Default Dataset

- Records: 30,000 customers
- Features: 23 predictor variables
- Target Variable:

```text
default.payment.next.month
```

Target Classes:

| Value | Meaning |
|---------|---------|
| 0 | No Default |
| 1 | Default |

Dataset Source:

https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients

---

## 🏗 Project Architecture

```text
Raw Dataset
     │
     ▼
Exploratory Data Analysis
     │
     ▼
Feature Engineering
     │
     ▼
Model Training
     │
     ▼
Model Evaluation
     │
     ▼
Explainable AI (SHAP)
     │
     ▼
Prediction Pipeline
     │
     ▼
Batch Risk Scoring
     │
     ▼
Streamlit Dashboard
```

---

## 📁 Project Structure

```text
RiskWise-AI/
│
├── app.py
├── README.md
├── requirements.txt
├── UCI_Credit_Card.csv
│
├── finance/
│   ├── 01_eda.py
│   ├── 02_feature_engineering.py
│   ├── 03_model_training.py
│   ├── 04_explainability.py
│   ├── 05_prediction_pipeline.py
│   ├── 06_batch_prediction.py
│   │
│   ├── models/
│   │   ├── xgb_model.pkl
│   │   ├── rf_model.pkl
│   │   ├── lr_model.pkl
│   │   └── feature_scaler.pkl
│   │
│   └── outputs/
│       ├── eda_plot.png
│       ├── feature_importance.png
│       ├── shap_summary.png
│       └── risk_predictions.csv
```

---

## 🔍 Exploratory Data Analysis

Key Findings:

- Dataset contains 30,000 customer records
- No missing values detected
- Default Rate ≈ 22%
- Dataset is moderately imbalanced
- Repayment history is highly predictive of default behavior

Generated Visualization:

- Target Distribution
- Age Distribution

---

## ⚙️ Feature Engineering

Performed:

- Removal of ID column
- Target separation
- Feature scaling using StandardScaler
- Dataset preparation for machine learning

Output Files:

```text
X_features.csv
y_target.csv
feature_scaler.pkl
```

---

## 🤖 Machine Learning Models

Three models were trained and evaluated:

### Logistic Regression

| Metric | Score |
|---------|--------|
| Accuracy | 0.6797 |
| Precision | 0.3672 |
| Recall | 0.6202 |
| F1 Score | 0.4613 |
| AUC | 0.7081 |

---

### Random Forest

| Metric | Score |
|---------|--------|
| Accuracy | 0.8172 |
| Precision | 0.6652 |
| Recall | 0.3489 |
| F1 Score | 0.4577 |
| AUC | 0.7733 |

---

### XGBoost (Best Model)

| Metric | Score |
|---------|--------|
| Accuracy | 0.8188 |
| Precision | 0.6657 |
| Recall | 0.3632 |
| F1 Score | 0.4700 |
| AUC | 0.7781 |

### 🏆 Best Model

**XGBoost**

AUC Score:

```text
0.7781
```

---

## 🧠 Explainable AI

SHAP was used to understand model decisions.

Top Risk Factors:

| Rank | Feature |
|---------|---------|
| 1 | PAY_0 |
| 2 | PAY_2 |
| 3 | PAY_3 |
| 4 | PAY_4 |
| 5 | PAY_5 |

### Business Insight

Past repayment behavior is the strongest predictor of future default risk.

Generated Outputs:

- Feature Importance Plot
- SHAP Summary Plot

---

## 📈 Batch Risk Scoring

30,000 customers were scored using the trained XGBoost model.

Results:

| Risk Category | Customers |
|--------------|----------:|
| LOW | 26,817 |
| MEDIUM | 2,791 |
| HIGH | 392 |

Output:

```text
outputs/risk_predictions.csv
```

---

## 🖥 Streamlit Dashboard

Features:

- Customer Risk Prediction
- Risk Classification
- Default Probability Score
- Loan Recommendation Engine

Run locally:

```bash
streamlit run app.py
```

---

## 🛠 Technology Stack

### Programming

- Python

### Data Processing

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-Learn
- XGBoost

### Explainable AI

- SHAP

### Deployment

- Streamlit

### Version Control

- Git
- GitHub

---

## 🚀 Installation

Clone Repository

```bash
git clone https://github.com/prasad-48/RISKWISE-AI.git
```

Move into Project

```bash
cd RISKWISE-AI
```

Create Virtual Environment

```bash
python -m venv venv
```

Activate

Windows:

```bash
venv\Scripts\activate
```

Install Dependencies

```bash
pip install -r requirements.txt
```

Run Dashboard

```bash
streamlit run app.py
```

---

## 🔮 Future Enhancements

- Real-time API integration
- Advanced Hyperparameter Tuning
- SHAP Visualizations inside Dashboard
- Batch CSV Upload Interface
- PDF Risk Report Generation
- Cloud Deployment

---

## 👨‍💻 Author

Prasad Kale

GitHub:
https://github.com/prasad-48

---

## ⭐ Project Highlights

✅ End-to-End Machine Learning Pipeline

✅ Explainable AI (SHAP)

✅ XGBoost Risk Prediction

✅ Batch Customer Scoring

✅ Interactive Streamlit Dashboard

✅ Production-Oriented Project Structure

---

If you found this project useful, consider giving it a ⭐ on GitHub.