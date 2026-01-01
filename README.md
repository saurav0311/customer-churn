# Customer Churn Prediction (End-to-End ML)

## Problem Statement
Predict whether a telecom customer will churn, enabling proactive retention.

## Dataset
IBM Telco Customer Churn dataset (7,043 rows, 21 features).

## Approach
- EDA and class imbalance analysis
- Feature engineering with pipelines
- Model comparison (Logistic, Decision Tree, Random Forest)
- Threshold tuning for recall–precision trade-off

## Model Choice
Logistic Regression selected for:
- Best recall–precision balance
- Interpretability
- Stable generalization

## Threshold Tuning
Default threshold (0.5) vs tuned threshold (0.35):
- Recall improved from 56% → 71%
- Accuracy decreased slightly (business trade-off)

## Deployment
- Saved preprocessing + model as pipeline
- Streamlit app for real-time predictions

## How to Run
```bash
pip install -r requirements.txt
streamlit run app/app.py
