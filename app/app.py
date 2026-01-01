import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# resolve project root
BASE_DIR = Path(__file__).resolve().parent.parent

# load model and threshold
model = joblib.load(BASE_DIR / "src/models/churn_pipeline.pkl")
threshold = joblib.load(BASE_DIR / "src/models/threshold.pkl")

st.title("📊 Customer Churn Prediction")
st.write("Enter customer details to predict churn risk.")

ALL_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]

# user inputs
tenure = st.number_input("Tenure (months)", 0, 100, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

senior_citizen = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Has Partner", ["Yes", "No"])
dependents = st.selectbox("Has Dependents", ["Yes", "No"])

input_data = {
    "gender": "Male",
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": internet_service,
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": contract,
    "PaperlessBilling": "Yes",
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": tenure * monthly_charges if tenure > 0 else 0
}

input_df = pd.DataFrame([input_data], columns=ALL_COLUMNS)

if st.button("Predict Churn"):
    prob = model.predict_proba(input_df)[0][1]
    prediction = "🔴 Churn" if prob >= threshold else "🟢 No Churn"

    st.subheader("Prediction Result")
    st.write(f"**Churn Probability:** {prob:.2f}")
    st.write(f"**Prediction:** {prediction}")
