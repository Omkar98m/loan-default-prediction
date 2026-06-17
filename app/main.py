import streamlit as st
import numpy as np
import joblib
import os

# Load model
model = joblib.load(os.path.join("models", "D:\RAG2026\loan_project\models\loan_model(xgb).pkl"))

st.title("🏦 Loan Default Prediction App")
st.write("Enter borrower details below:")

# Input fields
Current_Loan_Amount = st.number_input("Current Loan Amount", value=0.0)
Credit_Score = st.number_input("Credit Score", value=0.0)
Annual_Income = st.number_input("Annual Income", value=0.0)
Monthly_Debt = st.number_input("Monthly Debt", value=0.0)
Years_of_Credit_History = st.number_input("Years of Credit History", value=0)
Number_of_Open_Accounts = st.number_input("Number of Open Accounts", value=0)
Number_of_Credit_Problems = st.number_input("Number of Credit Problems", value=0)
Current_Credit_Balance = st.number_input("Current Credit Balance", value=0.0)
Maximum_Open_Credit = st.number_input("Maximum Open Credit", value=0.0)
Bankruptcies = st.number_input("Bankruptcies", value=0)
Tax_Liens = st.number_input("Tax Liens", value=0)
Term_Short_Term = st.selectbox("Loan Term (Short Term = 1 / Long Term = 0)", [0, 1])

# Predict button
if st.button("Predict Loan Default"):

    input_data = np.array([[
        Current_Loan_Amount,
        Credit_Score,
        Annual_Income,
        Monthly_Debt,
        Years_of_Credit_History,
        Number_of_Open_Accounts,
        Number_of_Credit_Problems,
        Current_Credit_Balance,
        Maximum_Open_Credit,
        Bankruptcies,
        Tax_Liens,
        Term_Short_Term
    ]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("📊 Result")

    if probability < 0.30:
        st.error(f"⚠️ High Risk of Default")
    elif probability < 0.60:
        st.warning(f"✅ Medium Risk of Default")
    else :
        st.warning(f"✅ Medium Risk of Default")
    st.write(f"Probability of Default: **{round(probability, 3)}**")