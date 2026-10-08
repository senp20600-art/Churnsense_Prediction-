import streamlit as st # pyright: ignore[reportMissingImports]
import numpy as np # pyright: ignore[reportMissingImports]
import joblib # pyright: ignore[reportMissingImports]
churn_model = joblib.load("Churnsense_Prediction_Model.pkl")
st.title("Churn Prediction App")
st.write("Enter the customer details to predict churn:")
gender=st.selectbox("Sex (1=Male, 0=Female)", [0, 1])
SeniorCitizen=st.selectbox("Senior Citizen", [0, 1])
Partner=st.selectbox("Partner(1=Yes, 0=No)", [0, 1])
Dependents=st.selectbox("Dependents(1=Yes, 0=No)", [0, 1])
tenure=st.number_input("Tenure (in months)", min_value=0, max_value=100, value=0)
PhoneService=st.selectbox("Phone Service(1=Yes, 0=No)", [0, 1])
MultipleLines=st.selectbox("Multiple Lines(0=No,1=No phone service,2=Yes)", [0,1,2])
InternetService=st.selectbox("Internet Service(0=DSL,1=Fiber optic,2=No)", [0, 1, 2])
OnlineSecurity=st.selectbox("Online Security(0=No,1=No internet service,2=Yes)", [0, 1, 2])
OnlineBackup=st.selectbox("Online Backup(0=No,1=No internet service,2=Yes)", [0, 1, 2])
DeviceProtection=st.selectbox("Device Protection(0=No,1=No internet service,2=Yes)", [0, 1, 2])
TechSupport=st.selectbox("Tech Support(0=No,1=No internet service,2=Yes)", [0, 1, 2])
StreamingTV=st.selectbox("Streaming TV(0=No,1=No internet service,2=Yes)", [0, 1, 2])
StreamingMovies=st.selectbox("Streaming Movies(0=No,1=No internet service,2=Yes)", [0, 1, 2])
Contract=st.selectbox("Contract(0=Month-to-month,1=One year,2=Two year)", [0, 1, 2])
PaperlessBilling=st.selectbox("Paperless Billing(0=No,1=Yes)", [0, 1])   
PaymentMethod=st.selectbox("Payment Method(0=Bank transfer (automatic),1=Credit card (automatic),2=Electronic check,3=Mailed check)", [0, 1, 2, 3])
MonthlyCharges=st.number_input("Monthly Charges", min_value=0.0, max_value=500.0, value=0.0)
TotalCharges=st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=0.0)
if st.button("Predict Churn"):
    input_data = np.array([[gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines,
                            InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
                            StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
                            MonthlyCharges, TotalCharges]])
    prediction = churn_model.predict(input_data)
    if prediction[0] == 1:
        st.success("The customer is likely to churn.")
    else:
        st.success("The customer is not likely to churn.")
