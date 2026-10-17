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
tenure=st.slider("Tenure (in months)", min_value=0, max_value=80, value=12)
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
MonthlyCharges=st.number_input("Monthly Charges", min_value=0.0, max_value=500.0, value=88.12)
TotalCharges=st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=946.45)
if st.button("Predict Churn"):
    input_data = np.array([[gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines,
                            InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
                            StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
                            MonthlyCharges, TotalCharges]])
    churn_proba = churn_model.predict_proba(input_data)
    st.write("---")
    st.markdown("### Risk Assessment Results")
    if churn_proba[0][1]*100 < 30:
        status_color = "green"
        st.success(f"**Low Risk:** The customer is likely to stay. (Churn Probability: {churn_proba[0][1]*100:.2f}%)")
        st.markdown("Maintain good customer engagement and satisfaction to retain the customer.")
    elif 30 <= churn_proba[0][1]*100 < 60:
        status_color = "yellow"
        st.warning(f"**Medium Risk:** The customer may churn. (Churn Probability: {churn_proba[0][1]*100:.2f}%)")
        st.markdown("Consider targeted retention strategies to reduce churn risk.")
        st.markdown("**Suggested Actions:**")
        st.markdown("- Offer personalized promotions or discounts.")
        st.markdown("- Enhance customer support and engagement.")
        st.markdown("- Monitor customer feedback and address concerns promptly.")
        st.markdown("- Provide loyalty programs or incentives for long-term commitment.")
        st.markdown("- Analyze customer behavior and usage patterns to identify potential churn indicators.")
        st.markdown("- Implement proactive communication strategies to keep customers informed and engaged.")
    else:
        status_color = "red"
        st.error(f"**High Risk:** The customer is likely to churn. (Churn Probability: {churn_proba[0][1]*100:.2f}%)")
        st.markdown("Immediate action is required to retain the customer.")
        st.markdown("**Suggested Actions:**")
        st.markdown("- Conduct a personalized outreach to understand the customer's concerns.")
        st.markdown("- Offer tailored retention offers or incentives to encourage loyalty.")
        st.markdown("- Provide exceptional customer support and address any issues promptly.")
        st.markdown("- Implement a customer feedback loop to gather insights and improve services.")
        st.markdown("- Monitor customer behavior closely and identify early warning signs of churn.")
        st.markdown("- Consider implementing a customer loyalty program to reward long-term customers.")
        st.markdown("- Collaborate with the customer to find solutions that meet their needs and expectations.")
