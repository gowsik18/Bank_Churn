import streamlit as st


# --------------------------------------------------
# Load trained pipeline
# --------------------------------------------------
model = joblib.load('churn_pipeline.pkl')

st.set_page_config(page_title="Bank Churn Prediction", layout="centered")
st.title("🏦 Bank Customer Churn Prediction System")

st.markdown("Predict customer churn risk using an industrial ML model.")

# --------------------------------------------------
# Sidebar Inputs (RAW FEATURES ONLY)
# --------------------------------------------------
st.sidebar.header("Customer Details")

CreditScore = st.sidebar.number_input("Credit Score", 300, 900, 650)
Geography = st.sidebar.selectbox("Geography", ["France", "Spain", "Germany"])
Gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
Age = st.sidebar.number_input("Age", 18, 80, 35)
Tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 3)
Balance = st.sidebar.number_input("Account Balance", 0.0, 300000.0, 50000.0)
NumOfProducts = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4])
HasCrCard = st.sidebar.selectbox("Has Credit Card", [0, 1])
IsActiveMember = st.sidebar.selectbox("Is Active Member", [0, 1])
EstimatedSalary = st.sidebar.number_input("Estimated Salary", 10000.0, 200000.0, 60000.0)

# --------------------------------------------------
# Create Input DataFrame
# --------------------------------------------------
input_df = pd.DataFrame([{
    "CreditScore": CreditScore,
    "Geography": Geography,
    "Gender": Gender,
    "Age": Age,
    "Tenure": Tenure,
    "Balance": Balance,
    "NumOfProducts": NumOfProducts,
    "HasCrCard": HasCrCard,
    "IsActiveMember": IsActiveMember,
    "EstimatedSalary": EstimatedSalary
}])

# --------------------------------------------------
# FEATURE ENGINEERING (MUST MATCH TRAINING)
# --------------------------------------------------

# Ratio features
input_df["Balance_to_Salary_ratio"] = (
    input_df["Balance"] / (input_df["EstimatedSalary"] + 1)
)

input_df["Is_Zero_Balance"] = (input_df["Balance"] == 0).astype(int)

input_df["Products_per_year"] = (
    input_df["NumOfProducts"] / (input_df["Tenure"] + 1)
)

# Business rules
HIGH_VALUE_BALANCE_THRESHOLD = 75000  # use same logic as training
input_df["High_value_customer"] = (
    input_df["Balance"] > HIGH_VALUE_BALANCE_THRESHOLD
).astype(int)

input_df["Inactive_High_Balance"] = (
    (input_df["IsActiveMember"] == 0) &
    (input_df["High_value_customer"] == 1)
).astype(int)

# Bucketing
input_df["Age_Bucket"] = pd.cut(
    input_df["Age"],
    bins=[0, 30, 45, 60, 100],
    labels=["Young", "Mid", "Senior", "Very_Senior"]
)

input_df["Tenure_Bucket"] = pd.cut(
    input_df["Tenure"],
    bins=[-1, 2, 5, 10],
    labels=["New", "Medium", "Loyal"]
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
st.markdown("---")

if st.button("🔮 Predict Churn"):
    churn_prob = model.predict_proba(input_df)[0][1]

    st.subheader(f"📊 Churn Probability: **{churn_prob:.2%}**")

    # Risk classification
    if churn_prob >= 0.60:
        st.error("🚨 HIGH RISK CUSTOMER")
        st.markdown("**Recommended Action:** 📞 Relationship Manager Call + Retention Offer")

    elif churn_prob >= 0.30:
        st.warning("⚠️ MEDIUM RISK CUSTOMER")
        st.markdown("**Recommended Action:** 🎁 Cashback / Product Bundling")

    else:
        st.success("✅ LOW RISK CUSTOMER")
        st.markdown("**Recommended Action:** 😊 No Immediate Action Needed")

    # Explainability summary
    st.markdown("### 🧠 Risk Drivers (Business Logic)")
    if IsActiveMember == 0:
        st.write("- Customer is inactive")
    if Balance == 0:
        st.write("- Zero balance account")
    if NumOfProducts == 1:
        st.write("- Only one product with the bank")
    if Tenure <= 2:
        st.write("- Low tenure (early-stage customer)")
