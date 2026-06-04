import seaborn as sns
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Financial Fraud Detection Dashboard")

st.title("💳 Financial Fraud Detection System")

# Load dataset
df = pd.read_csv("credit_card_fraud_dataset.csv")

st.subheader("Dataset Preview")

col1, col2 = st.columns(2)

with col1:
    st.write("Total Records:", len(df))

with col2:
    st.metric("Accuracy", "98.93%")

st.write("Total Records:", len(df))

st.dataframe(df.head(10))

# Fraud Distribution
st.subheader("Fraud Distribution")

fraud_counts = df["IsFraud"].value_counts()

fig, ax = plt.subplots()

ax.bar(
    ["Not Fraud", "Fraud"],
    fraud_counts.values
)

ax.set_ylabel("Count")

st.pyplot(fig)

# Load model
model = joblib.load("models/fraud_model.pkl")

# Load confusion matrix
cm = joblib.load("models/confusion_matrix.pkl")

st.subheader("Confusion Matrix")

fig2, ax2 = plt.subplots()

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=ax2
)

ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")

st.pyplot(fig2)


st.subheader("Transaction Fraud Prediction")

transaction_id = st.number_input("Transaction ID", value=1)

amount = st.number_input("Amount", value=100.0)

merchant_id = st.number_input("Merchant ID", value=100)

transaction_type = st.selectbox(
    "Transaction Type",
    ["purchase", "refund"]
)

location = st.selectbox(
    "Location",
    [
        "New York",
        "Chicago",
        "Dallas",
        "Houston",
        "Phoenix",
        "Philadelphia",
        "San Antonio",
        "San Diego",
        "San Jose"
    ]
)

# Simple encoding
type_map = {
    "purchase": 0,
    "refund": 1
}

location_map = {
    "Chicago": 0,
    "Dallas": 1,
    "Houston": 2,
    "New York": 3,
    "Philadelphia": 4,
    "Phoenix": 5,
    "San Antonio": 6,
    "San Diego": 7,
    "San Jose": 8
}

if st.button("Predict Fraud"):

    data = [[
        transaction_id,
        amount,
        merchant_id,
        type_map[transaction_type],
        location_map[location]
    ]]

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("⚠ Fraudulent Transaction Detected")
    else:
        st.success("✅ Legitimate Transaction")