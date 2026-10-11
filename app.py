import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔎",
    layout="centered"
)

model = joblib.load("fraud_detection_model.pkl")

st.title("Fraud Detection System")
st.write(
    "Analyze a transaction and estimate its fraud risk."
)

st.subheader("Transaction Details")

transaction_type = st.selectbox(
    "Transaction Type",
    ["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT"]
)

amount = st.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=100000.0,
    step=1000.0
)

oldbalance_org = st.number_input(
    "Origin Balance Before Transaction",
    min_value=0.0,
    value=150000.0,
    step=1000.0
)

oldbalance_dest = st.number_input(
    "Destination Balance Before Transaction",
    min_value=0.0,
    value=0.0,
    step=1000.0
)

step = st.number_input(
    "Transaction Time Step",
    min_value=1,
    max_value=743,
    value=300
)

if st.button("Analyze Transaction"):
    transaction = pd.DataFrame([{
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalance_org,
        "oldbalanceDest": oldbalance_dest,
        "step": step
    }])

    score = model.predict_proba(transaction)[0, 1]
    risk_score = score * 100

    st.metric("Risk Score", f"{risk_score:.2f}/100")

    if score >= 0.90:
        st.error("High Risk: Transaction flagged for review.")
    else:
        st.success("Below Alert Threshold: No alert from this model.")

    reasons = []

    if transaction_type in ["TRANSFER", "CASH_OUT"]:
        reasons.append(
            "This transaction type was associated with fraud in the training dataset."
        )

    if amount >= 200000:
        reasons.append("The transaction amount is relatively high.")

    if oldbalance_org < amount:
        reasons.append(
            "The origin balance is lower than the transaction amount."
        )

    st.subheader("Review Notes")

    if reasons:
        for reason in reasons:
            st.write("- " + reason)
    else:
        st.write(
            "No simple rule-based warning was triggered. "
            "The model score should still be reviewed."
        )

    st.caption(
        "Prototype using synthetic PaySim data. "
        "Risk Score is a model score, not a calibrated fraud probability. "
        "Do not use as the sole basis for real financial decisions."
    )
