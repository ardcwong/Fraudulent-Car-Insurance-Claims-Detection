# general libraries
import pickle
import pandas as pd

# model deployment
#from flask import Flask
import streamlit as st


# read model and holdout data
model = pickle.load(open('lr.pkl', 'rb'))
#X_holdout = pd.read_csv('holdout.csv', index_col=0)
X_holdout = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    # Read CSV data into DataFrame
    X_holdout = pd.read_csv(uploaded_file, index_col=0)  # Assuming the index is in the first column

    # Display DataFrame
    st.write(df)
else:
    # Inform the user to upload a file
    st.write("Please upload a CSV file.")
holdout_transactions = X_holdout.index.to_list()

st.title("Car Insurance Fraud Detection")
html_temp = """
<div style="background:#025246 ;padding:10px">
<h2 style="color:white;text-align:center;"> Car Insurance Fraud Detection ML App </h2>
</div>
"""
st.markdown(html_temp, unsafe_allow_html = True)

#adding a selectbox
choice = st.selectbox(
    "Select Claim Reference Number:",
    options = holdout_transactions)


def predict_if_fraud(transaction_id):
    transaction = X_holdout.loc[transaction_id].values.reshape(1, -1)
    prediction_num = model.predict(transaction)[0]
    pred_map = {1: 'Fraud', 0: 'Not Fraud'}
    prediction = pred_map[prediction_num]
    return prediction

if st.button("Predict"):
    output = predict_if_fraud(choice)

    if output == 'Fraud':
        st.error('This transaction may be FRAUDULENT', icon="🚨")
    elif output == 'Not Fraud':
        st.success('This transaction is approved!', icon="✅")



