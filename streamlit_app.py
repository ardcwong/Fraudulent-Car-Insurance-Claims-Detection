# general libraries
import pickle
import pandas as pd

# model deployment
#from flask import Flask
import streamlit as st
import streamlit as st

from imblearn.over_sampling import BorderlineSMOTE, SVMSMOTE, RandomOverSampler
from imblearn.over_sampling import BorderlineSMOTE, SVMSMOTE, RandomOverSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier

from imblearn.under_sampling import (TomekLinks, NearMiss, RandomUnderSampler, ClusterCentroids,
                                     EditedNearestNeighbours, AllKNN, NeighbourhoodCleaningRule)

# modelling
from sklearn.preprocessing import MinMaxScaler,StandardScaler

from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     StratifiedKFold)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier,
                              AdaBoostClassifier,
                              ExtraTreesClassifier,
                              )
from catboost import CatBoostClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.model_selection import cross_validate
from sklearn.metrics import (ConfusionMatrixDisplay,  precision_score, recall_score, f1_score)

# read model and holdout data
model = pickle.load(open('lr.pkl', 'rb'))
X_holdout = pd.read_csv('holdout.csv', index_col=0)
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
