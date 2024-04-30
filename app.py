import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from xgboost import XGBClassifier
import numpy as np

# Load pre-trained models and encoders
model = joblib.load('XGB_model.pkl')
gender_enc = joblib.load('gender_enc.pkl')
geo_enc = joblib.load('OHE_enc.pkl')
scaler = joblib.load('scaler.pkl')

def make_prediction(features):
    # Use the loaded model to make predictions
    # Replace this with the actual code for your model
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

# Main function
def main():
    st.title("Churn Prediction App")

    # Create input fields
    creditscore = st.number_input("Creditscore", min_value=0, max_value=1000, step=1)
    geography = st.radio("Geography", ["France", "Spain", "Germany"])
    gender = st.radio("Gender", ["Male", "Female"])
    age = st.slider("Age", min_value=0, max_value=100, step=1)
    tenure = st.slider("Tenure (in years)", min_value=0, max_value=10, step=1)
    balance = st.number_input("Balance", min_value=0, max_value=250000, step=1)
    numofproducts = st.slider("Number of Products", min_value=0, max_value=4, step=1)
    hascrcard = st.radio("Has CR card", ["Yes", "No"])
    isactivemember = st.radio("Active Member", ["Yes", "No"])
    estimatedsalary = st.number_input("Estimated Salary", min_value=0, max_value=200000, step=1)
    hascrcard = 1 if hascrcard == 'Yes' else 0
    isactivemember = 1 if isactivemember == 'Yes' else 0

    df = pd.DataFrame({
        'creditscore': [int(creditscore)],
        'geography': [geography],
        'gender': [gender],
        'age': [int(age)],
        'tenure': [int(tenure)],
        'balance': [int(balance)],
        'numofproducts': [int(numofproducts)],
        'hascrcard': [hascrcard],
        'isactivemember': [isactivemember],
        'estimatedsalary': [int(estimatedsalary)]
    })

    df = df.replace(gender_enc)
    geo_col = ['geography']
    geo_encode = pd.DataFrame(geo_enc.transform(df[['geography']]).toarray(), columns=geo_enc.get_feature_names_out())
    geo_encode = geo_encode.reset_index()
    df=pd.concat([df,geo_encode], axis=1)
    df=df.drop(['geography'],axis=1)
    scale_col = ['creditscore', 'balance', 'estimatedsalary']
    scaled_features = scaler.transform(df[scale_col])
    df[scale_col] = scaled_features

    if st.button('Make Prediction'):
        features=df      
        result = make_prediction(features)
        st.success(f'The prediction is: {result}')

if __name__ == "__main__":
  main()

