import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title = "Online Shoppers ML App", layout="wide")

st.title("Online Shoppers Purchase Prediction")
st.write("Compare multiple ML models on customer purchase  behaviour")

#load models
with open('model/saved_model.pkl', 'rb') as f:
    models, scaler, metrics = pickle.load(f)

#Sidebar
st.sidebar.header("Controls")
model_name =  st.sidebar.selectbox("Select ML Model", list(models.keys()))
file = st.sidebar.file_uploader("Upload Test CSV( features only)", type['csv'])

#Metrics Table
st.subheader("Model Evaluation Metrics")
st.dataframe(pd.DataFrame(metrics).T.syle.format('{:.3f}'))

if file is not None:
    df = pd.read_csv(file)
    X_scaled = scalar.transform(df)

    model=models[model_name]
    preds=model.predict(X_scaled)

    st.subheader("Prediction Output")
    st.write(preds)

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(preds, preds)
    fig,ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt ='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    st.subheader("Classification Report")
    st.text(classification_report(preds, preds))
