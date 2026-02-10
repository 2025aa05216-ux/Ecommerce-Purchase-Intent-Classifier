import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Ecommerce Purchase Intent", layout="wide")

# Load trained objects
with open("model/saved_model.pkl", "rb") as f:
    models, scaler, metrics_df = pickle.load(f)

st.title("🛒 Ecommerce Purchase Intent Classifier")

# Sidebar controls
st.sidebar.header("Controls")
model_name = st.sidebar.selectbox(
    "Select ML Model",
    list(models.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV (features only)",
    type=["csv"]
)

# Metrics table
st.subheader("📊 Model Evaluation Metrics")
st.dataframe(metrics_df)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    X_scaled = scaler.transform(df)
    model = models[model_name]
    predictions = model.predict(X_scaled)

    st.subheader("Prediction Results")
    st.write(pd.Series(predictions).value_counts().rename("Count"))

    # confusion matrix
    if "Revenue" in df.columns:
        y_true = df["Revenue"]

        st.subheader("Classification Report")
        st.text(classification_report(y_true, predictions))

        cm = confusion_matrix(y_true, predictions)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
else:
    st.info("⬅ Upload a CSV file from the sidebar to get predictions.")
