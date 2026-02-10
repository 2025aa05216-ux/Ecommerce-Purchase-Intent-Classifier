import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Ecommerce Purchase Intent", layout="wide")

# ---------------- LOAD MODEL ----------------
with open("model/saved_model.pkl", "rb") as f:
    models, scaler, metrics_df = pickle.load(f)

FEATURE_COLUMNS = list(scaler.feature_names_in_)

# ---------------- UI ----------------
st.title("🛒 Ecommerce Purchase Intent Classifier")

st.sidebar.header("Controls")

model_name = st.sidebar.selectbox(
    "Select ML Model",
    list(models.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV (features only)",
    type=["csv"]
)

# ---------------- METRICS ----------------
st.subheader("📊 Model Evaluation Metrics")
st.dataframe(metrics_df)

# ---------------- SAMPLE TEST CSV ----------------
st.subheader("📥 Download Sample Test CSV")

st.write(
    "Use this sample CSV to test the app. "
    "It contains valid feature columns with dummy values."
)

# Create dummy test data
sample_data = pd.DataFrame(
    [[0] * len(FEATURE_COLUMNS)],
    columns=FEATURE_COLUMNS
)

csv_bytes = sample_data.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Sample Test CSV",
    data=csv_bytes,
    file_name="sample_test_data.csv",
    mime="text/csv"
)

# ---------------- PREDICTION ----------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    # Ensure correct column order
    df = df[FEATURE_COLUMNS]

    X_scaled = scaler.transform(df)
    model = models[model_name]
    predictions = model.predict(X_scaled)

    st.subheader("🔮 Prediction Results")
    result_df = pd.DataFrame({
        "Prediction": predictions
    })
    st.dataframe(result_df)

    st.write("Prediction Counts")
    st.write(result_df["Prediction"].value_counts())

    # Optional evaluation if target exists
    if "Revenue" in df.columns:
        y_true = df["Revenue"]

        st.subheader("📑 Classification Report")
        st.text(classification_report(y_true, predictions))

        cm = confusion_matrix(y_true, predictions)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
else:
    st.info("⬅ Download the sample CSV or upload your own test CSV to get predictions.")
