import streamlit as st
import pandas as pd
import joblib
import io
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Ecommerce Purchase Intent Classifier",
    layout="wide"
)

st.title("🛒 Ecommerce Purchase Intent Classifier")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_artifacts():
    with open("model/saved_model.pkl", "rb") as f:
        return pickle.load(f)

models, scaler, metrics_df = load_artifacts()
FEATURE_COLUMNS = list(scaler.feature_names_in_)

# ---------------- SIDEBAR ----------------
st.sidebar.header("Controls")

model_name = st.sidebar.selectbox(
    "Select ML Model",
    list(models.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV (features only or with Revenue)",
    type=["csv"]
)

# ---------------- METRICS TABLE ----------------
st.subheader("📊 Model Evaluation Metrics")
st.dataframe(metrics_df.style.format("{:.4f}"))

# ---------------- SAMPLE TEST CSV ----------------
st.subheader("📥 Download Sample Test CSV")

st.write(
    "This sample CSV contains **realistic customer sessions**. "
    "Row 1 = low purchase intent, Row 2 = high purchase intent."
)

sample_df = pd.DataFrame({
    "Administrative": [1, 4],
    "Administrative_Duration": [40, 300],
    "Informational": [0, 3],
    "Informational_Duration": [0, 180],
    "ProductRelated": [8, 55],
    "ProductRelated_Duration": [500, 2800],
    "BounceRates": [0.18, 0.01],
    "ExitRates": [0.20, 0.02],
    "PageValues": [0.0, 90.0],
    "SpecialDay": [0.0, 0.6],
    "Month": [5, 11],
    "OperatingSystems": [2, 3],
    "Browser": [1, 2],
    "Region": [1, 3],
    "TrafficType": [3, 2],
    "VisitorType": [0, 1],
    "Weekend": [0, 1],
    "Revenue": [0, 1]  # optional (used only for evaluation)
})

sample_df = sample_df[FEATURE_COLUMNS + ["Revenue"]]

csv_buffer = io.StringIO()
sample_df.to_csv(csv_buffer, index=False)

st.download_button(
    label="Download Sample Test CSV",
    data=csv_buffer.getvalue(),
    file_name="sample_test_data.csv",
    mime="text/csv"
)

# ---------------- PREDICTION ----------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    # Extract target if available
    y_true = None
    if "Revenue" in df.columns:
        y_true = df["Revenue"]
        df = df.drop(columns=["Revenue"])

    # Validate columns
    missing_cols = set(FEATURE_COLUMNS) - set(df.columns)
    if missing_cols:
        st.error(f"❌ Missing columns: {missing_cols}")
        st.stop()

    df = df[FEATURE_COLUMNS]

    # Scale & Predict
    X_scaled = scaler.transform(df)
    model = models[model_name]
    predictions = model.predict(X_scaled)

    # Prediction Results
    st.subheader("🔮 Prediction Results")

    result_df = pd.DataFrame({
        "Prediction": predictions
    })

    st.dataframe(result_df)

    st.subheader("📊 Prediction Counts")
    st.dataframe(
        result_df["Prediction"]
        .value_counts()
        .rename_axis("Prediction")
        .reset_index(name="Count")
    )

    # ---------------- EVALUATION ----------------
    if y_true is not None:
        st.subheader("📑 Classification Report")
        report = classification_report(y_true, predictions, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        st.subheader("🧩 Confusion Matrix")
        cm = confusion_matrix(y_true, predictions)

        fig, ax = plt.subplots()
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

else:
    st.info("⬅ Download the sample CSV or upload your own CSV to start predictions.")
