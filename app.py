import streamlit as st
#from pdf2image import convert_from_path
from PIL import Image
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(layout="wide")
st.title("📈 Customer Churn Predictor Dashboard")

# Sidebar
st.sidebar.header("Upload Data & Select Model")

# Load models
lr_model = joblib.load("models/logistic_model.pkl")
xgb_model = joblib.load("models/xgb_model.pkl")
rf_model = joblib.load("models/rf_model.pkl")
columns = joblib.load("models/train_columns.pkl")

# Model selection dropdown
model_option = st.sidebar.selectbox("Select Model", ("Logistic Regression", "XGBoost", "Random Forest"))

# File uploader
uploaded = st.sidebar.file_uploader("Upload customer CSV file", type="csv")

# Power BI dashboard
st.markdown("### 📄 Customer Retention Dashboard image")
# Load and display the image
image_path = "powerbi/customer_retention.png"
image = Image.open(image_path)

st.image(image, caption="Customer Churn Dashboard", use_column_width=True)

import streamlit as st

st.markdown("### 🧩 Download or Open Power BI Dashboard (.pbix)")

pbix_path = "powerbi/customer_retention_dashboard.pbix"

with open(pbix_path, "rb") as f:
    st.download_button(
        label="📥 Download .pbix File",
        data=f,
        file_name="Customer_retention_dashboard.pbix",
        mime="application/octet-stream"
    )



# Prediction logic
if uploaded:
    df = pd.read_csv(uploaded)
    customer_ids = df['customerID']
    df.drop('customerID', axis=1, inplace=True)

    # Preprocessing
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    bin_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in bin_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    df.replace(['No internet service', 'No phone service'], 'No', inplace=True)
    df = pd.get_dummies(df, drop_first=True)

    # Ensure all columns match training set
    for col in columns:
        if col not in df.columns:
            df[col] = 0
    df = df[columns]

    # Feature scaling
    scaler = StandardScaler()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Model prediction
    if model_option == "Logistic Regression":
        preds = lr_model.predict(df)
    elif model_option == "XGBoost":
        preds = xgb_model.predict(df)
    else:
        preds = rf_model.predict(df)

    # Results
    result_df = pd.DataFrame({
        "CustomerID": customer_ids,
        "Churn Prediction": ['Yes' if p == 1 else 'No' for p in preds]
    })

    st.subheader("📋 Prediction Results")
    st.dataframe(result_df)
    import json
    # Load metrics
    with open("models/model_metrics.json", "r") as f:
        model_metrics = json.load(f)

    # Show metrics
    st.sidebar.markdown("### 📊 Model Performance Metrics")
    selected_metrics = model_metrics.get(model_option)
    if selected_metrics:
        for k, v in selected_metrics.items():
            st.sidebar.metric(k, f"{v:.4f}")

    st.download_button("📥 Download Results", result_df.to_csv(index=False), "churn_predictions.csv")
