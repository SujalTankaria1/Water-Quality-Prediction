# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Load model & scaler
model = joblib.load("models/water_quality_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Page config
st.set_page_config(page_title="💧 Water Quality Prediction", layout="wide")

# ================================
# 🎨 Custom CSS Styling
# ================================
page_bg = """
<style>
/* Main background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #e0f7fa, #80deea, #b39ddb, #7e57c2);
    color: #000000;
    background-attachment: fixed;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: rgba(30, 30, 30, 0.85);
    backdrop-filter: blur(10px);
    color: white;
    border-right: 2px solid #7e57c2;
}

/* Sidebar text */
[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* Title and headers */
h1, h2, h3 {
    color: #004d40 !important;
    text-align: center;
    font-weight: 700;
}

/* Buttons */
.stButton button {
    background: linear-gradient(90deg, #26c6da, #7e57c2);
    color: white;
    border-radius: 12px;
    padding: 0.6em 1.2em;
    font-weight: bold;
    border: none;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
    transition: all 0.3s ease;
}
.stButton button:hover {
    background: linear-gradient(90deg, #7e57c2, #26c6da);
    transform: scale(1.05);
    color: #fff;
}

/* Dataframe tables */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Title
st.title("💧 Water Quality Prediction App")
st.markdown("Choose between **Single Sample Prediction** or **CSV Upload** for water quality testing.")

# Sidebar for mode selection
mode = st.sidebar.radio("🔎 Select Mode", ["Single Sample Input", "Upload CSV File"])

# ========================================
# 1️⃣ Single Sample Input Mode
# ========================================
if mode == "Single Sample Input":
    st.subheader("🧪 Enter Water Sample Features")

    # Input fields (example features from dataset)
    ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, step=0.1)
    hardness = st.number_input("Hardness", min_value=0.0, step=0.1)
    solids = st.number_input("Solids (ppm)", min_value=0.0, step=1.0)
    chloramines = st.number_input("Chloramines", min_value=0.0, step=0.1)
    sulfate = st.number_input("Sulfate", min_value=0.0, step=0.1)
    conductivity = st.number_input("Conductivity", min_value=0.0, step=0.1)
    organic_carbon = st.number_input("Organic Carbon", min_value=0.0, step=0.1)
    trihalomethanes = st.number_input("Trihalomethanes", min_value=0.0, step=0.1)
    turbidity = st.number_input("Turbidity", min_value=0.0, step=0.1)

    if st.button("🔮 Predict"):
        # Prepare data
        sample = pd.DataFrame([[ph, hardness, solids, chloramines, sulfate,
                                conductivity, organic_carbon, trihalomethanes, turbidity]],
                              columns=["ph", "Hardness", "Solids", "Chloramines",
                                       "Sulfate", "Conductivity", "Organic_carbon",
                                       "Trihalomethanes", "Turbidity"])

        # Handle missing values
        imputer = SimpleImputer(strategy="median")
        sample_imputed = imputer.fit_transform(sample)

        # Scale
        sample_scaled = scaler.transform(sample_imputed)

        # Prediction
        pred = model.predict(sample_scaled)[0]
        prob = model.predict_proba(sample_scaled)[0][1]

        # Display result with styled card
        if pred == 1:
            st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #4CAF50, #81C784);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 20px;
        font-weight: bold;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.3);">
        ✅ Safe for Drinking <br>
        Confidence: {prob:.2f}
    </div>
    """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #FF5252, #FF8A80);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 20px;
        font-weight: bold;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.3);">
        ❌ Not Safe for Drinking <br>
        Confidence: {prob:.2f}
    </div>
    """, unsafe_allow_html=True)

        # Visualization (Pie Chart)
        labels = ["Not Safe", "Safe"]
        values = [1 - prob, prob]
        colors = ["#FF4B4B", "#4CAF50"]

        fig, ax = plt.subplots()
        ax.pie(values, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90)
        st.pyplot(fig)

# ========================================
# 2️⃣ CSV Upload Mode
# ========================================
elif mode == "Upload CSV File":
    st.subheader("📂 Upload CSV for Bulk Prediction")

    uploaded_file = st.file_uploader("Upload your water dataset (CSV)", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("📊 Uploaded Data Preview", data.head())

        if "Potability" in data.columns:
            X = data.drop("Potability", axis=1)
        else:
            X = data

        # Imputation
        imputer = SimpleImputer(strategy="median")
        X_imputed = imputer.fit_transform(X)

        # Scaling
        X_scaled = scaler.transform(X_imputed)

        # Prediction
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[:, 1]

        # Add results
        results = pd.DataFrame({
            "Prediction": ["Safe" if p == 1 else "Not Safe" for p in predictions],
            "Confidence": np.round(probabilities, 2)
        })

        output = pd.concat([data.reset_index(drop=True), results], axis=1)

        st.subheader("✅ Predictions")
        st.dataframe(output.head(20))

        # Visualization
        safe_count = (predictions == 1).sum()
        not_safe_count = (predictions == 0).sum()

        st.subheader("📊 Prediction Distribution")
        fig, ax = plt.subplots()
        sns.barplot(x=["Safe", "Not Safe"], y=[safe_count, not_safe_count],
                    palette=["#4CAF50", "#FF4B4B"], ax=ax)
        st.pyplot(fig)

        # Download
        csv = output.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Download Predictions as CSV",
            data=csv,
            file_name="water_quality_predictions.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.caption("🚀 Powered by Machine Learning | Interactive Mode with Streamlit")
