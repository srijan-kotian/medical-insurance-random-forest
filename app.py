import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load Model and Feature Schema
model = joblib.load('medical_model.pkl')
model_features = joblib.load('model_features.pkl')

st.set_page_config(page_title="FAU Research: Medical AI", layout="wide")
st.title("🩺 Medical Informatics: Cost Prediction & Uncertainty Analysis")

# Sidebar
st.sidebar.header("Patient Demographics")
age = st.sidebar.slider("Age", 18, 100, 25)
bmi = st.sidebar.slider("BMI", 15.0, 50.0, 24.0)
children = st.sidebar.selectbox("Children", [0, 1, 2, 3, 4, 5])
smoker = st.sidebar.radio("Smoker", ("Yes", "No"))
sex = st.sidebar.selectbox("Sex", ["male", "female"])
region = st.sidebar.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

if st.button("Generate Clinical Report"):
    # Create Input Dataframe (Matching One-Hot Schema)
    input_df = pd.DataFrame(0, index=[0], columns=model_features)
    input_df['age'] = age
    input_df['bmi'] = bmi
    input_df['children'] = children
    input_df['smoker'] = 1 if smoker == "Yes" else 0
    
    # Set One-Hot flags
    if f'sex_{sex}' in input_df.columns: input_df[f'sex_{sex}'] = 1
    if f'region_{region}' in input_df.columns: input_df[f'region_{region}'] = 1

    # Prediction + Uncertainty (Standard Deviation across Trees)
    all_tree_preds = np.array([tree.predict(input_df.values) for tree in model.estimators_])
    mean_pred = np.mean(all_tree_preds)
    std_pred = np.std(all_tree_preds)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Estimated Premium", f"${mean_pred:,.2f}", delta=f"±${std_pred:,.2f} (Uncertainty)")
        st.write("**Model Interpretability (Global Feature Importance)**")
        fig, ax = plt.subplots()
        ax.barh(model_features[:4], model.feature_importances_[:4], color='#3498db')
        st.pyplot(fig)

    with col2:
        st.info("### Statistical Interpretation")
        st.write(f"The model displays a confidence interval of **±${std_pred:,.2f}**. This variance represents the disagreement between individual decision trees, reflecting data density in this patient segment.")