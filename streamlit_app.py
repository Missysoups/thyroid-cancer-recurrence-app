import os
import joblib
import streamlit as st
import numpy as np
import pandas as pd

## Load the trained model
model_path = os.path.join(
    os.path.dirname(__file__),
    "Thyroid_Diff_Model.pkl"
)

model = joblib.load(model_path)

## Check that the correct model was loaded
if not hasattr(model, "predict"):
    st.error("The uploaded PKL file is not a trained scikit-learn model.")
    st.stop()

## Streamlit app
st.title("Thyroid Cancer Recurrence Prediction")

## Define the input options

hx_radiotherapy_options = ["Yes", "No"]

thyroid_functions = [
    "Euthyroid",
    "Clinical Hyperthyroidism",
    "Clinical Hypothyroidism",
    "Subclinical Hyperthyroidism",
    "Subclinical Hypothyroidism"
]

physical_exams = [
    "Normal",
    "Single nodular goiter-left",
    "Single nodular goiter-right",
    "Multinodular goiter",
    "Diffuse goiter"
]

adenopathy_options = [
    "No",
    "Right",
    "Left",
    "Posterior",
    "Extensive"
]

pathology_options = [
    "Micropapillary",
    "Papillary",
    "Follicular",
    "Hurthel cell"
]

focality_options = [
    "Uni-Focal",
    "Multi-Focal"
]

risk_options = [
    "Low",
    "Intermediate",
    "High"
]

t_stages = [
    "T1a", "T1b", "T2", "T3a", "T3b", "T4a", "T4b"
]

n_stages = [
    "N0", "N1a", "N1b"
]

m_stages = [
    "M0", "M1"
]

responses = [
    "Excellent",
    "Indeterminate",
    "Structural Incomplete",
    "Biochemical Incomplete"
]


## User inputs
age = st.slider("Age", 15, 100, 40)

hx_radiotherapy = st.selectbox("History of Radiotherapy", hx_radiotherapy_options)

thyroid_function = st.selectbox("Thyroid Function", thyroid_functions)

physical_exam = st.selectbox("Physical Examination", physical_exams)

adenopathy = st.selectbox("Adenopathy", adenopathy_options)

pathology = st.selectbox("Pathology", pathology_options)

focality = st.selectbox("Focality", focality_options)

risk = st.selectbox("Risk", risk_options)

t_stage = st.selectbox("T Stage", t_stages)

n_stage = st.selectbox("N Stage", n_stages)

m_stage = st.selectbox("M Stage", m_stages)

response = st.selectbox("Response", responses)

## Predict button
if st.button("Predict Thyroid Cancer Recurrence"):

    df_input = pd.DataFrame({
        'Age': [age],
        'Hx Radiothreapy': [hx_radiotherapy],
        'Thyroid Function': [thyroid_function],
        'Physical Examination': [physical_exam],
        'Adenopathy': [adenopathy],
        'Pathology': [pathology],
        'Focality': [focality],
        'Risk': [risk],
        'T': [t_stage],
        'N': [n_stage],
        'M': [m_stage],
        'Response': [response]
    })

    # One-hot encoding
    df_input = pd.get_dummies(df_input)

    # Make the input columns exactly match the trained model
    df_input = df_input.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    ## Make prediction
    prediction = model.predict(df_input)[0]

    ## Convert prediction to Yes or No
    if prediction == "Yes" or prediction == "Y" or prediction == 1:
        result = "Yes"
    else:
        result = "No"

    st.success(
        f"Predicted Thyroid Cancer Recurrence: {result}"
    )

## Page design
st.markdown(
    f"""
    <style>
    .stApp {{
        background: url("https://i.pinimg.com/736x/36/fc/c8/36fcc8fa8b772f3f71e54c2b90888a1a.jpg");
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
)
