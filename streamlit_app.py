import joblib
import streamlit as st
import numpy as np
import pandas as pd

df = pd.read_csv("Thyroid_Diff.csv")
df.to_pickle("Thyroid_Diff.pkl")

## Load trained model
model = joblib.load("Thyroid_Diff.pkl")

## Streamlit app
st.title("Thyroid Cancer Recurrence Prediction")

## Define the input options

genders = ["M", "F"]

smoking_options = ["Yes", "No"]

hx_smoking_options = ["Yes", "No"]

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
    "Bilateral",
    "Posterior"
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

stages = [
    "I", "II", "III", "IVA", "IVB"
]

responses = [
    "Excellent",
    "Indeterminate",
    "Structural Incomplete",
    "Biochemical Incomplete"
]


## User inputs
age = st.slider("Age", 15, 100, 40)

gender = st.selectbox("Gender", genders)

smoking = st.selectbox("Smoking", smoking_options)

hx_smoking = st.selectbox("History of Smoking", hx_smoking_options)

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

stage = st.selectbox("Cancer Stage", stages)

response = st.selectbox("Response", responses)

## Predict button
if st.button("Predict Thyroid Cancer Recurrence"):

    ## Create dict for input features
    input_data = {
        'age': age,
        'gender': gender,
        'smoking': smoking,
        'hx_smoking': hx_smoking,
        'hx_radiotherapy': hx_radiotherapy,
        'thyroid_function': thyroid_function,
        'physical_exam': physical_exam,
        'adenopathy': adenopathy,
        'pathology': pathology,
        'focality': focality,
        'risk': risk,
        't_stage': t_stage,
        'n_stage': n_stage,
        'm_stage': m_stage,
        'stage': stage,
        'response': response
    }

    ## Convert input data to a DataFrame
    df_input = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'smoking': [smoking],
        'hx_smoking': [hx_smoking],
        'hx_radiotherapy': [hx_radiotherapy],
        'thyroid_function': [thyroid_function],
        'physical_exam': [physical_exam],
        'adenopathy': [adenopathy],
        'pathology': [pathology],
        'focality': [focality],
        'risk': [risk],
        't_stage': [t_stage],
        'n_stage': [n_stage],
        'm_stage': [m_stage],
        'stage': [stage],
        'response': [response]
    })

    ## One-hot encoding
    df_input = pd.get_dummies(
        df_input, columns=[
            'Gender',
            'Smoking',
            'Hx Smoking',
            'Hx Radiothreapy',
            'Thyroid Function',
            'Physical Examination',
            'Adenopathy',
            'Pathology',
            'Focality',
            'Risk',
            'T',
            'N',
            'M',
            'Stage',
            'Response'
        ]
    )
    
    # df_input = df_input.to_numpy()

    df_input = df_input.reindex(columns = model.feature_names_in_,
                                fill_value=0)



    ## Predict
    y_unseen_pred = model.predict(df_input)[0]
    st.success(f"Predicted Thyroid Cancer Recurrence: {y_unseen_pred}")

## Page design
st.markdown(
    f"""
    <style>
    .stApp {{
        background: url("https://www.shutterstock.com/shutterstock/videos/1025418011/thumb/1.jpg");
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
)
