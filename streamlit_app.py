import os
import joblib
import streamlit as st
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

## Page configuration

st.set_page_config(
    page_title="Thyroid Cancer Recurrence Prediction",
    page_icon="🩺",
    layout="wide"
)

## App design

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background-image:
        linear-gradient(rgba(15, 15, 25, 0.92), rgba(15, 15, 25, 0.92)),
        url("https://i.pinimg.com/1200x/4b/45/f0/4b45f0cf396f804a1c49a3a118457a9c.jpg");

        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }


    /* Main title */
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    /* Section headers */
    .section-title {
        font-size: 24px;
        font-weight: bold;
        margin-top: 15px;
        margin-bottom: 10px;
    }

    /* Result box */
    .result-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        background-color: white;
        margin-top: 20px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.15);
    }

    .result-title {
        font-size: 20px;
        font-weight: bold;
    }

    .result-value {
        font-size: 40px;
        font-weight: bold;
        margin-top: 10px;
    }

    /* Information box */
    .info-box {
        padding: 15px;
        border-radius: 10px;
        background-color: rgba(255,255,255,0.9);
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

## Title

st.markdown(
    '<div class="main-title">🩺 Thyroid Cancer Recurrence Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning Prediction Tool</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-box">
    This application uses a trained Random Forest machine learning model
    to predict whether thyroid cancer recurrence is likely based on the
    clinical information provided.
    </div>
    """,
    unsafe_allow_html=True
)

## Input options

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
    "T1a",
    "T1b",
    "T2",
    "T3a",
    "T3b",
    "T4a",
    "T4b"
]

n_stages = [
    "N0",
    "N1a",
    "N1b"
]

m_stages = [
    "M0",
    "M1"
]

responses = [
    "Excellent",
    "Indeterminate",
    "Structural Incomplete",
    "Biochemical Incomplete"
]

## User information

st.markdown(
    '<div class="section-title">Patient Information</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    age = st.slider(
        "Age",
        15,
        100,
        40
    )

    hx_radiotherapy = st.selectbox(
        "History of Radiotherapy",
        hx_radiotherapy_options
    )

    thyroid_function = st.selectbox(
        "Thyroid Function",
        thyroid_functions
    )

    physical_exam = st.selectbox(
        "Physical Examination",
        physical_exams
    )

    adenopathy = st.selectbox(
        "Adenopathy",
        adenopathy_options
    )

    pathology = st.selectbox(
        "Pathology",
        pathology_options
    )


with col2:

    focality = st.selectbox(
        "Focality",
        focality_options
    )

    risk = st.selectbox(
        "Risk",
        risk_options
    )

    t_stage = st.selectbox(
        "T Stage",
        t_stages
    )

    n_stage = st.selectbox(
        "N Stage",
        n_stages
    )

    m_stage = st.selectbox(
        "M Stage",
        m_stages
    )

    response = st.selectbox(
        "Response",
        responses
    )

## Prediction

st.markdown(
    '<div class="section-title">Prediction</div>',
    unsafe_allow_html=True
)

predict_button = st.button(
    "🔍 Predict Thyroid Cancer Recurrence",
    use_container_width=True
)


if predict_button:

    ## Create DataFrame containing the user's input
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

    ## One-hot encoding

    df_input = pd.get_dummies(df_input)

    # Make sure the input columns are identical to
    # the columns used when training the model
    df_input = df_input.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    ## Make prediction

    prediction = model.predict(df_input)[0]

    ## Convert prediction to Yes or No
    if str(prediction).lower() in ["yes", "y", "1", "true"]:
        result = "Yes"
    else:
        result = "No"

    ## Display result

    st.markdown(
        '<div class="result-box">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="result-title">Predicted Thyroid Cancer Recurrence</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="result-value">{result}</div>',
        unsafe_allow_html=True
    )

    if result == "Yes":

        st.warning(
            "The model predicts that thyroid cancer recurrence is likely."
        )

    else:

        st.success(
            "The model predicts that thyroid cancer recurrence is unlikely."
        )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )
    
