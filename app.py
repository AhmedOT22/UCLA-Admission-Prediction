import streamlit as st
import pandas as pd
from src.models.prediction import load_model_and_scaler, preprocess_user_input, align_features
from src.config import PROCESSED_DATA_DIR

# --- App Config ---
st.set_page_config(page_title="UCLA Admission Predictor", layout="centered")
st.title("UCLA Admission Chance Predictor")
st.markdown("Enter your academic profile below to check your predicted admission outcome.")

# --- Load Model & Scaler ---
model, scaler = load_model_and_scaler()
reference_columns = pd.read_csv(PROCESSED_DATA_DIR / "Admission_processed.csv").drop("Admit_Chance", axis=1).columns

# --- Manual Input Form ---
with st.form("applicant_form"):
    col1, col2 = st.columns(2)

    with col1:
        gre = st.number_input("GRE Score", min_value=260, max_value=340, value=300)
        toefl = st.number_input("TOEFL Score", min_value=0, max_value=120, value=100)
        sop = st.slider("Statement of Purpose (SOP)", 1.0, 5.0, 3.5, step=0.5)
        cgpa = st.number_input("CGPA (out of 10)", min_value=0.0, max_value=10.0, value=8.0)

    with col2:
        univ_rating = st.selectbox("University Rating", options=[1, 2, 3, 4, 5], index=2)
        lor = st.slider("Letter of Recommendation (LOR)", 1.0, 5.0, 3.0, step=0.5)
        research = st.radio("Research Experience", options=["No", "Yes"])

    submit = st.form_submit_button("Predict Admission Chance")

# --- Handle Submission ---
if submit:
    try:
        user_input = pd.DataFrame([{
            "GRE_Score": gre,
            "TOEFL_Score": toefl,
            "University_Rating": univ_rating,
            "SOP": sop,
            "LOR": lor,
            "CGPA": cgpa,
            "Research": 1 if research == "Yes" else 0
        }])

        processed = preprocess_user_input(user_input)
        aligned = align_features(processed, reference_columns)
        scaled = scaler.transform(aligned)
        prediction = model.predict(scaled)[0]

        st.subheader("Prediction Result")
        if prediction == 1:
            st.success("✅ You are likely to be **ADMITTED** to UCLA!")
        else:
            st.error("❌ You are likely to be **REJECTED**. Consider improving your profile.")

        with st.expander("View Your Input Summary"):
            st.dataframe(user_input.T.rename(columns={0: "Your Entry"}))

    except Exception as e:
        st.error(f"Something went wrong during prediction: {e}")
