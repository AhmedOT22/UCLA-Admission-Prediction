import streamlit as st
import pandas as pd

def get_user_input():
    with st.form("applicant_form"):
        col1, col2 = st.columns(2)

        with col1:
            gre = st.slider("GRE Score", min_value=260, max_value=340, value=300)
            toefl = st.slider("TOEFL Score", min_value=0, max_value=120, value=100)
            sop = st.slider("Statement of Purpose (SOP)", 1.0, 5.0, 3.5, step=0.5)
            cgpa = st.slider("CGPA (out of 10)", min_value=0.0, max_value=10.0, value=8.0)

        with col2:
            univ_rating = st.selectbox("University Rating", options=[1, 2, 3, 4, 5], index=2)
            lor = st.slider("Letter of Recommendation (LOR)", 1.0, 5.0, 3.0, step=0.5)
            research = st.radio("Research Experience", options=["No", "Yes"])

        submit = st.form_submit_button("Predict Admission Chance")

    if submit:
        user_input = pd.DataFrame([{
            "GRE_Score": gre,
            "TOEFL_Score": toefl,
            "University_Rating": univ_rating,
            "SOP": sop,
            "LOR": lor,
            "CGPA": cgpa,
            "Research": 1 if research == "Yes" else 0
        }])
        return user_input, True
    return None, False
