import streamlit as st
import pandas as pd
from src.models.prediction import (
    load_model_and_scaler,
    preprocess_user_input,
    align_features,
    get_admission_probability
)
from src.utils.form import get_user_input
from src.utils.gauge import generate_gauge_chart
from src.config import PROCESSED_DATA_DIR

# --- App Config ---
st.set_page_config(page_title="UCLA Admission Predictor", layout="centered")
st.title("🎓 UCLA Admission Chance Predictor")
st.markdown("Enter your academic profile below to check your predicted admission outcome.")

# --- Load Model & Scaler ---
model, scaler = load_model_and_scaler()
reference_columns = pd.read_csv(PROCESSED_DATA_DIR / "Admission_processed.csv").drop("Admit_Chance", axis=1).columns

# --- Get User Input ---
user_input, submit = get_user_input()

# --- Handle Submission ---
if submit:
    try:
        # Process input
        processed = preprocess_user_input(user_input)
        aligned = align_features(processed, reference_columns)
        scaled = scaler.transform(aligned)

        # Predict probability
        admit_prob = round(get_admission_probability(model, scaled) * 100, 2)
        interpretation = "✅ Likely Admitted" if admit_prob >= 50 else "❌ Likely Rejected"

        # Gauge chart
        st.subheader("🎯 Admission Probability")
        fig = generate_gauge_chart(admit_prob, interpretation, color=None)
        st.plotly_chart(fig, use_container_width=True)

        # Confidence display
        st.markdown(f"**Model Confidence:** `{admit_prob:.2f}%` chance of being admitted.")

        # Feedback message
        if admit_prob >= 80:
            feedback = "🌟 Strong candidate – Excellent chance of admission!"
        elif admit_prob >= 60:
            feedback = "👍 Good profile – Competitive but not guaranteed."
        elif admit_prob >= 40:
            feedback = "⚠️ Moderate chance – Consider improving your profile."
        else:
            feedback = "❌ Low chance – Admission unlikely with current profile."

        st.markdown(f"**Feedback:** {feedback}")
        st.markdown("**Note:** This is a statistical model and should not be the sole basis for your application decisions.")

        # Input summary
        with st.expander("📋 View Your Input Summary"):
            st.dataframe(user_input.T.rename(columns={0: "Your Entry"}))

    except Exception as e:
        st.error(f"Something went wrong during prediction: {e}")
