import pandas as pd
import joblib
from src.config import MODEL_DIR
from sklearn.preprocessing import MinMaxScaler

def load_model_and_scaler():
    """load_model_and_scaler - loads the pre-trained model and scaler from the specified directory.
        Returns:
            model: The pre-trained model.
            scaler: The scaler used for preprocessing the data.
    """
    model = joblib.load(MODEL_DIR / "mlp_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    return model, scaler

def preprocess_user_input(df):
    df = df.copy()
    df = df.drop(["Serial_No"], axis=1, errors="ignore")
    df["University_Rating"] = df["University_Rating"].astype("object")
    df["Research"] = df["Research"].astype("object")
    df_encoded = pd.get_dummies(df, columns=["University_Rating", "Research"], dtype="int")
    return df_encoded

def align_features(df_encoded, reference_columns):
    # Add missing columns
    for col in reference_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    return df_encoded[reference_columns]

def get_admission_probability(model, scaled_input):
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(scaled_input)[0][1]  # Class 1 (Admitted)
        return prob
    else:
        raise ValueError("Model does not support probability prediction.")
