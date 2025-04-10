import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from src.config import PROCESSED_DATA_DIR, TEST_SIZE, RANDOM_STATE


def preprocess_data(df):
    """preprocess_data - preprocesses the input DataFrame by binarizing the target variable,
        dropping unnecessary columns, converting categorical columns to object type,
        one-hot encoding categorical columns, and scaling the features.
    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Step 1: Binarize target
    df["Admit_Chance"] = (df["Admit_Chance"] >= 0.8).astype(int)

    # Step 2: Drop unnecessary columns
    df = df.drop(["Serial_No"], axis=1)

    # Step 3: Convert categorical columns
    df["University_Rating"] = df["University_Rating"].astype("object")
    df["Research"] = df["Research"].astype("object")

    # Step 4: One-hot encode categorical columns
    df_encoded = pd.get_dummies(df, columns=["University_Rating", "Research"], dtype="int")

    # === Save the processed dataset ===
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    processed_path = PROCESSED_DATA_DIR / "Admission_processed.csv"
    df_encoded.to_csv(processed_path, index=False)
    print(f"[INFO] Processed data saved to: {processed_path}")

    # Step 5: Train-test split
    X = df_encoded.drop("Admit_Chance", axis=1)
    y = df_encoded["Admit_Chance"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Step 6: Feature scaling
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler