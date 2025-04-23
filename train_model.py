from src.data_processing import preprocess, data_loading
from src.models import building, training, visualization, evaluation
from src.config import MODEL_DIR
import joblib

def main():
    """
        main function that orchestrates the entire machine learning pipeline for training a model.
        It includes loading and preprocessing data, building the model, training it, evaluating its performance,
    """
    print("[INFO] Loading and preprocessing data...")
    df = data_loading.load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess.preprocess_data(df)

    print("[INFO] Building sklearn model...")
    model = building.build_ann()

    print("[INFO] Training sklearn model...")
    model = training.train_model(model, X_train, y_train)

    print("[INFO] Evaluating model...")
    y_pred = evaluation.evaluate_model(model, X_test, y_test)

    print("[INFO] Visualizing loss curve...")
    visualization.plot_loss_curve(model)

    print("[INFO] Saving model and scaler...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "mlp_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

if __name__ == "__main__":
    main()
