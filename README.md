# UCLA Admission Predictor

This project predicts a student's likelihood of being admitted to UCLA based on academic and profile-related features such as GRE score, TOEFL score, CGPA, SOP/LOR strength, research experience, and university rating.

The model uses a trained **Multi-Layer Perceptron (MLP)** classifier built with **scikit-learn**, and includes a **Streamlit UI** for user interaction via manual input.

## Live Demo

https://ucla-admission-prediction-yuozl4rzy3dryn2gkuspwc.streamlit.app/

## Model Architecture

- **Model Type:** MLPClassifier (scikit-learn)
- **Architecture:** 2 hidden layers (3 neurons each)
- **Scaler:** MinMaxScaler
- **Target:** Binary — Admitted (`1`) / Rejected (`0`)  
- **Threshold:** Admission Chance ≥ 0.8 ⇒ `Admitted`


## How To Run

1. Clone the repo
```bash
git clone https://github.com/your-username/UCLA-Admission-Prediction.git
cd UCLA-Admission-Prediction
```

2. Install Requirements
```bash
pip install -r requirements.txt
```

3. Train the model (optional if using pre-trained model)
```bash
python train_model.py
```

4. Launch the Streamlit app
```bash
streamlit run app.py
```

## Features
- Manual user input form
- Preprocessing with one-hot encoding and scaling
- Trained MLPClassifier using scikit-learn
- Real-time predictions
- Clean and simple UI with results display

## Example Prediction

GRE	TOEFL	CGPA	SOP	LOR	Research	Prediction
320	110	8.5	4.5	4.0	Yes	✅ Admitted

## Future Enhancements

- Add confidence scores or probability gauge
- Add Keras version toggle for advanced users


## License
This project is licensed under the MIT License. See LICENSE for details
