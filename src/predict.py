import joblib
import pandas as pd

def load_model(filepath):
    return joblib.load(filepath)

def predict_patient(model, patient_data):
    df = pd.DataFrame([patient_data])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    return prediction, probability

if __name__ == "__main__":
    # Test the prediction script independently
    try:
        model = load_model('models/heart_disease_model.pkl')
        sample_patient = {
            'age': 54, 'sex': 1, 'chest pain type': 4, 'resting bp s': 150,
            'cholesterol': 195, 'fasting blood sugar': 0, 'resting ecg': 0,
            'max heart rate': 122, 'exercise angina': 0, 'oldpeak': 0.0, 'ST slope': 1
        }
        pred, prob = predict_patient(model, sample_patient)
        print(f"Prediction: {'Heart Disease' if pred==1 else 'Normal'} ({prob:.1%})")
    except FileNotFoundError:
        print("Model not found. Run main.py first.")
