import os

# Define the project structure and file contents
files = {
    # 1. Dependencies
    "requirements.txt": """pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
""",

    # 2. Preprocessing Script
    "src/preprocessing.py": """import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_data(filepath):
    return pd.read_csv(filepath)

def clean_data(df):
    # Replace 0s with median for specific columns based on EDA
    df['cholesterol'] = df['cholesterol'].replace(0, df['cholesterol'].median())
    df['resting bp s'] = df['resting bp s'].replace(0, df['resting bp s'].median())
    return df

def get_preprocessor():
    numerical_features = ['age', 'resting bp s', 'cholesterol', 'max heart rate', 'oldpeak']
    categorical_features = ['chest pain type', 'resting ecg', 'ST slope']

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    return preprocessor

def split_data(df, target_col='target', test_size=0.2):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=42)
""",

    # 3. Training Script (Random Forest)
    "src/train.py": """import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from src.preprocessing import get_preprocessor

def train_model(X_train, y_train):
    print("Initializing Random Forest Classifier...")
    preprocessor = get_preprocessor()

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    print("Fitting model to training data...")
    model.fit(X_train, y_train)
    return model

def save_model(model, filepath):
    joblib.dump(model, filepath)
    print(f"✅ Model saved successfully to {filepath}")
""",

    # 4. Evaluation Script
    "src/evaluate.py": """import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def get_feature_names(model):
    try:
        preprocessor = model.named_steps['preprocessor']
        num_names = preprocessor.transformers_[0][2]
        cat_transformer = preprocessor.transformers_[1][1]
        cat_names = cat_transformer.get_feature_names_out(preprocessor.transformers_[1][2])
        return list(num_names) + list(cat_names)
    except:
        return []

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    print("\\n" + "="*30)
    print("       MODEL PERFORMANCE       ")
    print("="*30)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print("\\n--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred))
    print("\\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    print("\\n" + "="*30)
    print("      FEATURE IMPORTANCE       ")
    print("="*30)
    try:
        importances = model.named_steps['classifier'].feature_importances_
        feature_names = get_feature_names(model)
        if feature_names:
            feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            print(feat_df.sort_values(by='Importance', ascending=False).head(10).to_string(index=False))
    except Exception as e:
        print(f"Could not extract feature importance: {e}")
""",

    # 5. Prediction Script
    "src/predict.py": """import joblib
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
""",

    # 6. Main Execution Script
    "main.py": """import os
from src.preprocessing import load_data, clean_data, split_data
from src.train import train_model, save_model
from src.evaluate import evaluate_model

DATA_PATH = 'data/raw/dataset.csv'
MODEL_PATH = 'models/heart_disease_model.pkl'

def main():
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: File not found at {DATA_PATH}")
        print("   Please move 'dataset.csv' into the 'data/raw/' folder.")
        return

    print("--- 1. Loading & Cleaning ---")
    df = load_data(DATA_PATH)
    df = clean_data(df)

    print("--- 2. Splitting Data ---")
    X_train, X_test, y_train, y_test = split_data(df)

    print("--- 3. Training Model ---")
    model = train_model(X_train, y_train)

    print("--- 4. Evaluation ---")
    evaluate_model(model, X_test, y_test)

    print("--- 5. Saving Model ---")
    os.makedirs('models', exist_ok=True)
    save_model(model, MODEL_PATH)

if __name__ == "__main__":
    main()
""",
    
    # 7. Init file
    "src/__init__.py": ""
}

def setup():
    # Create directories
    directories = ["data/raw", "data/processed", "models", "notebooks", "src"]
    for d in directories:
        os.makedirs(d, exist_ok=True)
        print(f"Created directory: {d}")

    # Write files
    for filepath, content in files.items():
        with open(filepath, "w") as f:
            f.write(content)
        print(f"Created file: {filepath}")

    print("\n✅ Project structure created successfully!")
    print("👉 NEXT STEP: Move your 'dataset.csv' file into the 'data/raw' folder.")

if __name__ == "__main__":
    setup()