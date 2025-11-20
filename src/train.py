import joblib
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
