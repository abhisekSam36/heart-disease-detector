import os
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
