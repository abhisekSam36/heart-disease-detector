import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def get_feature_names(model):
    """
     robustly extracts feature names from the preprocessor
    """
    try:
        # For scikit-learn >= 1.0, this single line handles everything (numeric, one-hot, and passthrough)
        return model.named_steps['preprocessor'].get_feature_names_out()
    except Exception as e:
        print(f"Warning: Could not get feature names automatically: {e}")
        return []

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    print("\n" + "="*30)
    print("       MODEL PERFORMANCE       ")
    print("="*30)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred))
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    print("\n" + "="*30)
    print("      FEATURE IMPORTANCE       ")
    print("="*30)
    
    try:
        # Get feature importances from the random forest classifier
        importances = model.named_steps['classifier'].feature_importances_
        feature_names = get_feature_names(model)
        
        # Clean up feature names (remove "num__", "cat__", "remainder__" prefixes for readability)
        clean_names = [name.split('__')[-1] for name in feature_names]
        
        if len(clean_names) == len(importances):
            feat_df = pd.DataFrame({'Feature': clean_names, 'Importance': importances})
            print(feat_df.sort_values(by='Importance', ascending=False).head(10).to_string(index=False))
        else:
            print(f"Length mismatch: {len(clean_names)} names vs {len(importances)} importance values.")
    except Exception as e:
        print(f"Could not extract feature importance: {e}")