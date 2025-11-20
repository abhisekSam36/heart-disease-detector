import pandas as pd
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
