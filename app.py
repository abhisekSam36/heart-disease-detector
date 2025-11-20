import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(page_title="Heart Disease Assistant", page_icon="❤️", layout="wide")

# Load Model and Data
@st.cache_resource
def load_resources():
    model = joblib.load('models/heart_disease_model.pkl')
    # Load raw data for the analysis tab
    data = pd.read_csv('data/raw/dataset.csv') 
    return model, data

model, df = load_resources()

st.title("❤️ Heart Disease Diagnostic Assistant")

# Create Tabs
tab1, tab2 = st.tabs(["🏥 Patient Diagnosis", "📊 Data Insights"])

# --- TAB 1: PREDICTION FORM ---
with tab1:
    st.subheader("Patient Vitals")
    st.write("Enter the patient's details below to generate a risk assessment.")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", 20, 100, 50)
            sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x==1 else "Female")
            chest_pain = st.selectbox("Chest Pain Type", [1, 2, 3, 4], 
                                      format_func=lambda x: f"Type {x}: " + 
                                      {1:"Typical Angina", 2:"Atypical Angina", 3:"Non-Anginal", 4:"Asymptomatic"}[x])
            resting_bp = st.number_input("Resting BP (mm Hg)", 80, 200, 120)
        
        with col2:
            cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
            fasting_bs = st.selectbox("Fasting BS > 120 mg/dl?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            resting_ecg = st.selectbox("Resting ECG", [0, 1, 2], 
                                       format_func=lambda x: {0:"Normal", 1:"ST-T Abnormality", 2:"LV Hypertrophy"}[x])
            max_hr = st.number_input("Max Heart Rate", 60, 220, 150)

        with col3:
            exercise_angina = st.selectbox("Exercise Induced Angina?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0, 0.0, step=0.1)
            st_slope = st.selectbox("ST Slope", [1, 2, 3], format_func=lambda x: {1:"Upward", 2:"Flat", 3:"Downward"}[x])

        predict_btn = st.form_submit_button("Analyze Patient")
    
    if predict_btn:
        # Prepare data
        input_data = pd.DataFrame([{
            'age': age, 'sex': sex, 'chest pain type': chest_pain, 'resting bp s': resting_bp,
            'cholesterol': cholesterol, 'fasting blood sugar': fasting_bs, 'resting ecg': resting_ecg,
            'max heart rate': max_hr, 'exercise angina': exercise_angina, 'oldpeak': oldpeak, 'ST slope': st_slope
        }])
        
        # Predict
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]
        
        st.markdown("---")
        if pred == 1:
            st.error(f"🚨 **High Risk Detected** (Probability: {prob:.1%})")
            st.info("Recommendation: Immediate cardiology consultation required.")
        else:
            st.success(f"✅ **Normal Result** (Risk Probability: {prob:.1%})")

# --- TAB 2: VISUALIZATIONS ---
with tab2:
    st.header("Dataset Analysis")
    st.write("Explore the relationships in the training data.")
    
    # 1. Correlation Heatmap
    st.subheader("1. What factors are correlated?")
    fig, ax = plt.subplots(figsize=(10, 6))
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    
    # 2. Interactive Histogram
    st.subheader("2. Distribution of Vitals")
    feature_to_plot = st.selectbox("Select Feature to Visualize:", ['age', 'cholesterol', 'max heart rate', 'resting bp s'])
    
    fig2, ax2 = plt.subplots()
    sns.histplot(data=df, x=feature_to_plot, hue='target', multiple="stack", palette="pastel", ax=ax2)
    plt.title(f"{feature_to_plot.capitalize()} Distribution (Normal vs Heart Disease)")
    st.pyplot(fig2)