❤️ Heart Disease Detection System

A complete Machine Learning application that predicts the likelihood of heart disease in patients based on medical attributes like age, cholesterol, and ECG results. This project includes a full ML pipeline (cleaning, training, evaluation) and an interactive Web App for doctors to use.
🚀 Project Overview
Objective: Build a system to classify patients as "Healthy" or "Heart Disease" (Binary Classification).
Model: Random Forest Classifier (Selected after comparing vs. Logistic Regression, SVM, and Gradient Boosting).
Performance: ~95% Accuracy on test data with high recall (sensitivity).
Interface: Interactive Streamlit Web App with diagnostic tools and data visualization.
🛠️ Installation & Setup
Follow these steps to run the project locally.
1. Clone the Repository
git clone [https://github.com/YOUR_USERNAME/heart-disease-detector.git](https://github.com/YOUR_USERNAME/heart-disease-detector.git)
cd heart-disease-detector


2. Set Up Environment
It is recommended to use Conda to manage dependencies.
# Create environment
conda create -n heart-disease-env python=3.10 -y

# Activate environment
conda activate heart-disease-env

# Install dependencies
pip install -r requirements.txt


3. Add Data
This project requires the dataset.csv file (not included in the repo for privacy reasons).
Download the dataset.
Place it in the folder: data/raw/dataset.csv.
🖥️ Usage
1. Train the Model (ML Pipeline)
Run the main script to clean data, train the Random Forest, evaluate performance, and save the model to the models/ directory.
python main.py


2. Run the Web App (Dashboard)
Launch the interactive dashboard to test predictions and view charts.
streamlit run app.py


3. Single Prediction (CLI)
Test the model on a dummy patient via the command line to verify it works.
python src/predict.py


📂 Project Structure
heart-disease-detection/
│
├── data/                   # Data storage
│   └── raw/                # Place dataset.csv here
│
├── models/                 # Saved trained models
│   └── heart_disease_model.pkl
│
├── notebooks/              # Jupyter notebooks for analysis
│   └── 01_analysis.ipynb   # EDA and visualization
│
├── src/                    # Source code
│   ├── preprocessing.py    # Data cleaning & transformation pipeline
│   ├── train.py            # Model training logic
│   ├── evaluate.py         # Evaluation metrics & feature importance
│   └── predict.py          # Inference script for single predictions
│
├── app.py                  # Streamlit Web Application
├── main.py                 # Main execution script
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation


📊 Model Performance
We evaluated multiple algorithms and selected Random Forest for its robustness and ability to handle mixed data types.
Metric
Score
Meaning
Accuracy
94.96%
Overall correctness of predictions.
Recall
96.95%
Ability to catch positive heart disease cases (Crucial for medical apps).
F1-Score
95.49%
Balance between precision and recall.

Key Predictors
The model identified the following features as most critical for detection:
ST Slope (Upward): Strongest indicator of a healthy heart during exercise.
Max Heart Rate: Lower max heart rates often correlate with heart disease.
Oldpeak: ST depression induced by exercise relative to rest.
Chest Pain Type: Specifically asymptomatic cases (Type 4) were highly correlated with disease.
🩺 Dataset Details
The dataset contains 1,190 patient records with 11 features:
age: Age of the patient (years)
sex: Male (1) or Female (0)
chest pain type: Typical, Atypical, Non-anginal, Asymptomatic
resting bp s: Resting blood pressure (mm Hg)
cholesterol: Serum cholesterol (mg/dl)
fasting blood sugar: > 120 mg/dl (1 = true; 0 = false)
resting ecg: Resting electrocardiogram results
max heart rate: Maximum heart rate achieved
exercise angina: Exercise-induced angina (1 = yes; 0 = no)
oldpeak: ST depression induced by exercise relative to rest
ST slope: The slope of the peak exercise ST segment
🤝 Contributing
Contributions are welcome! Here is how you can help:
Fork the repository.
Create a Feature Branch (git checkout -b feature/NewFeature).
Commit your changes (git commit -m 'Add NewFeature').
Push to the branch (git push origin feature/NewFeature).
Open a Pull Request.
📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
👤 Author
Your Name
GitHub: @YourUsername
LinkedIn: Your Name
