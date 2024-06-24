import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load('best_model.pkl')

# Define a function to make predictions
def predict_heart_disease(data):
    df = pd.DataFrame(data, index=[0])
    prediction = model.predict(df)
    probability = model.predict_proba(df)[:, 1]
    return prediction[0], probability[0]

# Define the input fields for the user
st.title("Heart Disease Prediction")
st.write("Enter the details of the patient to predict the likelihood of heart disease.")

# Collect user input
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.number_input("Chest Pain Type", min_value=0, max_value=3, value=1)
trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
chol = st.number_input("Serum Cholestoral in mg/dl", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["True", "False"])
restecg = st.number_input("Resting Electrocardiographic Results", min_value=0, max_value=2, value=1)
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
oldpeak = st.number_input("ST depression induced by exercise", min_value=0.0, max_value=10.0, value=1.0)
slope = st.number_input("Slope of the peak exercise ST segment", min_value=0, max_value=2, value=1)
ca = st.number_input("Number of major vessels (0-3) colored by flourosopy", min_value=0, max_value=3, value=0)
thal = st.number_input("Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)", min_value=1, max_value=3, value=2)

# Convert inputs to a DataFrame with correct types
data = {
    'age': [int(age)],
    'sex': [1 if sex == "Male" else 0],
    'cp': [int(cp)],
    'trestbps': [int(trestbps)],
    'chol': [int(chol)],
    'fbs': [1 if fbs == "True" else 0],
    'restecg': [int(restecg)],
    'thalach': [int(thalach)],
    'exang': [1 if exang == "Yes" else 0],
    'oldpeak': [float(oldpeak)],
    'slope': [int(slope)],
    'ca': [int(ca)],
    'thal': [int(thal)]
}
input_data = pd.DataFrame(data)

# Ensure there are no missing values
if input_data.isnull().values.any():
    st.error("Please fill all the fields correctly.")
else:
    # Make prediction
    try:
        prediction = model.predict(input_data)
        prediction_prob = model.predict_proba(input_data)

        # Display results
        st.write(f"Prediction: {'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'}")
        st.write(f"Prediction Probability: {prediction_prob[0][1]*100:.2f}% chance of Heart Disease")
    except Exception as e:
        st.error(f"An error occurred: {e}")