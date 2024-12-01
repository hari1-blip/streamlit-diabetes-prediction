import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(page_title="Diabetes Prediction App", layout="wide")

# Load and preprocess data
@st.cache_data
def load_data():
    # Load the diabetes dataset
    df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv')
    df.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                  'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    return df

# Train model
@st.cache_resource
def train_model(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Get predictions
    y_pred = model.predict(X_test_scaled)
    
    return model, scaler, accuracy_score(y_test, y_pred)

def main():
    st.title("Diabetes Prediction System")
    st.write("Enter patient information to predict diabetes risk")
    
    # Load data and train model
    df = load_data()
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    model, scaler, accuracy = train_model(X, y)
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Patient Information")
        # Input fields
        pregnancies = st.slider("Number of Pregnancies", 0, 17, 3)
        glucose = st.slider("Glucose Level", 0, 200, 120)
        blood_pressure = st.slider("Blood Pressure", 0, 122, 70)
        skin_thickness = st.slider("Skin Thickness", 0, 100, 20)
        insulin = st.slider("Insulin", 0, 846, 79)
        bmi = st.number_input("BMI", 0.0, 67.1, 31.4)
        dpf = st.number_input("Diabetes Pedigree Function", 0.078, 2.42, 0.3725)
        age = st.slider("Age", 21, 81, 33)
        
        # Create a button for prediction
        if st.button("Predict Diabetes Risk"):
            # Prepare input data
            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                                  insulin, bmi, dpf, age]])
            # Scale the input
            input_scaled = scaler.transform(input_data)
            # Make prediction
            prediction = model.predict(input_scaled)
            prediction_proba = model.predict_proba(input_scaled)
            
            # Display result
            st.subheader("Prediction Result")
            if prediction[0] == 1:
                st.error("High Risk of Diabetes")
            else:
                st.success("Low Risk of Diabetes")
            
            # Display probability
            st.write(f"Probability of Diabetes: {prediction_proba[0][1]:.2%}")
            
            # Create gauge chart for probability
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prediction_proba[0][1] * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Risk Level"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 33], 'color': "lightgreen"},
                        {'range': [33, 66], 'color': "yellow"},
                        {'range': [66, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': prediction_proba[0][1] * 100
                    }
                }
            ))
            st.plotly_chart(fig)


if __name__ == "__main__":
    main()
