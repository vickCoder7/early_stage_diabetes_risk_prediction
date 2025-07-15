import numpy as np
import pandas as pd
import pickle
import streamlit as st

# Load your model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

def diabetes_prediction(input_data):
    """
    Predict diabetes from input list.
    """
    input_data = np.asarray(input_data).reshape(1, -1)
    prediction = loaded_model.predict(input_data)

    if prediction == 0:
        return "Patient is non-diabetic"
    else:
        return "Patient might be diabetic"


def main():
    st.title("🩺 Diabetes Prediction Web App")

    binary_map = {'Yes': 1, 'No': 0}
    gender_map = {'Male': 1, 'Female': 0}

    inputs = []
    age = st.text_input("Age")

    try:
        age_value = int(age)
        inputs.append(age_value)
    except ValueError:
        st.error("Please enter a valid numeric age")
        return

    # Gender
    gender = st.selectbox("Gender", ('Male', 'Female'))
    inputs.append(gender_map[gender])

    # Other binary features
    features = {
        "polyuria": "Produces large amount of urine",
        "polydipsia": "Excessive thirst or urge to drink fluids all the time",
        "sudden_weight_loss": "Sudden weight loss",
        "weakness": "Weakness",
        "polyphagia": "Excessive hunger or appetite",
        "genital_thrush": "Genital thrush",
        "visual_blurring": "Visual blurring",
        "itching": "Itching",
        "irritability": "Irritability",
        "delayed_healing": "Delayed Healing",
        "partial_paresis": "Partial paresis",
        "muscle_stiffness": "Muscle stiffness",
        "alopecia": "Hair Loss",
        "obesity": "Obesity",
    }

    for label in features.values():
        user_input = st.selectbox(label, ('No', 'Yes'))
        inputs.append(binary_map[user_input])
    
    inputs = pd.DataFrame(inputs)

    # Prediction button
    if st.button("Diabetes Diagnosis"):
        diagnosis = diabetes_prediction(inputs)
        st.success(diagnosis)

if __name__ == "__main__":
    main()
