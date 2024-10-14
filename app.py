from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os
import sys
import pickle
import numpy as np
import pandas as pd
import streamlit as st

with open('linear_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)


def normalize_values(study_hours, practice_tests, attendance_rate, sleep_hours, assignment_scores):
    max_study_hours = 100
    max_practice_tests = 25
    max_attendance_rate = 100
    max_sleep_hours = 24
    max_assignment_scores = 100

    normalized_study_hours = study_hours / max_study_hours
    normalized_practice_tests = practice_tests / max_practice_tests
    normalized_attendance_rate = attendance_rate / max_attendance_rate
    normalized_sleep_hours = sleep_hours / max_sleep_hours
    normalized_assignment_scores = assignment_scores / max_assignment_scores

    return np.array([[
        normalized_study_hours,
        normalized_practice_tests,
        normalized_attendance_rate,
        normalized_sleep_hours,
        normalized_assignment_scores
    ]])

# Streamlit app
st.title("Exam Score Prediction App")

st.write("Enter the following details to predict your exam score:")

# Input fields for user to enter feature values
study_hours = st.number_input("Study Hours", min_value=0, max_value=100, step=1)
practice_tests = st.number_input("Practice Tests Taken", min_value=0, max_value=25, step=1)
attendance_rate = st.number_input("Attendance Rate (%)", min_value=0, max_value=100, step=1)
sleep_hours = st.number_input("Average Sleep Hours per Day", min_value=0, max_value=24, step=1)
assignment_scores = st.number_input("Assignment Scores (%)", min_value=0, max_value=100, step=1)

# Button to predict the exam score
if st.button("Predict Exam Score"):
    # Normalize input values
    normalized_input = normalize_values(study_hours, practice_tests, attendance_rate, sleep_hours, assignment_scores)

    # Predict the exam score using the model
    predicted_score = model.predict(normalized_input)[0]

    # Display the predicted score
    st.success(f"Predicted Exam Score: {predicted_score:.2f}/100")