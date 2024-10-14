from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os
import sys
import pickle
import numpy as np
import pandas as pd
import streamlit as st

n_samples = 1000
np.random.seed(47)
study_hours = np.random.randint(0,100,n_samples)
practice_tests = np.random.randint(0,25, n_samples)
attendance_rate = np.random.randint(0,100,n_samples)
sleep_hours = np.random.randint(0,24,n_samples)
assignment_scores = np.random.randint(0,100,n_samples)

df = pd.DataFrame({
    'study_hours': study_hours,
    'practice_tests': practice_tests,
    'attendance_rate': attendance_rate,
    'sleep_hours': sleep_hours,
    'assignment_scores': assignment_scores
})


w1 = 0.5  # Weight for study hours
w2 = 0.2  # Weight for practice tests
w3 = 0.05 # Weight for attendance rate
w4 = 0.05  # Weight for sleep hours
w5 = 0.2 # Weight for assignment scores

df['normalized_study_hours'] = df['study_hours'] / 100
df['normalized_practice_tests'] = df['practice_tests'] / 25
df['normalized_attendance_rate'] = df['attendance_rate'] / 100
df['normalized_sleep_hours'] = df['sleep_hours'] / 24
df['normalized_assignment_scores'] = df['assignment_scores'] / 100

df['exam_scores'] = (
    w1 * df['normalized_study_hours'] +
    w2 * df['normalized_practice_tests'] +
    w3 * df['normalized_attendance_rate'] +
    w4 * df['normalized_sleep_hours'] +
    w5 * df['normalized_assignment_scores']
) * 100  # Multiply by 100 to get scores out of 100

df.head()

x = df[['normalized_study_hours', 'normalized_practice_tests',
        'normalized_attendance_rate', 'normalized_sleep_hours',
        'normalized_assignment_scores']]
y = df['exam_scores']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

x_train

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

msr = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
accuracy = model.score(x_test, y_test)*100
print("Mean Squared Error:", msr)
print("R2 Score:", r2)
print("Accuracy:", accuracy)

with open('linear_regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)


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

