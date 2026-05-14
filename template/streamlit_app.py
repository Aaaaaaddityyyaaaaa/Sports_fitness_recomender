import streamlit as st
import requests

st.title("Athlete Performance Predictor")

with st.form("athlete_form"):
    st.subheader("Personal Info")
    age    = st.number_input("Age",          min_value=10,   max_value=80,    step=1,   value=25)
    height = st.number_input("Height (cm)",  min_value=100.0, max_value=250.0, step=0.1, value=170.0)
    weight = st.number_input("Weight (kg)",  min_value=30.0,  max_value=200.0, step=0.1, value=70.0)
    bmi    = st.number_input("BMI",          min_value=10.0,  max_value=60.0,  step=0.1, value=24.0)

    st.subheader("Training")
    training_hours_per_week = st.number_input("Training Hours Per Week", min_value=0.0,  max_value=40.0,  step=0.5,  value=10.0)
    training_intensity      = st.number_input("Training Intensity",      min_value=0.0,  max_value=10.0,  step=0.1,  value=5.0)
    experience_years        = st.number_input("Experience Years",        min_value=0.0,  max_value=40.0,  step=0.5,  value=3.0)
    injury_history          = st.selectbox("Injury History",             options=[0, 1], format_func=lambda x: "Yes" if x else "No")

    st.subheader("Health & Recovery")
    sleep_hours          = st.number_input("Sleep Hours",            min_value=0.0,  max_value=24.0,  step=0.1,  value=7.0)
    hydration_level      = st.number_input("Hydration Level (L)",    min_value=0.0,  max_value=10.0,  step=0.1,  value=2.5)
    resting_heart_rate   = st.number_input("Resting Heart Rate",     min_value=30.0, max_value=120.0, step=1.0,  value=60.0)
    average_heart_rate   = st.number_input("Average Heart Rate",     min_value=30.0, max_value=220.0, step=1.0,  value=75.0)
    stress_level         = st.number_input("Stress Level",           min_value=0.0,  max_value=10.0,  step=0.1,  value=5.0)
    recovery_score       = st.number_input("Recovery Score",         min_value=0.0,  max_value=100.0, step=0.1,  value=60.0)
    body_fat_percentage  = st.number_input("Body Fat Percentage",    min_value=0.0,  max_value=60.0,  step=0.1,  value=20.0)
    muscle_mass          = st.number_input("Muscle Mass (kg)",       min_value=0.0,  max_value=100.0, step=0.1,  value=40.0)

    st.subheader("Mental & Motivation")
    mental_focus_level = st.number_input("Mental Focus Level", min_value=0.0, max_value=100.0, step=0.1, value=70.0)
    motivation_level   = st.number_input("Motivation Level",   min_value=0.0, max_value=100.0, step=0.1, value=70.0)

    st.subheader("Physiological")
    vo2_max          = st.number_input("VO2 Max",             min_value=0.0,  max_value=100.0,  step=0.1,  value=45.0)
    reaction_time_ms = st.number_input("Reaction Time (ms)",  min_value=100.0, max_value=600.0, step=1.0,  value=300.0)

    st.subheader("Performance Scores")
    speed_score       = st.number_input("Speed Score",       min_value=0.0, max_value=100.0, step=0.1, value=50.0)
    endurance_score   = st.number_input("Endurance Score",   min_value=0.0, max_value=100.0, step=0.1, value=50.0)
    flexibility_score = st.number_input("Flexibility Score", min_value=0.0, max_value=100.0, step=0.1, value=50.0)

    st.subheader("Nutrition")
    daily_caloric_intake = st.number_input("Daily Caloric Intake (kcal)", min_value=500.0,  max_value=6000.0, step=10.0, value=2500.0)
    protein_intake       = st.number_input("Protein Intake (g)",          min_value=0.0,    max_value=500.0,  step=1.0,  value=150.0)

    submitted = st.form_submit_button("Predict Performance")

if submitted:
    
    payload = {
    "Age":                     age,
    "Height_cm":               height,
    "Weight_kg":               weight,
    "BMI":                     bmi,
    "Training_Hours_Per_Week": training_hours_per_week,
    "Training_Intensity":      training_intensity,
    "Sleep_Hours":             sleep_hours,
    "Hydration_Level":         hydration_level,
    "VO2_Max":                 vo2_max,
    "Average_Heart_Rate":      average_heart_rate,
    "Resting_Heart_Rate":      resting_heart_rate,
    "Mental_Focus_Level":      mental_focus_level,
    "Stress_Level":            stress_level,
    "Motivation_Level":        motivation_level,
    "Body_Fat_Percentage":     body_fat_percentage,
    "Muscle_Mass":             muscle_mass,
    "Daily_Caloric_Intake":    daily_caloric_intake,
    "Protein_Intake":          protein_intake,
    "Recovery_Score":          recovery_score,
    "Reaction_Time_ms":        reaction_time_ms,
    "Speed_Score":             speed_score,
    "Endurance_Score":         endurance_score,
    "Flexibility_Score":       flexibility_score,
    "Experience_Years":        experience_years,
    "Injury_History":          injury_history,

    }

    response = requests.post("http://127.0.0.1:8000/predict", json=payload)

    if response.status_code == 200:
        result = response.json()
        st.success(f"Performance Score: {result['performance_metric']:.2f}")
        st.info(f"Tier: {result['tier']}")
        st.write(f"AI precription {result["prescription"]}")
    else:
        st.error(f"Error {response.status_code}: {response.text}")
