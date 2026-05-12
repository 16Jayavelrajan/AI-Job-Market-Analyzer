import streamlit as st
import pandas as pd
import pickle
import ast
from collections import Counter
import os

# ---------------------------------------------------
# GET CURRENT FILE DIRECTORY
# ---------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------
# CREATE PATHS
# ---------------------------------------------------

DATA_PATH = os.path.join(BASE_DIR, "..", "data", "cleaned_jobs.csv")

MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "job_classifier.pkl")

VECTORIZER_PATH = os.path.join(BASE_DIR, "..", "models", "tfidf_vectorizer.pkl")

# ---------------------------------------------------
# PAGE TITLE
# ---------------------------------------------------

st.title("AI Job Market Analyzer")

st.write(
    "AI-powered dashboard for job market analysis and experience prediction."
)

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

df = pd.read_csv(DATA_PATH)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

with open(VECTORIZER_PATH, "rb") as file:
    vectorizer = pickle.load(file)

# ---------------------------------------------------
# DATASET PREVIEW
# ---------------------------------------------------

st.subheader("Dataset Preview")

st.dataframe(df.head())

# ---------------------------------------------------
# EXPERIENCE LEVEL DISTRIBUTION
# ---------------------------------------------------

st.subheader("Experience Level Distribution")

st.bar_chart(df['experienceLevel'].value_counts())

# ---------------------------------------------------
# TOP SKILLS
# ---------------------------------------------------

all_skills = []

for skills in df['skills_found']:

    if isinstance(skills, str):
        skills = ast.literal_eval(skills)

    all_skills.extend(skills)

skill_counts = Counter(all_skills)

skill_df = pd.DataFrame(
    skill_counts.items(),
    columns=['Skill', 'Count']
)

skill_df = skill_df.sort_values(
    by='Count',
    ascending=False
).head(10)

st.subheader("Top 10 Skills")

st.bar_chart(skill_df.set_index('Skill'))

# ---------------------------------------------------
# PREDICTION SECTION
# ---------------------------------------------------

st.subheader("Predict Experience Level")

user_input = st.text_area(
    "Enter Job Description"
)

if st.button("Predict"):

    cleaned_input = user_input.lower()

    input_vector = vectorizer.transform([cleaned_input])

    prediction = model.predict(input_vector)

    st.success(
        f"Predicted Experience Level: {prediction[0]}"
    )