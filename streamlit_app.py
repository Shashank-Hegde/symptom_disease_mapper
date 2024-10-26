import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rapidfuzz import process
import torch
import zipfile
import requests
import shutil

st.title("🩺 O-Health LLM")
st.write("""
    Enter your symptoms below, and the platform will suggest possible diseases with associated probabilities.
""")

# URL to your model zip file hosted externally
MODEL_URL = "https://www.dropbox.com/scl/fi/pt0anz8mefta72rxjpyol/medical-bert-symptom-ner.zip?rlkey=ovtc18kbhw8njs3qwplcc76do&st=6y26kyl7&dl=1"

# Path to the model directory
model_dir = 'medical-bert-symptom-ner'  # Path where the model will be extracted

def download_and_unzip_model(model_url, model_dir):
    if not os.path.exists(model_dir):
        st.info("Downloading the model. Please wait...")
        # Download the model zip file
        with st.spinner('Downloading model...'):
            response = requests.get(model_url, stream=True)
            if response.status_code == 200:
                with open('model.zip', 'wb') as out_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            out_file.write(chunk)
            else:
                st.error("Failed to download the model. Please check the URL.")
                st.stop()
        # Unzip the model
        try:
            with zipfile.ZipFile('model.zip', 'r') as zip_ref:
                zip_ref.extractall('.')
            st.success("Model downloaded and extracted successfully.")
        except zipfile.BadZipFile:
            st.error("Downloaded file is not a valid zip file.")
            st.stop()
        finally:
            os.remove('model.zip')

# Download and unzip the model if it doesn't exist
download_and_unzip_model(MODEL_URL, model_dir)

# Check if the model directory exists after extraction
if not os.path.exists(model_dir):
    st.error(f"Model directory '{model_dir}' not found after extraction.")
    st.stop()

# Load the tokenizer and model using caching
@st.cache_resource
def load_ner_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(model_dir, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=device)

ner_pipeline = load_ner_pipeline()
st.sidebar.success("NER model loaded successfully!")

# Load 'disease_symptom_mapping.csv'
if not os.path.exists("disease_symptom_mapping.csv"):
    st.error("'disease_symptom_mapping.csv' not found in the current directory.")
    st.stop()
df_disease_symptom = pd.read_csv("disease_symptom_mapping.csv")

# Prepare a list of known symptoms
known_symptoms = df_disease_symptom['SymptomName'].unique()

# Function to match extracted symptoms with known symptoms
def match_symptoms(extracted_symptoms):
    matched_symptoms = set()
    for symptom in extracted_symptoms:
        match = process.extractOne(symptom, known_symptoms, score_cutoff=80)
        if match:
            matched_symptoms.add(match[0])
    return matched_symptoms

# User input
user_input = st.text_area("📋 Enter your symptoms:", height=150,
                          placeholder="e.g., I have been experiencing fever and cough for the past few days.")

# Diagnose button
if st.button("Diagnose"):
    if user_input.strip() == "":
        st.warning("Please enter your symptoms for diagnosis.")
    else:
        with st.spinner("Analyzing your symptoms..."):
            entities = ner_pipeline(user_input)
            if not entities:
                st.error("No symptoms detected. Please try again.")
            else:
                # Extract unique symptoms
                extracted_symptoms = set([entity['word'].title() for entity in entities])
                symptoms = match_symptoms(extracted_symptoms)
                if not symptoms:
                    st.error("No matching symptoms found in our database.")
                else:
                    st.success(f"Extracted Symptoms: {', '.join(symptoms)}")
                    # Create disease-symptom mapping
                    disease_symptom_map = df_disease_symptom.groupby('DiseaseName')['SymptomName'].apply(set).to_dict()

                    # Assume prior probabilities are equal for all diseases
                    prior = 1 / len(disease_symptom_map)

                    # Calculate likelihoods and posterior probabilities
                    disease_scores = {}
                    for disease, symptoms_set in disease_symptom_map.items():
                        matched = symptoms.intersection(symptoms_set)
                        total_symptoms = len(symptoms_set)
                        if total_symptoms == 0:
                            continue
                        # Simple likelihood estimation
                        likelihood = len(matched) / total_symptoms
                        # Posterior probability proportional to likelihood * prior
                        posterior = likelihood * prior
                        disease_scores[disease] = posterior

                    if disease_scores:
                        # Normalize the probabilities
                        total = sum(disease_scores.values())
                        for disease in disease_scores:
                            disease_scores[disease] = round((disease_scores[disease] / total) * 100, 2)
                        # Sort diseases by probability
                        sorted_diseases = dict(sorted(disease_scores.items(), key=lambda item: item[1], reverse=True))
                        # Display results
                        st.subheader("🩺 Probable Diseases:")
                        for disease, prob in sorted_diseases.items():
                            st.write(f"**{disease}**: {prob}%")

                        # Plot bar chart
                        fig, ax = plt.subplots(figsize=(8, 4))
                        sns.barplot(x=list(sorted_diseases.keys()), y=list(sorted_diseases.values()), ax=ax)
                        ax.set_xlabel("Disease")
                        ax.set_ylabel("Probability (%)")
                        ax.set_title("Probable Diseases")
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                    else:
                        st.info("No probable diseases found based on the entered symptoms.")
