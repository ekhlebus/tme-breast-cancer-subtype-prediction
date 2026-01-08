#!/usr/bin/env python
# coding: utf-8

# 1. Run this script to start a local web service for subtype prediction: 
#         python predict.py
#    Note: If we don't want to reload on every code change, we can run instead: 
#         uvicorn predict:app --host 0.0.0.0 --port 9696 --reload

# 2. Go to http://localhost:9696/docs in browser to see the interactive API documentation.

# 3. To get the prediction we need to sent patient data as json not just a POST request:
#

import pandas as pd
import pickle

import uvicorn
from fastapi import FastAPI
from typing import Dict


# Define the application using FastAPI
app = FastAPI(title='subtype-prediction')

# Load the model from the file model.bin
with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

# Function to make a prediction for a single patient based on immune composition
def predict_single(patient: Dict[str, float]):
    # Convert JSON → DataFrame
    X_patient = pd.DataFrame([patient])

    # Predict probability of Basal-like subtype
    response = pipeline.predict_proba(X_patient)[0, 1]

    return response


# Add decorator to turn function into web service
# Tell FastAPI that we expect a dictionary as input
@app.post("/predict")  # This function will be accessible at address /predict using POST method
def predict(patient: Dict[str, float]):
    subtype_responce = predict_single(patient)
    
    return {
        "Basal_probability": float(subtype_responce),
        "Predicted_subtype": "Basal-like" if subtype_responce >= 0.5 else "Luminal (A/B)"
    }

# We are using "__main__" top-level script environment to run the app
if __name__ == "__main__":
    # Run the app on port 9696 at localhost
    uvicorn.run(app, host='0.0.0.0', port=9696) 