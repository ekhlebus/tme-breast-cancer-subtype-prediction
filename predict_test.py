#!/usr/bin/env python
# coding: utf-8

# This script load the trained model and makes a prediction for a single patient.

import pandas as pd
import pickle

# Load the model from the file model.bin
with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

patient_id = "TCGA-XX-YYYY"

# Example patient immune composition (CIBERSORT-style fractions)
patient = {
    "B.cells.naive": 0.02,
    "B.cells.memory": 0.05,
    "Plasma.cells": 0.01,
    "T.cells.CD8": 0.12,
    "T.cells.CD4.naive": 0.03,
    "T.cells.CD4.memory.resting": 0.18,
    "T.cells.CD4.memory.activated": 0.04,
    "T.cells.follicular.helper": 0.02,
    "T.cells.regulatory..Tregs.": 0.03,
    "T.cells.gamma.delta": 0.01,
    "NK.cells.resting": 0.06,
    "NK.cells.activated": 0.02,
    "Monocytes": 0.09,
    "Macrophages.M0": 0.15,
    "Macrophages.M1": 0.07,
    "Macrophages.M2": 0.06,
    "Dendritic.cells.resting": 0.01,
    "Dendritic.cells.activated": 0.01,
    "Mast.cells.resting": 0.01,
    "Mast.cells.activated": 0.00,
    "Eosinophils": 0.00,
    "Neutrophils": 0.01,
    "age": 57
}

# Convert to DataFrame
X_patient = pd.DataFrame([patient])

# Predict probability of Basal-like subtype
response = pipeline.predict_proba(X_patient)[0, 1]
print(f"Basal probability: {response:.3f}")

# Interpret model output
if response >= 0.5:
    print(f"Patient {patient_id}: predicted Basal-like subtype")
else:
    print(f"Patient {patient_id}: predicted Luminal (A/B) subtype")
