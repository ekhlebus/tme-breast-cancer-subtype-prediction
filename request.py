# Before running this script, make sure that the web service is running by executing:
#    ```python predict.py```

import requests

# Define the URL for the churn prediction model
url = 'http://localhost:9696/predict'

patient = {
  "B.cells.naive": 0.06,
  "B.cells.memory": 0.10,
  "Plasma.cells": 0.04,
  "T.cells.CD8": 0.04,
  "T.cells.CD4.naive": 0.07,
  "T.cells.CD4.memory.resting": 0.20,
  "T.cells.CD4.memory.activated": 0.02,
  "T.cells.follicular.helper": 0.05,
  "T.cells.regulatory..Tregs.": 0.02,
  "T.cells.gamma.delta": 0.00,
  "NK.cells.resting": 0.05,
  "NK.cells.activated": 0.01,
  "Monocytes": 0.06,
  "Macrophages.M0": 0.05,
  "Macrophages.M1": 0.03,
  "Macrophages.M2": 0.04,
  "Dendritic.cells.resting": 0.02,
  "Dendritic.cells.activated": 0.01,
  "Mast.cells.resting": 0.10,
  "Mast.cells.activated": 0.03,
  "Eosinophils": 0.02,
  "Neutrophils": 0.01,
  "age": 62
}

response = requests.post(url, json=patient)
prob = response.json()

print('response:', prob)

if prob['Basal_probability'] >= 0.5:
   print('Basal-like subtype predicted')
else:
   print('Luminal (A/B) subtype predicted')
