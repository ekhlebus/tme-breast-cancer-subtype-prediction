#!/usr/bin/env python
# coding: utf-8

# This script trains the final model on the full dataset.
# Model evaluation was performed separately.

import pickle

import pandas as pd
import numpy as np
import sklearn

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


# Print versions of the libraries we are using
print(f'pandas=={pd.__version__}')
print(f'numpy=={np.__version__}')
print(f'sklearn=={sklearn.__version__}')

# Define immune columns prefixes
immune_prefixes = (
    "B.cells",
    "Plasma",
    "T.cells",
    "NK.cells",
    "Monocytes",
    "Macrophages",
    "Dendritic",
    "Mast.cells",
    "Eosinophils",
    "Neutrophils",
)

# Make function for data downloading and preparation
def load_data():
    clin_data = 'https://raw.githubusercontent.com/ekhlebus/tme-breast-cancer-subtype-prediction/refs/heads/main/data/BRCA_clinicalMatrix.tsv'
    clinical = pd.read_csv(clin_data, sep="\t")

    ## Select ONLY necessary clinical columns
    selected_cols = [
        'sampleID',
        'Age_at_Initial_Pathologic_Diagnosis_nature2012',
        'Gender_nature2012',
        'ER_Status_nature2012',
        'PR_Status_nature2012',
        'HER2_Final_Status_nature2012',
        'PAM50Call_RNAseq' # intrinsic subtype (our target variable)
    ]

    clinical_sub = clinical.loc[:, clinical.columns.intersection(selected_cols)]

    ## Rename columns to cleaner names
    clinical_sub = clinical_sub.rename(columns={
        'Age_at_Initial_Pathologic_Diagnosis_nature2012': 'age',
        'Gender_nature2012': 'gender',
        'ER_Status_nature2012': 'ER',
        'PR_Status_nature2012': 'PR',
        'HER2_Final_Status_nature2012': 'HER2',
        'PAM50Call_RNAseq': 'subtype'
    })

    tme_data = 'https://raw.githubusercontent.com/ekhlebus/tme-breast-cancer-subtype-prediction/refs/heads/main/data/TCGA.Kallisto.fullIDs.cibersort.relative.tsv'
    cibersort = pd.read_csv(tme_data, sep="\t")
    cibersort["patient_id"] = cibersort.iloc[:, 0].str[:15]
    cibersort["patient_id"] = cibersort["patient_id"].str.replace(".", "-", regex=False)

    cibersort_brca = cibersort[cibersort.CancerType == "BRCA"]
    
    non_cell_cols = [
        'SampleID', 'CancerType', 'P.value',
        'Correlation', 'RMSE', 'patient_id'
    ]
    
    cell_cols = [c for c in cibersort.columns if c not in non_cell_cols]
    
    cibersort_agg = (
        cibersort_brca
        .groupby('patient_id', as_index=False)[cell_cols]
        .mean()
    )
    
    df = clinical_sub.merge(cibersort_agg,
                            left_on="sampleID",
                            right_on="patient_id",
                            how="inner"
                            )
    df = df.drop(columns=['patient_id'])

    # List of immune cell types (features) from CIBERSORT output
    immune_cols = [
        c for c in df.columns
        if c.startswith((
            "B.cells",
            "Plasma",
            "T.cells",
            "NK.cells",
            "Monocytes",
            "Macrophages",
            "Dendritic",
            "Mast.cells",
            "Eosinophils",
            "Neutrophils"
        ))
    ]

    # Filter to Basal vs Luminal only, Remove rare / noisy classes
    df_sub = df[df["subtype"].isin([
        "Basal",
        "LumA",
        "LumB"
    ])].copy()

    df_sub["y_target"] = df_sub["subtype"].map({
        "Basal": 1,
        "LumA": 0,
        "LumB": 0
    })

    # Replace missing age values with median values
    df_sub["age"] = df_sub["age"].fillna(df_sub["age"].median())

    df_full = df_sub[immune_cols + ["age"] + ["y_target"]]
    
    #scaler = StandardScaler()
    #X_train = scaler.fit_transform(df_full)

    return df_full


# Function to train the model
def train_model(df):

    immune_cols = [c for c in df.columns if c.startswith(immune_prefixes)]

    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000,
                           class_weight="balanced",
                           #solver='liblinear',
                           random_state=1)
    )

    X_train = df[immune_cols + ["age"]]
    y_train = df.y_target

    pipeline.fit(X_train, y_train)

    return pipeline


# Function to save the model
def save_model(filename, model):
    with open(filename, 'wb') as f_out:
        pickle.dump(model, f_out)
    
    print(f'model saved to {filename}')


df = load_data()
pipeline = train_model(df)
save_model('model.bin', pipeline)