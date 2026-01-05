## Immune Landscape–Based Prediction of Breast Cancer Subtypes

#### 📌 Project Overview

Breast cancer molecular subtypes are clinically important and guide treatment decisions. While subtype classification is typically based on tumor gene expression, the tumor immune microenvironment (TME) also differs substantially between subtypes.

This project evaluates whether immune cell composition alone, estimated using CIBERSORT, can distinguish Basal-like from Luminal (A/B) breast cancer subtypes in TCGA BRCA patients using a machine learning model.

#### 🎯 Objective

To build and evaluate a binary classification model that predicts *Basal-like* vs *Luminal (A + B)* breast cancer subtypes
using CIBERSORT-estimated immune cell fractions.

#### 📂 Data Sources
1. Immune Cell Fractions (CIBERSORT)

    File: TCGA.Kallisto.fullIDs.cibersort.relative.tsv

    Description:
    Relative fractions of 22 immune cell types estimated using CIBERSORT from TCGA RNA-seq data.

    Features:
    B cells, T cells (multiple subsets), NK cells, macrophages, dendritic cells, mast cells, eosinophils, neutrophils.

2. Clinical and Subtype Annotations

    File: BRCA_clinicalMatrix

    Source: UCSC Xena (TCGA BRCA)

    Key column:

    PAM50Call_RNAseq — intrinsic molecular subtype labels

All data available in [this repo](https://github.com/ekhlebus/tme-breast-cancer-subtype-prediction/tree/main/data).

#### 🧬 Target Definition

The original PAM50 subtypes were filtered and mapped as follows:

PAM50 subtype	Binary label
Basal-like	1
Luminal A	0
Luminal B	0
HER2-enriched	excluded
Normal-like	excluded

This formulation focuses on a clinically meaningful and biologically distinct contrast.

#### ⚙️ Data Processing
*Sample Matching*

TCGA subsample barcodes were converted to sample-level IDs (first 15 characters).

Immune and clinical tables were merged using sample IDs.

Feature Selection

Included:

22 immune cell fractions (CIBERSORT)

Age at diagnosis

Excluded:

ER / PR / HER2 status (risk of label leakage)

Gender (extreme imbalance and low relevance)

Survival and treatment variables

*Missing Values*

Age missing values were imputed using the median.

Samples missing subtype labels were excluded.

⚠️ Rationale for Excluding ER / PR / HER2

ER, PR, and HER2 status are closely tied to the biological definitions of PAM50 subtypes and are often used clinically to infer subtype membership. Including them would introduce label leakage, artificially inflating model performance.

The primary goal is to assess whether immune composition alone can distinguish subtypes; therefore, these markers were excluded from the main model.

#### ⚖️ Class Balance

Basal-like: ~18%

Luminal: ~82%

This imbalance reflects real-world prevalence and was intentionally preserved.
To address it:

* Stratified train–test splitting was used

* ROC-AUC and class-specific metrics were emphasized (in progress)

* Logistic regression used class_weight="balanced"

#### 🤖 Modeling Approach
Model

Algorithm: Logistic Regression

Reasoning:

Well-suited for tabular clinical data

Interpretable coefficients

Robust with limited feature counts

Train/Test Split

75% training / 25% testing

Stratified by subtype to preserve class proportions

Preprocessing

Standardization of all features (immune fractions + age)

#### 📊 Evaluation Metrics (in progress)

Primary metric:

ROC-AUC

Additional evaluation:

Confusion matrix

Precision and recall for Basal-like subtype

Classification report

ROC-AUC values in the range of 0.70–0.80 indicate meaningful immune-related signal.

#### 📈 Results Summary

The baseline logistic regression model demonstrates that immune cell composition carries predictive information for distinguishing Basal-like from Luminal breast cancer subtypes.

Key observations:

Basal-like tumors show distinct immune profiles

The model performs substantially better than random

Performance is achieved without using tumor-intrinsic molecular markers