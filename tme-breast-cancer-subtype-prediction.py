# %% [markdown]
# # Immune Landscape–Based Prediction of Breast Cancer Molecular Subtypes

# %% [markdown]
# ## 1. Load & Prepare Data
# 
# Here we are loading CIBERSORT data with immune cells fractions and clinical data with breast cancer molecular subtypes. To be able to make one modeling table we need to prepare data accordingly, merge both tables, and filter breast cancer patients for future use.

# %%
import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

# %% [markdown]
# ### 1.1 Load clinical data for breast cancer samples

# %%
# Define the URL for the clinical dataset (BC Subtypes)
clin_data = 'https://raw.githubusercontent.com/ekhlebus/tme-breast-cancer-subtype-prediction/refs/heads/main/data/BRCA_clinicalMatrix.tsv'
clinical = pd.read_csv(clin_data, sep="\t")
clinical.shape

# %%
clinical.head(3)

# %%
clinical.columns.tolist()

# %%
# Select ONLY necessary clinical columns
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

# %%
# Rename columns to cleaner names
clinical_sub = clinical_sub.rename(columns={
    'Age_at_Initial_Pathologic_Diagnosis_nature2012': 'age',
    'Gender_nature2012': 'gender',
    'ER_Status_nature2012': 'ER',
    'PR_Status_nature2012': 'PR',
    'HER2_Final_Status_nature2012': 'HER2',
    'PAM50Call_RNAseq': 'subtype'
})

# %%
clinical_sub.info()

# %%
clinical_sub.sampleID.nunique()

# %%
clinical_sub.gender.value_counts(dropna=False)


# %%
# Quick data quality check for age column
clinical_sub.age.describe()

# %%
clinical_sub.subtype.value_counts(dropna=False)

# %% [markdown]
# ### 1.2 Load CIBERSORT data

# %%
# Define the URL for the immune profiles dataset (CIBERSORT)
tme_data = 'https://raw.githubusercontent.com/ekhlebus/tme-breast-cancer-subtype-prediction/refs/heads/main/data/TCGA.Kallisto.fullIDs.cibersort.relative.tsv'

# %%
cibersort = pd.read_csv(tme_data, sep="\t")
cibersort.shape

# %%
cibersort.head(3)

# %% [markdown]
# Extract sample-level TCGA ID and make new column with patient_id
# 
# CIBERSORT uses sub-sample-level barcodes, but clinical data is sample-level (first 15 symbols in SampleID).

# %%
cibersort["patient_id"] = cibersort.iloc[:, 0].str[:15]
cibersort[["SampleID", "patient_id"]].head(3)

# %%
# replace "." with "-" in the patient_id column, since clinical data (below) uses "-" as separator
cibersort["patient_id"] = cibersort["patient_id"].str.replace(".", "-", regex=False)
cibersort[["SampleID", "patient_id"]].head(2)

# %%
# Select only breast cancer samples
cibersort_brca = cibersort[cibersort.CancerType == "BRCA"]
cibersort_brca.head(3)

# %%
cibersort_brca.SampleID.nunique(), cibersort_brca.patient_id.nunique()

# %%
# Check duplicates in CIBERSORT
cibersort_brca['patient_id'].duplicated().sum()

# %%
# Check duplicated patients in cibersort data
cibersort_brca['patient_id'].value_counts().head(10)


# %% [markdown]
# Here we can see that cibersort contains multiple samples per patient! When merged we will see that one clinical row matches multiple CIBERSORT rows, what leads to duplicated clinical info. This is dangerous for ML, so we need to transform our data to have only one imput for each patient_id. In this case we will aggregate CIBERSORT immune fractions per patient.

# %%
non_cell_cols = [
    'SampleID', 'CancerType', 'P.value',
    'Correlation', 'RMSE', 'patient_id'
]

cell_cols = [c for c in cibersort.columns if c not in non_cell_cols]
print(len(cell_cols))
cell_cols


# %% [markdown]
# Aggregate CIBERSORT per patient (MEAN)

# %%
cibersort_agg = (
    cibersort_brca
    .groupby('patient_id', as_index=False)[cell_cols]
    .mean()
)
cibersort_agg.shape

# %%
# Check duplicates in CIBERSORT after aggregation
cibersort_agg.patient_id.duplicated().sum()

# %% [markdown]
# ### 1.3 Merge immune + clinical tables

# %%
clinical_sub.head(2)

# %%
cibersort_agg.head(2)

# %%
clinical_sub.sampleID.nunique(), cibersort_agg.patient_id.nunique()

# %%
df = clinical_sub.merge(cibersort_agg,
    left_on="sampleID",
    right_on="patient_id",
    how="inner"
)
df = df.drop(columns=['patient_id'])
print(df.shape)

df['sampleID'].duplicated().sum()


# %%
df.head(2)

# %% [markdown]
# ## 2. Data preparation for modeling (cleaning, filtering) + initial EDA

# %%
df.head(3)

# %% [markdown]
# Sometimes it is difficult to see all columns in a wide dataframe. To take a look at all columns we can transpose dataFrame, so rows become columns and now we can see them better.
# 

# %%
df.head().T

# %%
df.dtypes

# %% [markdown]
# Data for cell types and age are numerical as expected. No obvious data types errors.
# 
# Now let's consider features in more detailes. 

# %% [markdown]
# ### 2.1 Immune cells

# %%
# Define immune cell features
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

immune_cols, len(immune_cols)

# %% [markdown]
# ### 2.2 Subtype variable 
# Now let's work with subtype variable (our target variable) and select samples only with subtype we are interested in.

# %%
# Check subtype values
df.subtype.value_counts(dropna=False)

# %%
# Filter to Basal vs Luminal only, Remove rare / noisy classes
df_sub = df[df["subtype"].isin([
    "Basal",
    "LumA",
    "LumB"
])].copy()

# Look at the target variable distribution
df_sub.subtype.value_counts()

# %%
# Create binary target variable
df_sub["y_target"] = df_sub["subtype"].map({
    "Basal": 1,
    "LumA": 0,
    "LumB": 0
})

# %%
df_sub["y_target"].value_counts(normalize=True)


# %% [markdown]
# We see **class imbalance** here.
# 
# Class imbalance consideration:
# The dataset exhibits moderate class imbalance (≈18% Basal-like vs 82% Luminal). This reflects real-world subtype prevalence and intentionally preserved to maintain clinical realism. Model performance is therefore will be evaluated using ROC-AUC and class-specific precision/recall rather than accuracy alone. Stratified train–test splitting will be used to ensure consistent class proportions.

# %% [markdown]
# ### 2.3 Age
# Let's explore what we have for age.

# %%
# Check how many missing age values we have
df_sub.age.isnull().sum()

# %%
# Calculate the proportion of missing age values
df_sub.age.isna().mean()


# %% [markdown]
# Since we have less than 10% missing values, we will replace them with median values.

# %%
# Let's replace missing age values with median values
df_sub["age"] = df_sub["age"].fillna(df_sub["age"].median())
df_sub.age.isna().mean()

# %% [markdown]
# ### 2.4 Gender

# %%
# Check gender values
df_sub.gender.value_counts(dropna=False)

# %%
# Check percentages of each gender
df_sub.gender.value_counts(normalize=True, dropna=False)

# %% [markdown]
# **Drop gender from the model**
# 
# Note: Gender was excluded from modeling due to extreme class imbalance (>93% female), limited biological relevance for breast cancer subtype differentiation, and non-trivial missingness. Including such a feature risks introducing noise without improving predictive performance.

# %% [markdown]
# ### 2.5 What about ER / PR / HER2?
# 
# For now ER / PR / HER2 status will be not included as model features.
# 
# Note: Estrogen receptor (ER), progesterone receptor (PR), and HER2 status intentionally excluded from the primary model. These markers are closely related to the biological definitions of PAM50 breast cancer subtypes and are often used clinically to characterize or infer subtype membership. Including them as predictors would therefore introduce a risk of label leakage, artificially inflating model performance and obscuring the contribution of the immune microenvironment.
# 
# The goal of this analysis is to evaluate whether immune cell composition alone, with minimal clinical covariates, can distinguish Basal-like from Luminal breast cancer subtypes. ER/PR/HER2 may be explored separately in sensitivity analyses but are excluded from the main modeling pipeline.

# %% [markdown]
# ## 3. Train / Test Split + EDA for training data

# %% [markdown]
# ### 3.1 Splitting the data

# %%
from sklearn.model_selection import train_test_split

# %%
df_sub.T

# %%
# Define final df
df_full = df_sub[immune_cols + ["age"] + ["y_target"]]

# %%
df_full.shape

# %% [markdown]
# Since we have class imbalance, let's make next split:
# * TRAIN - 75%
# * TEST - 25%
# 
# Since we have class imbalance, let's use stratify parameter.
# Use stratified train/test split. This ensures both sets reflect the same imbalance.

# %%
# Split the dataset into training and testing sets
df_train_full, df_test = train_test_split(
    df_full,
    test_size=0.25,
    random_state=1,
    stratify=df_full["y_target"]
)

# test_size=0.25 - setting aside 25% of the data for testing
# random_state=1 - setting a seed for reproducibility

# %%
len(df_train_full), len(df_test)

# %%
df_train_full.head(3)

# %%
# Indexes have been reset for both training and testing sets
df_train_full = df_train_full.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# %%
df_train_full.head(3)

# %%
y_train = df_train_full.y_target.values
y_test = df_test.y_target.values

df_train = df_train_full.drop(columns=["y_target"])
df_test = df_test.drop(columns=["y_target"])

# %%
n_basal = (y_test == 1).sum()
n_total = len(y_test)

print(f"Basal-like cases in test set: {n_basal} / {n_total} ({n_basal / n_total:.1%})")


# %% [markdown]
# The stratified split resulted in sufficient representation of the Basal-like subtype in the test set.

# %%
# Verify values:
print(pd.Series(y_train).value_counts(normalize=False))
print()
print(pd.Series(y_test).value_counts(normalize=False))

# %%
# Verify balance:
print(pd.Series(y_train).value_counts(normalize=True))
print()
print(pd.Series(y_test).value_counts(normalize=True))

# %% [markdown]
# Train–test split explanation:
# The dataset was split into 75% training and 25% testing sets using stratified sampling. This proportion provides sufficient training data for stable model estimation while ensuring an adequate number of Basal-like cases (minority class) in the test set.

# %% [markdown]
# ### 3.2 Feature scaling
# 
# Immune fractions are on similar scales (from 0 to 1), but age is not, so we scale everything.

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(df_train)
X_test = scaler.transform(df_test)


# %%
len(X_train), len(X_test)

# %% [markdown]
# ### 3.3 Checking missing values (EDA)
# 
# Here we will use our full train dataset

# %%
df_train.head(3)

# %%
# Let's look for missing values in the full training set
df_train.isnull().sum()

# %% [markdown]
# No missing values! So don't need to do any additional data preparation steps.

# %%
df_train.dtypes

# %% [markdown]
# We are doing this to double check which columns are categorical and which are numerical. All columns are numerical, and before that we don't need one-hot encoding and DictVectorizer.
# 

# %%
# Let's look at the target variable distribution (proportion of classes)
pd.Series(y_train).value_counts(normalize=True)

# %% [markdown]
# ### 3.4 Feature Importance: Correlation
# 
# This is a way to measure Feature Importance to **numerical** vatiables. And here we are talking about Pearson's correlation - this is way to measure dependency between two variables.
# 
# Pearson correlation coefficient (r) - https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
# 
# -1 <= r <= 1
# 
# In our case:
# * y = 0 or y = 1 - binary subtype
# 
# So, since y is binary, we will use point-biserial correlation, which is numerically equivalent to Pearson.

# %%
df_train_full['age'].corr(df_train_full.y_target)

# %%
corr_df = df_train_full.drop(columns=['y_target', 'age']).corrwith(df_train_full.y_target).to_frame('correlation')
corr_df.head(5)

# %%
corr_df_sorted = corr_df.sort_values('correlation')
corr_df_sorted.head(3)

# %%
plt.figure(figsize=(8, 5))
plt.barh(corr_df_sorted.index, corr_df_sorted['correlation'])
plt.xlabel("Correlation with subtype (y_target)")
plt.ylabel("Immune cell type")
plt.title("Correlation between immune cell fractions and subtype")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Training Logistic Regression with Scikit-Learn
# 
# * Train a model with Scikit-Learn
# * Apply it to the test dataset

# %%
from sklearn.linear_model import LogisticRegression

# %%
# Create the model
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    #solver='liblinear', 
    random_state=1)

# Train the model with dataset, created earlier
model.fit(X_train, y_train)

# %%
# The value of the model output when all features are zero
model.intercept_[0]

# %%
# Let's look inside the model, this is how we can access the weights
model.coef_[0].round(3)

# %%
# let's use the model and get HARD predictions (1 - Basal, 0 - Luminal)
model.predict(X_train)

# %%
# Also we can get SOFT predictions with probabilities
model.predict_proba(X_train)
# First column in this 2D array is the probability of class 0 (Luminal)
# Second column is the probability of class 1 (Basal)

# %%
y_pred_proba = model.predict_proba(X_train)[:, 1]
#y_pred_proba

# %%
# Using those predicted probabilities, we can set up a custom threshold
y_pred = (model.predict_proba(X_train)[:, 1] >= 0.3).astype(int)
#y_pred

# %% [markdown]
# Now we have our predictions. Let's check how accurate they are.

# %% [markdown]
# ## 5. Measure performance of the model
# 
# To measure performance of our model we will use next metrics:
# 
# * Accuracy - it tells us how many correct predictions we made (not the best approach because of class imbalance, just for sanity check)
# * ROC-AUC (primary metric)
# * Confusion matrix
# * ROC curve

# %% [markdown]
# ### 5.1 Accuracy
# 
# Let's see how many actual subtype values and predicted values match and compare train vs test accuracy.

# %%
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_accuracy = (y_train == y_train_pred).mean()
test_accuracy = (y_test == y_test_pred).mean()  

print(f"Train Accuracy: {train_accuracy:.3f}")
print(f"Test Accuracy:  {test_accuracy:.3f}")

# %% [markdown]
# Note: Accuracy is reported for completeness but is not the primary evaluation metric due to class imbalance (~18% Basal-like). A model predicting only the majority class would achieve high accuracy without meaningful discrimination. Therefore, ROC-AUC and class-specific recall are emphasized in subsequent evaluation.
# 
# Here we used 0.5 threshold, but is it good or not?

# %%
from sklearn.metrics import accuracy_score

# %%
thresholds = np.linspace(0, 1, 11)
thresholds

# %%
# function to check different thresholds
thresholds = np.linspace(0, 1, 21)

accuracies = []

for t in thresholds:
    acc = accuracy_score(y_test, y_test_pred >= t) # from sklearn.metrics
    accuracies.append(acc)
    print('%0.2f %0.3f' % (t, acc))

# %% [markdown]
# Using accuracy is not the best way to evaluate our model as expected! 
# 
# So accuracy is a score that for cases with class imbalance can be misleading. In such cases it is useful to use other metrics and different way of eveluation a quality of our model, that is not affected by class imbalance.

# %%
plt.figure(figsize=(4, 2))

plt.plot(thresholds, accuracies, color='black')

plt.title('Threshold vs Accuracy')
plt.xlabel('Threshold')
plt.ylabel('Accuracy')

plt.xticks(np.linspace(0, 1, 11))

# plt.savefig('04_threshold_accuracy.svg')

plt.show()

# %% [markdown]
# ### 5.2 ROC-AUC
# 
# Due to class imbalance between Basal-like and Luminal tumors, ROC-AUC is used as the primary evaluation metric, as it measures discrimination across all classification thresholds and is not biased by class prevalence.

# %%
from sklearn.metrics import roc_auc_score

roc_auc = roc_auc_score(y_test, y_test_pred)
print(f"Test ROC-AUC: {roc_auc:.3f}")

# For diagnostic purposes, we can also look at train ROC-AUC
roc_auc_train = roc_auc_score(y_train, y_train_pred)
print(f"Train ROC-AUC: {roc_auc_train:.3f}")

# %%
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, y_test_pred)

plt.figure(figsize=(3, 3))
plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Basal-like vs Luminal")
plt.legend()
plt.show()


# %%
np.unique(y_test_pred).shape


# %% [markdown]
# ## 6. Model interpretation
# 
# * Look at the model coefficients
# * Convert coefficients to odds ratios

# %%
# Extract model coefficients
model.coef_[0].round(3)

# %%
coef_df = pd.DataFrame({
    "feature": df_full.columns.drop("y_target"),
    "coefficient": model.coef_[0]
})

coef_df = coef_df.sort_values("coefficient", ascending=False)
coef_df

# %% [markdown]
# Interpretation:
# 
# * Positive coefficient → higher probability of Basal-like
# * Negative coefficient → higher probability of Luminal

# %%
# Convert coefficients to odds ratios
coef_df["odds_ratio"] = np.exp(coef_df["coefficient"])
coef_df.head(3)


# %%
top_features = coef_df.head(10)

plt.figure(figsize=(4, 3))
plt.barh(top_features["feature"], top_features["coefficient"])
plt.gca().invert_yaxis()
plt.xlabel("Logistic Regression Coefficient")
plt.title("Top Immune Features Associated with Basal-like Subtype")
plt.show()

# %%
bottom_features = coef_df.tail(8)

plt.figure(figsize=(4, 3))
plt.barh(bottom_features["feature"], bottom_features["coefficient"])
plt.gca().invert_yaxis()
plt.xlabel("Logistic Regression Coefficient")
plt.title("Top Immune Features Associated with Luminal-like Subtype")
plt.show()

# %% [markdown]
# The model identifies higher abundance of macrophages (M0/M1), dendritic cells, and CD4 T-cell populations as positively associated with the Basal-like subtype, while mast cells are more strongly associated with Luminal tumors. These findings are consistent with published analyses of TCGA breast cancer, which report increased immune infiltration and antigen-presenting cell activity in Basal-like tumors, and enrichment of mast cells in hormone receptor–positive Luminal subtypes.
# 
# Importantly, these associations emerged without using tumor-intrinsic molecular markers, suggesting that immune context alone captures subtype-specific biology.

# %% [markdown]
# 


