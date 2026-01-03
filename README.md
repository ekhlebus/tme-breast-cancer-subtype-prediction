# tme-breast-cancer-subtype-prediction
Immune Landscape–Based Prediction of Breast Cancer Molecular Subtypes. Subtype Classification Based on CIBERSORT Immune Fractions. 

Here we evaluate whether immune cell composition alone is sufficient to distinguish clinically relevant cancer subtypes.

🎯 Binary classification definition (important decision)

To keep this industry-clean and statistically stable, we’ll do:

Basal-like vs Luminal (A + B)

Why this is the best binary setup:

Basal-like ≈ triple-negative (clinically important)

Luminal A/B ≈ hormone receptor–positive

Strong immune differences → CIBERSORT makes sense

Balanced enough for ML

Very common in real publications and industry work


## Immune Cell Fraction Aggregation

Immune cell proportions were estimated from bulk RNA-seq data using CIBERSORT.
Because TCGA patients may have multiple tumor samples or aliquots, immune cell estimates were initially available at the sample level.

To avoid patient duplication and data leakage in downstream machine learning analyses, immune cell fractions were aggregated at the patient level prior to merging with clinical data.

For each patient, immune cell proportions were averaged (mean aggregation) across all available samples. Mean aggregation was selected because:

Most patients had one primary tumor sample

CIBERSORT fractions are normalized and compositional

Mean aggregation is standard practice in TCGA-based immunogenomic studies

Only immune cell fraction columns were retained for aggregation; metadata and CIBERSORT quality control metrics (P-value, correlation, RMSE) were excluded.

After aggregation, each patient was represented by a single immune profile, which was then merged with clinical variables using TCGA patient barcodes.