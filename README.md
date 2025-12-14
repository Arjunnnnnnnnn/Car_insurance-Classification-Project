# Car Insurance Policy Binding Classification (ISBOUND)

Classification project predicting whether an insurance quote will become a **bound policy**
(`ISBOUND = 1`) using customer and policy features. Focus on **class imbalance**, **ROC-AUC**,
and understanding business trade-offs.



## 1. Project Overview

Insurance companies generate many quotes, but only a subset convert into active policies.
This project builds models to predict whether a quote will **bind** so that marketing and
sales teams can prioritise high-probability leads.



## 2. Dataset

- **Rows:** Insurance quotes.
- **Target:** `ISBOUND` (0 = not bound, 1 = bound).
- **Features (example categories):**
  - Customer demographics
  - Vehicle characteristics
  - Coverage / premium information
  - Channel, region, or agent IDs

Data issues handled:

- Missing values
- Class imbalance (relatively fewer bound policies than non-bound)
- Mix of categorical and numerical features

## 3. Methods

### 3.1. Preprocessing

- Train/validation/test split.
- One-hot encoding for categorical features.
- Standardisation of numeric features (for some models).
- Optionally, techniques for imbalance:
  - Class weights
  - SMOTE / oversampling (if used)
  - Undersampling majority class

### 3.2. Models

- **Logistic Regression** (baseline, interpretable coefficients).
- **Tree-based models**:
  - Random Forest
  - Gradient Boosted Trees (XGBoost / LightGBM or scikit-learn `HistGradientBoostingClassifier`)

### 3.3. Evaluation

- Main metrics:
  - **ROC-AUC**
  - **Precision/Recall** (especially for `ISBOUND=1`)
  - Confusion matrix at different decision thresholds
- Business-focused analysis:
  - Compare different thresholds for “positive” prediction.
  - Discuss trade-offs between capturing more bound policies vs false positives.



## 4. Repository Structure


car-insurance-policy-binding-classification/
├─ data/
│  ├─ insurance_train.csv
│  └─ insurance_test.csv
├─ notebooks/
│  ├─ 01_eda.ipynb
│  ├─ 02_baseline_models.ipynb
│  └─ 03_tuned_models_and_thresholds.ipynb
├─ src/
│  ├─ features.py
│  ├─ train_model.py
│  └─ evaluate_model.py
├─ reports/
│  └─ insurance_classification_report.pdf
└─ README.md
