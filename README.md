# 🏦 Predicting Retail Banking Loan Defaults Using Machine Learning, SHAP Interpretability, and Fairness-Aware Modeling

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-red)
![LightGBM](https://img.shields.io/badge/LightGBM-Boosting-lightgreen)
![SHAP](https://img.shields.io/badge/SHAP-Interpretability-blueviolet)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

##  Overview

Loan default prediction is one of the most consequential tasks in retail banking , a missed default costs the bank, an unfair rejection costs the customer. This project tackles both problems.

This project builds a full machine learning pipeline that not only **predicts loan defaults with high accuracy**, but also **explains why** each prediction was made using SHAP values, and **audits whether the model treats different demographic groups fairly**. The goal is a model that is accurate, transparent, and responsible.

Four models are trained and rigorously compared  from a simple logistic regression baseline to gradient boosting ensembles using a credit risk dataset with real-world borrower financial profiles. The best model (LightGBM) achieves over **93% ROC-AUC**. All predictions are then dissected using SHAP analysis at both global and individual levels, and fairness metrics are computed across age and income groups.

Six research hypotheses are formally tested using statistical significance tests throughout the pipeline.

---

## Project Hypotheses

| ID | Hypothesis | Test Method |
|----|-----------|-------------|
| **H1** | Top SHAP features show statistically significant association with loan default | Point-biserial correlation, Spearman correlation |
| **H2** | Advanced ML models (XGBoost, LightGBM) significantly outperform Logistic Regression on ROC-AUC | Pairwise ROC-AUC comparison with % improvement |
| **H3** | Ensemble methods outperform single-model approaches | Cross-model performance comparison |
| **H4** | SHAP importance rankings correlate strongly with native feature importance | Spearman rank correlation |
| **H5** | The best-performing model shows no statistically significant demographic bias | Demographic Parity Difference, Equalized Odds |
| **H6** | Feature engineering significantly improves predictive performance | Paired t-test on models trained with vs. without engineered features |

---

##  Repository Structure

```
📦 loan-default-prediction/
│
├── 📓 PROJECT_WORK.ipynb                    # Full pipeline notebook
├── 📄 README.md                             # This file
│
├── 📊 Data (generated at runtime)
│   ├── processed_credit_risk_FINAL.csv      # Cleaned + engineered dataset
│   ├── X_features.csv                       # Feature matrix only
│   ├── y_target.csv                         # Target vector only
│   └── fairness_groups.pkl                  # Serialized protected attributes
│
├── 📈 Results (generated at runtime)
│   ├── feature_engineering_impact.csv       # With vs. without engineered features
│   ├── feature_engineering_stats.csv        # Statistical significance results
│   ├── hypothesis_h1_statistical_tests.csv
│   ├── hypothesis_h2_results.csv
│   └── hypothesis_final_summary.csv
│
└── 🖼️ Visualisations (generated at runtime)
    ├── screenshot_01_target.png             # Target class distribution
    ├── roc_curves_comparison.png            # ROC curves for all 4 models
    ├── confusion_matrices.png               # 2×2 confusion matrix grid
    ├── shap_xgb_global_importance.png       # XGBoost SHAP summary
    ├── shap_lgbm_global_importance.png      # LightGBM SHAP summary
    └── shap_*_waterfall_*.png               # Individual prediction plots
```

---

##  Dataset

The project uses a **credit risk dataset** (`credit_risk_dataset.csv`) with borrower financial and demographic profiles. Key features include:

| Feature | Type | Description |
|---------|------|-------------|
| `person_age` | Numeric | Borrower age (used for fairness grouping) |
| `person_income` | Numeric | Annual income (used for fairness grouping) |
| `person_home_ownership` | Categorical | RENT / OWN / MORTGAGE / OTHER |
| `person_emp_length` | Numeric | Employment length in years |
| `loan_amnt` | Numeric | Requested loan amount |
| `loan_int_rate` | Numeric | Loan interest rate |
| `loan_intent` | Categorical | Purpose of loan (EDUCATION, MEDICAL, etc.) |
| `loan_grade` | Categorical | Credit grade (A–G) |
| `loan_percent_income` | Numeric | Loan amount as % of income |
| `cb_person_default_on_file` | Categorical | Historical default flag (Y/N) |
| `cb_person_cred_hist_length` | Numeric | Credit history length in years |
| `loan_status` | Binary | **Target**  0: No Default, 1: Default |


> https://www.kaggle.com/datasets/laotse/credit-risk-dataset
> 
> The link to the raw dataset is included in this repository. 
---

##  Pipeline — Detailed Breakdown

### Stage 1 — Data Preprocessing

**Missing Value Imputation:**
- Numeric columns → `SimpleImputer(strategy='median')`,  chosen for robustness against skewed income/loan distributions
- Categorical columns → `SimpleImputer(strategy='most_frequent')`

**Outlier Detection & Treatment:**
- Applied to: `person_age`, `person_income`, `loan_amnt`, `loan_int_rate`
- Method: **Applied IQR-based outlier detection using Tukey’s fences (3×IQR) instead of the standard 1.5×IQR.** A wider threshold was chosen because variables such as income, loan amount, and interest rate are inherently skewed and may contain legitimate extreme values.
- Treatment: Outliers were handled using **Winsorization**, capping values at the upper and lower bounds to preserve dataset size while reducing the impact of extreme observation

```python
Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1
df[col] = df[col].clip(lower=Q1 - 3*IQR, upper=Q3 + 3*IQR)
```

**Target Distribution:**
```python
y.value_counts()
# 0 (No Default): majority class
# 1 (Default):    minority class
# Class imbalance handled via class_weight='balanced' in applicable models
```
![Target Distribution](https://raw.githubusercontent.com/korede-folarin/IMAGES-FOR-FAIRNESS-PROJECT-/main/images/targetclasses.png)



---

### Stage 2: Feature Engineering

Four engineered features are created to capture financial risk signals:

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `debt_to_income_ratio` | `loan_amnt / person_income` | Core creditworthiness metric used by lenders |
| `income_per_emp_year` | `person_income / (person_emp_length + 1)` | Income stability relative to employment tenure |
| `has_default_history` | Binary encoding of `cb_person_default_on_file == 'Y'` | Strong binary signal for historical risk |
| `age_group` | Binned `person_age` into demographic bands | Used exclusively for fairness auditing |
| `income_quartile` | Quartile rank of `person_income` | Used exclusively for fairness auditing |

> **Important:** `age_group` and `income_quartile` are saved separately as protected attributes **before** being dropped from the modeling dataset, ensuring they never influence predictions while remaining available for post-hoc fairness analysis.

**One-Hot Encoding:**
Applied to nominal categorical columns: `person_home_ownership`, `loan_intent`, `loan_grade`
```python
df = pd.get_dummies(df, columns=['person_home_ownership', 'loan_intent', 'loan_grade'], drop_first=True)
```

---

### Stage 3:  Model Training

**Data Split:**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X_modeling, y,
    test_size=0.3,
    random_state=42,
    stratify=y          # Preserve class balance in both splits
)
```

**Scaling:**
Applied StandardScaler to normalize feature values for **Logistic Regression**, ensuring all variables contribute equally to model training. Scaling was fitted on training data and applied to test data to prevent data leakage.
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

**Model Configurations:**

I chose four models to enable systematic comparison:

**Logistic Regression**: Baseline linear model for interpretability benchmarking
**Random Forest**: Bagging ensemble to assess feature interaction learning
**XGBoost & LightGBM**: Gradient boosting variants to evaluate state-of-the-art performance and compare architectural differences (level-wise vs. leaf-wise growth)

```python
# Logistic Regression (Baseline)
LogisticRegression(max_iter=1000, random_state=42,
                   class_weight='balanced', solver='lbfgs')

# Random Forest
RandomForestClassifier(n_estimators=100, max_depth=10,
                       min_samples_split=20, min_samples_leaf=10,
                       class_weight='balanced', random_state=42, n_jobs=-1)

# XGBoost 
 XGBClassifier(n_estimators=100, max_depth=6,
                    learning_rate=0.1, scale_pos_weight=scale_pos_weight,
                    random_state=42,  eval_metric='logloss', use_label_encoder=False)


# LightGBM
LGBMClassifier(random_state=42)
```

---

### Stage 4: Model Evaluation

**Metrics computed for all models:**
- ROC-AUC (primary metric)
- Accuracy, Precision, Recall, F1-Score
- Confusion matrix (TP, TN, FP, FN)

**Cross-Validation:**
```python
cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
```
Applied on training set only to assess generalisation without touching the held-out test set.

**Visualisations generated:**
- Overlaid ROC curves for all 4 models → `roc_curves_comparison.png`
- 2×2 confusion matrix heatmap grid → `confusion_matrices.png`
- Feature importance plots (2×2 grid) using method appropriate to each model

---

### Stage 5 — SHAP Analysis

SHAP (SHapley Additive exPlanations) provides model-agnostic explanations grounded in game theory. Each feature's contribution to a prediction is calculated as its Shapley value.

**Explainer selection:**

| Model | Explainer | Reason |
|-------|-----------|--------|
| Logistic Regression | `shap.LinearExplainer` | Optimised for linear models; falls back to KernelExplainer if needed |
| Random Forest | `shap.TreeExplainer` | Exact Shapley values for tree ensembles |
| XGBoost | `shap.TreeExplainer` | Native tree structure exploitation |
| LightGBM | `shap.TreeExplainer` | Handles multi-output format (list → positive class extracted) |

**Background sample:** 100 samples drawn from `X_train` for KernelExplainer efficiency.

**Analyses performed:**
- **Global importance** — Mean |SHAP| bar plots for all models
- **Beeswarm summary plots** — Feature impact distribution across test set
- **Dependency plots** — Top 4 features by mean |SHAP| for each model, showing interaction effects
- **Waterfall plots** — Individual explanations for a representative DEFAULT case and NO-DEFAULT case
- **SHAP vs. Native importance** — Spearman rank correlation between SHAP rankings and model-native rankings (Gini impurity for RF, coefficients for LR, `feature_importances_` for XGB/LGBM)

**Top features consistently identified across models (H1):**
```
1. debt_to_income_ratio
2. loan_int_rate
3. person_income
4. loan_percent_income
5. has_default_history
```
Statistical association between these features and `loan_status` validated using point-biserial correlation (continuous features) and Spearman correlation.

---

### Stage 6 — Fairness Auditing

Post-hoc fairness analysis applied to the best-performing model (XGBoost).

**Protected attributes:**
- `age_group` — borrower age demographic bands
- `income_quartile` — Q1 (lowest income) through Q4 (highest income)

**Fairness metrics:**

```python
def demographic_parity_difference(y_pred, sensitive_attr):
    # Measures difference in positive prediction rates across groups
    # Ideal value: 0 (equal prediction rates)

def equalized_odds(y_pred, y_true, sensitive_attr):
    # Measures difference in TPR and FPR across groups
    # Ideal value: 0 (equal error rates)
```

**H5 Test:** The best-performing model is assessed for statistically significant bias. Results are compiled into a comprehensive fairness summary table covering approval rates, error rates, and disparity scores across all demographic groups.

---

### Stage 7 — Feature Engineering Impact (H6)

Two parallel pipelines are run:
- **Without engineered features** — trained on original dataset columns only
- **With engineered features** — includes `debt_to_income_ratio`, `income_per_emp_year`, `has_default_history`

All four model types are trained and evaluated in both configurations. A **paired t-test** is applied to determine whether the performance improvement from feature engineering is statistically significant.

---

## 📊 Results Summary

| Model | ROC-AUC | CV Mean ± Std | Notes |
|-------|---------|---------------|-------|
| Logistic Regression | Baseline | — | `class_weight='balanced'`, `solver='lbfgs'` |
| Random Forest | High | ± low variance | `n_estimators=100`, `max_depth=10` |
| XGBoost | Very High | ± low variance | GridSearchCV tuned |
| **LightGBM** | **>93%** | **± low variance** | **Best overall performance** |

---

## 🛠️ Installation

```bash
pip install pandas numpy scipy scikit-learn xgboost lightgbm shap matplotlib seaborn jupyter
```

Full dependency list:

```
pandas          # Data manipulation
numpy           # Numerical computing
scipy           # Statistical tests (chi2, pointbiserialr, spearmanr)
scikit-learn    # ML models, preprocessing, metrics, GridSearchCV
xgboost         # XGBClassifier
lightgbm        # LGBMClassifier
shap            # SHAP explainers (Linear, Tree, Kernel)
matplotlib      # Plotting
seaborn         # Heatmaps (confusion matrices)
pickle          # Saving fairness groups
```

---

## 🚀 How to Run

### Option 1: Google Colab

1. Upload `PROJECT_WORK.ipynb` to [Google Colab](https://colab.research.google.com)
2. Upload your dataset to Google Drive
3. Mount your Drive and update the `BASE` path in **Cell 4**:
   ```python
   BASE = "/content/drive/MyDrive/your_folder"
   df = pd.read_csv(f"{BASE}/credit_risk_dataset.csv", low_memory=False)
   ```
4. `Runtime → Run all`

### Option 2: Local (VS Code / Jupyter)

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/loan-default-prediction.git
cd loan-default-prediction

# 2. Install dependencies
pip install pandas numpy scipy scikit-learn xgboost lightgbm shap matplotlib seaborn jupyter

# 3. Place your dataset in the project folder
# Rename it to: credit_risk_dataset.csv

# 4. Update Cell 4 — replace the Colab mount with:
df = pd.read_csv("credit_risk_dataset.csv", low_memory=False)

# 5. Launch
jupyter notebook PROJECT_WORK.ipynb
```

> Run all cells **sequentially** — later stages depend on objects saved by earlier ones (`fairness_groups.pkl`, `processed_credit_risk_FINAL.csv`).

---

## 📁 Output Files Generated

| File | Stage | Description |
|------|-------|-------------|
| `processed_credit_risk_FINAL.csv` | Preprocessing | Cleaned, encoded, engineered dataset |
| `X_features.csv` | Preprocessing | Feature matrix (no target) |
| `y_target.csv` | Preprocessing | Target vector only |
| `fairness_groups.pkl` | Feature Engineering | Serialised protected attribute Series (age_group, income_quartile) |
| `feature_engineering_impact.csv` | H6 Testing | ROC-AUC comparison: with vs. without engineered features |
| `feature_engineering_stats.csv` | H6 Testing | Paired t-test p-values per model |
| `hypothesis_h1_statistical_tests.csv` | H1 Testing | Correlation stats for top SHAP features |
| `hypothesis_final_summary.csv` | All hypotheses | Consolidated H1–H6 results with verdicts |
| `roc_curves_comparison.png` | Evaluation | Overlaid ROC curves (300 DPI) |
| `confusion_matrices.png` | Evaluation | 2×2 subplot confusion matrix grid |
| `shap_*_global_importance.png` | SHAP | Bar plots of mean \|SHAP\| per model |
| `shap_*_waterfall_*.png` | SHAP | Individual prediction waterfall plots |

---

## 📚 References

- Breiman, L. (2001) 'Random Forests', *Machine Learning*, 45(1), pp. 5–32.
- Chen, T. and Guestrin, C. (2016) 'XGBoost: A Scalable Tree Boosting System', *Proceedings of KDD '16*, pp. 785–794.
- Ke, G. et al. (2017) 'LightGBM: A Highly Efficient Gradient Boosting Decision Tree', *Advances in Neural Information Processing Systems*, pp. 3146–3154.
- Lundberg, S. M. and Lee, S. I. (2017) 'A Unified Approach to Interpreting Model Predictions', *Advances in Neural Information Processing Systems*, pp. 4765–4774.
- Pedregosa, F. et al. (2011) 'Scikit-learn: Machine Learning in Python', *Journal of Machine Learning Research*, 12, pp. 2825–2830.

---

## 👤 Author

**Korede**
