# ✈️ Passenger Satisfaction — Data Science Case Study

A binary classification project built for **Borcelle Airlines**, aimed at predicting whether a passenger will be satisfied or not based on demographic information, flight type and in-flight service ratings.

---

## 🎯 Objective

Build a machine learning model capable of predicting passenger satisfaction and identify which service dimensions drive that outcome — providing actionable insights to improve both in-flight and ground services.

---

## 📂 Dataset

- **Source:** `train.csv`
- **Size:** 103,904 passengers · 25 features
- **Target:** `satisfaction` → binary (0 = neutral/dissatisfied, 1 = satisfied)
- **Features:** mix of demographic info, flight metadata and 1–5 service ratings (wifi, entertainment, seat comfort, etc.)

---

## 🔄 Project Workflow

### 1. Data Loading & Cleaning
- Removed `Unnamed: 0` (duplicate index) and `id` (no predictive value)
- Dropped `Arrival Delay in Minutes` — correlation of ~0.96 with `Departure Delay`
- Filled 310 missing values in delay column with median imputation

### 2. Exploratory Data Analysis
- Business class passengers are 3–4× more satisfied than Economy
- Business travelers show ~58% satisfaction vs ~10% for leisure travelers
- Online boarding, Inflight entertainment and Seat comfort show the largest rating gap between satisfied and dissatisfied passengers (+1.37, +1.07, +0.93)

### 3. Preprocessing
- Train / Validation / Test split: **70% / 15% / 15%** (cascaded, fully disjoint)
- Numerical features: median imputation
- Categorical features: OrdinalEncoder
- All wrapped in a **sklearn Pipeline** to prevent data leakage

### 4. Feature Selection
Three methods compared on the training set:

| Method | Top Features |
|--------|-------------|
| Chi-Square | Type of Travel, Class, Online boarding |
| Mutual Information | Online boarding, Inflight wifi, Class |
| T-Test (Welch) | Online boarding, Inflight entertainment, Seat comfort, Inflight wifi |

### 5. Modeling — Two Roads

**Road 1:** all available features (22 columns)

**Road 2:** top 4 features from T-test only
- Age, Flight Distance, Inflight wifi service, Ease of Online booking

Both roads follow the same pipeline:
1. Spot check with Stratified 5-Fold CV (default hyperparameters)
2. Select top 2 models
3. Hyperparameter tuning via GridSearchCV
4. Final evaluation on the held-out test set

### 6. Models
- Random Forest
- AdaBoost
- Gradient Boosting

---

## 📊 Results

| Road | Model | Test Accuracy |
|------|-------|--------------|
| Road 1 — all features | Random Forest (tuned) | ~97% |
| Road 2 — 4 features | Gradient Boosting (tuned) | ~82% |

**Winner: Random Forest on Road 1 — ~96% accuracy on the held-out test set.**

---

## 💡 Key Insights

- **Online boarding** is the single most discriminating feature (t_stat = -156)
- **Class** and **Type of Travel** dominate Chi-square and Mutual Information
- Using only 4 features reaches 84% — adding the full feature set gains 12 points
- **Gate location** shows virtually no difference between groups (p_value ~0.8) — not useful

---

## 🛠️ Action Points

- Fix the online boarding experience — largest satisfaction gap in the dataset
- Invest in inflight entertainment and seat comfort
- Upgrade inflight wifi — top predictor across all three selection methods
- Focus on Economy cabin and leisure travelers — lowest satisfaction rates, highest room for improvement

---

## 🔧 Tech Stack

- Python 3
- pandas · numpy
- scikit-learn
- matplotlib · seaborn
- scipy

---

## 📁 Structure

```
├── esame_finale.ipynb   # Main notebook
├── train.csv            # Dataset
└── README.md
```

---

## 🚀 Possible Improvements

- Try XGBoost or LightGBM
- Analyze Random Forest feature importances
- Explore SMOTE for class imbalance
- Correct Road 2 spot check to use truly default hyperparameters
```
