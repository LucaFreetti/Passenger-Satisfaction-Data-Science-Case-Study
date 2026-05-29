# ✈️ Passenger Satisfaction — Data Science Case Study

A binary classification project built for **Sicilian Airlines**, aimed at predicting whether a passenger will be satisfied or not based on demographic information, flight type and in-flight service ratings.

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
- Satisfied passengers tend to be slightly older — median age ~43 vs ~36

### 3. Outlier Analysis
- `Flight Distance` and `Departure Delay in Minutes` show long right tails confirmed via IQR criterion
- Outliers not removed — they carry real signal
- Median imputation chosen precisely for its robustness to extreme values

### 4. Preprocessing
- Train / Validation / Test split: **70% / 15% / 15%** (cascaded, fully disjoint)
- Numerical features: median imputation
- Categorical features: OrdinalEncoder
- All wrapped in a **sklearn Pipeline** to prevent data leakage
- Global seed set to `SEED = 7` for full reproducibility (`random.seed`, `np.random.seed`, all model constructors)

### 5. Baseline
A `DummyClassifier` (majority class predictor) was used as a trivial benchmark before any real model:
- **DummyClassifier accuracy: ~57%**
- This is the floor — everything above represents real learning

### 6. Feature Selection
Three methods compared on the training set:

| Method | Top Features |
|--------|-------------|
| Chi-Square | Type of Travel, Class, Flight Distance, Age, Customer Type |
| Mutual Information | Online boarding, Inflight wifi, Class, Type of Travel, Inflight entertainment |
| T-Test (Welch) | Age, Flight Distance, Inflight wifi service, Ease of Online booking |

Features appearing in at least 2 out of 3 methods: **Class, Age, Type of Travel, Flight Distance, Online boarding, Inflight wifi service**

### 7. Modeling — Two Roads

**Road 1:** all available features (22 columns)  
**Road 2:** top 4 features from T-test only — Age, Flight Distance, Inflight wifi service, Ease of Online booking

Both roads follow the same pipeline:
1. Spot check with Stratified 5-Fold CV (default hyperparameters) — scoring: **Accuracy, F1, ROC-AUC**
2. Select top 2 models
3. Hyperparameter tuning via GridSearchCV (5-fold)
4. Final evaluation on the held-out test set

### 8. Models
- Logistic Regression *(baseline reference)*
- Random Forest
- AdaBoost
- Gradient Boosting

### 9. Seed Analysis
The final model was re-evaluated across 3 different random splits (seeds 7, 21, 99) to confirm the ~97% result is stable and not the product of a single lucky split.

---

## 📊 Results

| Road | Model | Test Accuracy | Test F1 |
|------|-------|--------------|---------|
| Baseline | DummyClassifier | ~57% | — |
| Road 1 — all features | Random Forest (tuned) | ~97% | ~97% |
| Road 2 — 4 features | Gradient Boosting (tuned) | ~82% | ~81% |

**Winner: Random Forest on Road 1 — ~97% accuracy on the held-out test set (+40pp vs baseline).**

---

## 💡 Key Insights

- **Online boarding** is the single most discriminating feature across all three selection methods
- **Class** and **Type of Travel** dominate Chi-square and Mutual Information rankings
- Using only 4 features reaches ~82% — adding the full feature set gains ~15 points
- **Gate location** shows virtually no difference between groups (p_value ~0.8) — not useful

---

## 🛠️ Action Points

- Fix the online boarding experience — largest satisfaction gap in the dataset
- Invest in inflight entertainment and seat comfort
- Upgrade inflight wifi — top predictor across all three selection methods
- Focus on Economy cabin and leisure travelers — lowest satisfaction rates, highest room for improvement

---

## ⚠️ Limitations

- **Single-airline snapshot:** results may not generalize across carriers or time periods
- **Self-reported ratings:** service scores (0–5) are subjective and subject to response bias
- **OrdinalEncoder on `Class`:** assumes uniform distance between Eco / Eco Plus / Business, which may not reflect real experience differences

---

## 🔧 Tech Stack

- Python 3
- pandas · numpy · scipy
- scikit-learn
- matplotlib · seaborn

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
- Analyze Random Forest feature importances in depth
- Explore SMOTE for class imbalance
- Extend seed analysis to a wider range of splits
