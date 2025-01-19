# Credit Card Fraud Detection

This project focuses on detecting fraudulent credit card transactions using machine learning techniques. The dataset used is from kaggle.

## Project Overview

The goal of this project is to detect fraudulent credit card transactions. Fraud detection is a significant challenge due to the highly imbalanced nature of the dataset. This project demonstrates:
- Preprocessing the data.
- Balancing the dataset using SMOTE.
- Building machine learning models.
- Evaluating model performance using metrics like confusion matrix, AUC-ROC, and SHAP analysis.

---

## Technologies Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)
- SHAP

---

## Dataset

The dataset contains credit card transactions from European cardholders in September 2013. It consists of 31 features:
- **Time**: Seconds elapsed between transactions.
- **Amount**: Transaction amount.
- **Class**: 0 (non-fraudulent) or 1 (fraudulent).

The dataset is highly imbalanced, with only ~0.17% fraudulent transactions.

---

## Exploratory Data Analysis (EDA)

- **Class Distribution**: A pie chart visualizing the imbalance between fraudulent and non-fraudulent transactions.
- **Transaction Amounts Over Time**: Scatter plot showing transaction amounts by time for each class.
- **Correlation Matrix**: Heatmap to identify relationships between features.

---

## Data Preprocessing

1. **Missing Values**: Verified and dropped missing values.
2. **Feature Scaling**: Standardized `Time` and `Amount` features using `StandardScaler`.
3. **Balancing the Dataset**: Used SMOTE to balance the dataset.

---

## Modeling

### Logistic Regression

- Built a baseline logistic regression model.
- Evaluated using metrics like:
  - Confusion Matrix
  - Classification Report
  - AUC-ROC Curve

### Random Forest Classifier

- Trained a random forest model to extract feature importance.
- Visualized the top 10 features contributing to fraud detection.

---

## Feature Importance

- Used Random Forest feature importances to identify the most critical features for fraud detection.

---

## SHAP Analysis

- Performed SHAP (SHapley Additive exPlanations) analysis for:
  - Global interpretability (summary plot and bar chart).
  - Local interpretability (individual prediction explanations).
