# 🚀 Conversion Rate Prediction — Supervised Machine Learning (CDSD · Jedha · Bloc 3)

## Overview
This project tackles the challenge of predicting **whether a user will convert** based on demographic and behavioral web session data. The goal is to build an accurate and interpretable classifier using advanced ensemble methods and neural networks.

The project compares a variety of models including **Logistic Regression**, **Random Forest**, **XGBoost**, **SGDClassifier**, and **Multi-Layer Perceptron (MLP)**. The best-performing models were combined via a **VotingClassifier** with soft voting, and the decision threshold was tuned to maximize F1-score on imbalanced data.

🔗 **Colab Notebook**: [Open in Colab]([https://colab.research.google.com/](https://colab.research.google.com/drive/1n1MrT0vhJf1kBOgBWFmOmPVVjs8XyjBA?usp=sharing 
))

## Methodology

### Preprocessing
- **Numeric features**: scaled using `StandardScaler`.
- **Categorical features**: one-hot encoded (`OneHotEncoder`, `drop='first'`).
- Features include: `country`, `age`, `new_user`, `source`, `total_pages_visited`.
- All transformations managed using `ColumnTransformer` and `Pipeline` from `scikit-learn`.

### Train/Test Split
- Stratified train/test split (80%/20%) to maintain class distribution.
- Evaluation of models based on test set F1-score and optimal probability thresholds.

### Modeling
- **Baseline models tested**:
  - `LogisticRegression` (standalone and inside `BaggingClassifier`)
  - `RandomForestClassifier`
  - `XGBClassifier` (XGBoost)
  - `SGDClassifier` with log loss
  - `MLPClassifier` (2 hidden layers, ReLU activation)
- **Ensemble**:
  - `VotingClassifier` with soft voting, combining:
    - `BaggingClassifier(LogisticRegression)`
    - `MLPClassifier`
- **Threshold Optimization**:
  - Threshold tuned post-prediction to maximize **F1-score** on the test set.

### Threshold Selection
- Instead of using the default 0.5 cutoff, we computed F1-scores across a range of thresholds (`0.1 → 0.9`) and selected the one yielding the best F1 performance.

### Final Prediction
- The best-performing model (`VotingClassifier`) was retrained on **all labeled data**.
- Applied to a **new unlabeled dataset** (`conversion_data_test.csv`) after applying the same preprocessing pipeline.
- Predictions were thresholded using the previously optimized value and exported to CSV.

## Results & Insights
- Combining `Bagging(LogReg)` and `MLP` via soft voting yielded **stronger F1 performance** than any model alone.
- **Threshold tuning** significantly improved classification on imbalanced data by reducing false positives.

## Tech Stack
- Python (pandas, numpy, scikit-learn, xgboost)
- Models: `LogisticRegression`, `SGDClassifier`, `RandomForestClassifier`, `XGBClassifier`, `MLPClassifier`, `BaggingClassifier`, `VotingClassifier`
- Preprocessing: `StandardScaler`, `OneHotEncoder`, `ColumnTransformer`, `Pipeline`
- Metrics: `F1-score`, `classification_report`, `confusion_matrix`
- Environment: Google Colab / Jupyter Notebook
