# 🛒 Walmart Sales Forecasting — Supervised Machine Learning (CDSD · Jedha · Bloc 3)

## Overview
This project addresses **weekly sales forecasting** across Walmart stores using linear models with and without regularization. The objective is to benchmark **Linear Regression**, **Ridge (L2)**, and **Lasso (L1)** in terms of predictive performance, stability, and feature interpretability.

🔗 **Colab Notebook:** [Open in Colab](https://colab.research.google.com/drive/1pkOd0fQmOiZsRZ3TH-XpG8MKBYnF3?usp=sharing)

## Methodology

### Preprocessing
- **Numeric features**: imputation (mean), scaling (`StandardScaler`).
- **Categorical features**: one-hot encoding (`OneHotEncoder`, `drop="first"`).
- **Temporal features**: sine/cosine transformations for cyclic effects (day, month).
- Implemented via `ColumnTransformer` inside scikit-learn pipelines.

### Train/Test Split
- Data split chronologically by **date**: train on earlier periods, test on later periods.
- Prevents look-ahead bias and better simulates real forecasting.

### Modeling
- **Baseline**: `LinearRegression`.
- **RidgeCV**: evaluates a log-spaced grid of `alpha ∈ [1e-4, 1e2]` with 5-fold CV to select optimal regularization.
- **LassoCV**: 5-fold CV to automatically select best `alpha`.
- Metrics: **R²** on train/test, generalization gap, CV mean ± std.

### Regularization Effects
- **Ridge**: shrinks coefficients smoothly, improving generalization without removing features.
- **Lasso**: performs feature selection by driving non-informative coefficients to zero.
- Both reduce overfitting compared to plain Linear Regression.

### Feature Importance
- Coefficients mapped back to feature names via `preprocessor.get_feature_names_out()`.
- Comparative visualization: grouped bar chart of coefficients across Linear, Ridge, and Lasso.
- Sorting by absolute value highlights the strongest drivers of sales.

## Results & Insights
- **Store identity** is the dominant driver of sales (large absolute weights).
- **Holiday flag** shows a negative coefficient once controls are included, contrasting with raw median sales → indicates confounding effects.
- **Macroeconomic variables** (CPI, Fuel Price, Unemployment) contribute modestly but consistently.
- **Ridge** reduces coefficient variance → more stable.
- **Lasso** enforces sparsity → fewer but more interpretable predictors.

## Tech Stack
- Python (pandas, numpy, scikit-learn)
- Models: `LinearRegression`, `RidgeCV`, `LassoCV`
- Visualization: Plotly, Matplotlib
- Environment: Google Colab
