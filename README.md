# Fuel_Price_ML_Model

Machine learning project that models and predicts Nigerian fuel (petrol) prices per litre using a small historical dataset spanning 2005–2025.

## Dataset

`nigeria_fuel_prices.csv` contains annual records with the following columns:

| Column | Description |
|---|---|
| `Year` | Calendar year |
| `Crude_Oil_USD` | Crude oil price in USD |
| `Subsidy_NGN_Billion` | Fuel subsidy amount in NGN billions |
| `FX_NGN_per_USD` | NGN/USD exchange rate |
| `Import_Duty_Pct` | Import duty percentage |
| `VAT_Pct` | VAT percentage |
| `Price_NGN_per_Litre` | Target: fuel price in NGN per litre |

21 rows, no missing values. Prices jump sharply in 2023–2025 following subsidy removal and FX devaluation.

## Notebook

### `fuel_price.ipynb` — Polynomial Lasso Regression

- Loads and cleans the data, coercing `Price_NGN_per_Litre` and `Year` to numeric, dropping rows missing target/year, requiring at least 5 records.
- Centers `Year` by its minimum to reduce numerical instability.
- Uses all columns except the target as features.
- Builds a scikit-learn `ColumnTransformer` pipeline: median imputation + `StandardScaler` for numeric features, most-frequent imputation + `OneHotEncoder` (version-aware `sparse_output`/`sparse`) for categorical features.
- Trains a `Pipeline` of preprocessor → `PolynomialFeatures(degree=4)` → `Lasso(alpha=6, max_iter=10000)`.
- Evaluates with MAE/MSE on train and test, plus train R² (test R² only if ≥2 test records).
- Prints Lasso coefficients over polynomial feature names.
- K-Fold (up to 5 splits) and Leave-One-Out cross-validation using negated MAE, then compares Training/Testing/K-Fold/LOOCV MAE.
- Plots actual vs predicted prices by year and a bar chart of MAE by evaluation method.
- Predicts the 2030 price by reusing the last record's features with `Year` set to 2030 (centered).

## Running

Open the notebook in Jupyter and run cells top to bottom; ensure `nigeria_fuel_prices.csv` is in the working directory. Dependencies: `pandas`, `numpy`, `matplotlib`, `scikit-learn`.
