# Fuel_Price_ML_Model

A machine learning project that demonstrates how to train, evaluate, and use a regression model to predict Nigerian petrol prices per litre using historical economic data from 2005 to 2025.

The project uses **Polynomial Lasso Regression** to explore nonlinear relationships between fuel prices and economic indicators such as crude oil prices, exchange rates, fuel subsidies, import duties, and VAT.

It was built primarily as a practical learning project to demonstrate how to assess predictive performance, identify potential overfitting, and experiment with techniques for improving model accuracy using **Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R²)**.

The notebook also supports interactive predictions, allowing users to specify a future year and provide their own economic assumptions.

> **Note:** This project is educational. Predictions depend on the quality, size, and patterns of the supplied dataset and should not be treated as authoritative fuel price forecasts.

## Table of Contents

- [Project Objectives](#project-objectives)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Machine Learning Workflow](#machine-learning-workflow)
- [Model Configuration](#model-configuration)
- [Model Evaluation Metrics](#model-evaluation-metrics)
- [Cross-Validation](#cross-validation)
- [Visualizations](#visualizations)
- [Interactive Fuel Price Prediction](#interactive-fuel-price-prediction)
- [Running the Project with Anaconda](#running-the-project-with-anaconda)
- [Understanding the Results](#understanding-the-results)
- [Limitations](#limitations)
- [Possible Improvements](#possible-improvements)
- [Learning Outcomes](#learning-outcomes)

## Project Objectives

The project demonstrates how to:

- Prepare historical data for machine learning.
- Identify target variables and independent features.
- Handle missing values and preprocess numerical and categorical data.
- Apply feature scaling using `StandardScaler`.
- Transform features using polynomial regression.
- Train a regularized regression model using Lasso.
- Evaluate model performance using MAE, MSE, and R².
- Compare training and testing results to investigate overfitting.
- Apply K-Fold and Leave-One-Out cross-validation.
- Visualize actual fuel prices against model predictions.
- Accept user inputs and predict fuel prices for a selected year.
- Understand the limitations of forecasting beyond the observed data.

## Dataset

The project uses `nigeria_fuel_prices.csv`, containing 21 annual observations spanning 2005–2025.

### Dataset columns

| Column | Description |
|---|---|
| `Year` | Calendar year of the observation |
| `Crude_Oil_USD` | Crude oil price in USD |
| `Subsidy_NGN_Billion` | Fuel subsidy amount in NGN billions |
| `FX_NGN_per_USD` | Nigerian naira to US dollar exchange rate |
| `Import_Duty_Pct` | Import duty percentage |
| `VAT_Pct` | Value Added Tax percentage |
| `Price_NGN_per_Litre` | Target variable: petrol price in NGN per litre |

### Target variable

The model predicts:

`Price_NGN_per_Litre`

The remaining columns are used as input features.

The supplied dataset contains no missing values. However, the notebook includes preprocessing steps that can handle missing feature values if they occur in other datasets.

The historical data includes significant fuel price increases during 2023–2025, alongside changes in exchange rates and fuel subsidy policy.

## Technologies Used

- **Python** — Programming language.
- **Anaconda** — Python distribution and environment management.
- **Jupyter Notebook** — Interactive development environment.
- **Pandas** — Data loading, cleaning, and manipulation.
- **NumPy** — Numerical operations.
- **Matplotlib** — Data visualization.
- **Scikit-learn** — Preprocessing, regression, model evaluation, and cross-validation.

## Machine Learning Workflow

The notebook, `fuel_price.ipynb`, implements the following workflow.

### 1. Data Loading and Preparation

The dataset is loaded into a Pandas DataFrame.

The notebook:

- Checks that the target variable and `Year` column exist.
- Converts the target and year columns to numeric values.
- Removes records with missing target values or years.
- Requires at least five valid records.
- Separates features from the target variable.
- Centers the year feature using the earliest year in the dataset.
- Removes feature columns that are entirely empty.

Centering the year reduces the magnitude of the year values supplied to the model, which can help with numerical stability during polynomial feature generation.

### 2. Feature Identification

The notebook automatically identifies numerical and categorical features.

Numerical features include:

- `Year`
- `Crude_Oil_USD`
- `Subsidy_NGN_Billion`
- `FX_NGN_per_USD`
- `Import_Duty_Pct`
- `VAT_Pct`

Categorical features are detected automatically if present in the dataset.

This makes the preprocessing workflow adaptable to datasets containing additional categorical variables.

### 3. Data Preprocessing

The notebook uses scikit-learn's `ColumnTransformer` and `Pipeline` to apply different preprocessing operations to numerical and categorical features.

**Numerical preprocessing**

- Missing values are replaced using median imputation.
- Numerical features are standardized using `StandardScaler`.

**Categorical preprocessing**

- Missing values are replaced using the most frequent value.
- `OneHotEncoder` converts categorical variables into numerical features.
- The encoder ignores previously unseen categories during prediction.

Preprocessing is fitted on the training data and then applied to validation, test, and prediction data through the fitted pipeline. This prevents preprocessing statistics from being learned directly from the test set.

### 4. Train/Test Split

The dataset is divided into:

- **80% training data** — Used to fit the model.
- **20% testing data** — Used to evaluate performance on observations excluded from the training split.

The notebook uses `train_test_split` with `random_state=42` to make the random split reproducible.

The current implementation uses a random split for educational comparison. For more realistic future-year forecasting, a chronological split is recommended.

### 5. Polynomial Feature Engineering

The notebook uses:

`PolynomialFeatures(degree=4, include_bias=False)`

Polynomial feature engineering expands the original features into higher-degree terms and interactions.

For example, a numerical feature \(x\) can generate:

\[
x,\ x^2,\ x^3,\ x^4
\]

Interactions between different features may also be generated.

This enables the model to represent nonlinear relationships rather than being restricted to a simple linear relationship between the original features and fuel prices.

However, polynomial expansion can substantially increase the number of features, especially when several input variables are present.

### 6. Lasso Regression

The project uses Lasso regression with the following configuration:

```python
Lasso(
    alpha=6,
    max_iter=10000
)
```

Lasso adds an L1 regularization penalty to the regression objective.

It helps control model complexity by penalizing large coefficients and can reduce some coefficients to zero.

The `alpha` parameter controls regularization strength:

- A smaller alpha generally allows the model to fit the training data more closely.
- A larger alpha applies stronger regularization and can simplify the model.
- Excessive regularization can cause underfitting.

The appropriate alpha should be selected based on validation performance rather than assuming that a larger value always produces a better model.

## Model Configuration

The complete model is implemented as a scikit-learn pipeline:

```python
model = Pipeline([
    ("preprocessor", preprocessor),
    ("poly", PolynomialFeatures(
        degree=4,
        include_bias=False
    )),
    ("regression", Lasso(
        alpha=6,
        max_iter=10000
    ))
])
```

The pipeline ensures that preprocessing, polynomial feature generation, and regression are applied in the correct sequence.

The model is trained using:

```python
model.fit(X_train, y_train)
```

Predictions are generated using:

```python
y_pred_test = model.predict(X_test)
```

The same trained pipeline is used for future predictions, so the original fitted imputation values, scaling parameters, encoder, polynomial transformation, and regression coefficients are reused.

## Model Evaluation Metrics

The project evaluates regression performance using MAE, MSE, and R².

### Mean Absolute Error (MAE)

MAE measures the average absolute difference between actual and predicted fuel prices.

\[
MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
\]

Where:

- \(y_i\) is the actual fuel price.
- \(\hat{y}_i\) is the predicted fuel price.
- \(n\) is the number of evaluated observations.

MAE is expressed in NGN per litre.

For example, an MAE of 50 means the model's predictions differ from actual prices by an average absolute amount of ₦50 per litre over the evaluated observations.

**Lower MAE indicates better average predictive accuracy.**

### Mean Squared Error (MSE)

MSE measures the average squared prediction error.

\[
MSE=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
\]

MSE penalizes large errors more heavily than small errors.

Because the error is squared, its units are NGN² per litre².

**Lower MSE indicates smaller squared prediction errors.**

### R-squared (R²)

R² measures how much of the variation in the target variable is explained by the model relative to a mean-based baseline.

\[
R^2=1-\frac{\sum_i(y_i-\hat{y}_i)^2}{\sum_i(y_i-\bar y)^2}
\]

Interpretation:

- **R² = 1:** Perfect predictions on the evaluated observations.
- **R² close to 1:** Strong agreement with the observed target variation.
- **R² = 0:** Performance equivalent to predicting the evaluation set's mean target value.
- **R² below 0:** The model performs worse than that mean-based baseline.

R² does not directly measure average error, and a high R² does not guarantee accurate future forecasts.

### Training and Testing Comparison

The notebook calculates MAE and MSE for training and testing data, along with R².

Comparing the results helps investigate whether the model generalizes beyond the data used for fitting.

| Observation | Possible interpretation |
|---|---|
| Low training error and low testing error | Potentially good fit and generalization |
| Low training error but high testing error | Possible overfitting |
| High training error and high testing error | Possible underfitting or insufficient predictive information |
| Negative testing R² | Model performs worse than the test-set mean baseline |

These are diagnostic indicators rather than absolute rules. Conclusions should consider the dataset size, data distribution, validation design, and the scale of the errors.

## Cross-Validation

The notebook implements two cross-validation techniques to examine how performance varies across different subsets of the training data.

### K-Fold Cross-Validation

The notebook uses `KFold` with up to five splits, shuffling the training observations with a fixed random seed.

For each fold:

1. One subset is used for validation.
2. The remaining subsets are used to fit the pipeline.
3. MAE is calculated on the validation subset.
4. The process repeats until every fold has served as validation data.

The notebook reports the MAE for each fold, the average MAE, and the standard deviation.

Scikit-learn's `neg_mean_absolute_error` scoring convention returns negative values, so the notebook negates these scores to report positive MAE values.

### Leave-One-Out Cross-Validation (LOOCV)

LOOCV holds out one training observation at a time and fits the model on all remaining training observations.

It reports:

- Number of validation iterations.
- Average LOOCV MAE.
- Standard deviation of LOOCV MAE.

LOOCV makes extensive use of a small training dataset, but its results can have high variance and may be computationally expensive for larger datasets.

### Comparing Evaluation Results

The notebook compares:

- Training MAE
- Testing MAE
- K-Fold cross-validation MAE
- LOOCV MAE

These comparisons help demonstrate why evaluating a model on a single split may not provide a complete picture of its predictive performance.

For forecasting, shuffled cross-validation should be supplemented or replaced with chronological validation to preserve the ordering of historical observations.

## Model Coefficients and Feature Analysis

The notebook extracts the fitted Lasso regression coefficients and associates them with the generated polynomial feature names.

The resulting table helps demonstrate:

- The effect of polynomial expansion on feature count.
- How feature interactions are represented.
- How Lasso regularization can shrink coefficients to zero.
- Which generated terms have larger absolute coefficients in the fitted model.

Coefficient magnitude alone does not establish causation or reliably measure economic importance, particularly when polynomial terms and correlated features are present.

## Visualizations

The notebook generates the following visualizations.

### Actual vs Predicted Fuel Prices

A scatter plot shows historical fuel prices, while a line shows the model's predictions for those observations.

This helps visualize how closely the fitted predictions follow the historical price pattern.

### MAE Comparison Chart

A bar chart compares MAE across training, testing, K-Fold cross-validation, and LOOCV.

This makes differences in average absolute prediction error easier to inspect.

### Interactive Prediction Visualization

After the user generates a prediction, an additional chart displays:

- Historical actual fuel prices.
- Historical fitted model predictions.
- The predicted price for the selected year.
- A dashed connection between the latest historical fitted prediction and the selected-year prediction.

The dashed connection is only a visual guide; it does not represent model predictions for every intermediate year.

## Interactive Fuel Price Prediction

The notebook allows users to predict fuel prices for a selected year rather than using a hardcoded prediction year.

The user supplies the prediction year and values for the remaining input features.

### Example: Predicting Fuel Prices for 2027

The following hypothetical values can be used to test the model:

| Feature | Example input |
|---|---:|
| `Year` | 2027 |
| `Crude_Oil_USD` | 85 |
| `Subsidy_NGN_Billion` | 0 |
| `FX_NGN_per_USD` | 1700 |
| `Import_Duty_Pct` | 5 |
| `VAT_Pct` | 7.5 |

These are illustrative test assumptions, not verified forecasts of future economic conditions.

The interactive prediction cell:

1. Requests the target year.
2. Requests numerical feature values.
3. Requests categorical feature values if the dataset contains categorical features.
4. Allows blank feature inputs, which are handled through the pipeline's fitted imputation strategy.
5. Creates a DataFrame matching the training feature structure.
6. Applies the same year-centering convention used during training.
7. Passes the data through the fitted pipeline.
8. Displays the predicted fuel price in NGN per litre.

Users can run the prediction cell repeatedly with different years and economic assumptions without retraining the model.

**Important:** Future predictions depend on the supplied assumptions. The model does not independently forecast crude oil prices, exchange rates, subsidy policy, or tax rates.

## Running the Project with Anaconda

### Requirements

- Anaconda Distribution
- Jupyter Notebook or JupyterLab
- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

### 1. Open Anaconda

Launch Anaconda Navigator or open a terminal where Conda is available.

### 2. Create a Conda Environment (Optional)

Create a dedicated environment for the project:

```bash
conda create -n fuel-price-ml python=3.11
```

Activate it:

```bash
conda activate fuel-price-ml
```

If you already have a working Anaconda environment, you can use it instead.

### 3. Install Dependencies

Install the required packages:

```bash
conda install numpy pandas matplotlib scikit-learn notebook
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Alternatively, launch Jupyter Notebook or JupyterLab from Anaconda Navigator.

### 5. Open the Project

Open `fuel_price.ipynb` and ensure that `nigeria_fuel_prices.csv` is in the notebook's working directory.

### 6. Run the Notebook

Execute the cells sequentially, beginning with imports and dataset preparation, followed by preprocessing, training, evaluation, cross-validation, and visualizations.

Finally, run the interactive prediction cell and enter the desired year and economic feature values.

## Understanding the Results

The main objective is to learn how to assess predictive performance rather than simply produce a predicted fuel price.

A model should not be considered accurate merely because it generates a plausible-looking number.

Instead, examine:

- **MAE:** How far predictions are from actual prices on average.
- **MSE:** Whether the model makes particularly large errors.
- **R²:** How well the model explains target variation relative to a mean baseline.
- **Training vs Testing:** Whether the model generalizes beyond its training observations.
- **Cross-validation:** Whether estimated performance varies substantially across different data subsets.

A model with low training error but high testing error may be overfitting. If regularization adjustments do not improve validation performance, consider revisiting the dataset, feature selection, polynomial degree, or regression algorithm.

For a forecasting application, performance should be measured on later years that were not used during training or model selection.

## Limitations

This project has several important limitations.

### Small Dataset

The dataset contains only 21 annual observations. This is a small sample for a model that expands six input features into polynomial terms up to degree four.

The large number of potential polynomial features relative to the number of observations increases the risk of overfitting and unstable predictions.

### Limited Feature Variation

Import duty remains at 5% throughout the supplied dataset, while VAT has only two observed values.

Consequently, the model has limited information for estimating the effects of these features.

### Historical Relationships May Change

Fuel prices are influenced by economic conditions, government policy, market structures, and other factors not fully represented in the dataset.

Historical relationships may not remain stable in future years.

### Extrapolation Risk

Polynomial regression can behave unpredictably outside the range of the training data.

Lasso regularization does not guarantee that predictions will remain economically plausible.

### Evaluation Design

Random train/test splitting and shuffled cross-validation can place observations from later years in training folds while earlier years appear in validation folds.

This does not fully simulate forecasting future fuel prices from past data. Chronological evaluation is a more appropriate next step.

### Educational Purpose

The dataset and model demonstrate machine learning techniques; they do not establish causal relationships between individual economic indicators and fuel prices, nor do they guarantee accurate future predictions.

## Possible Improvements

Future versions of the project could explore:

- Collecting additional historical observations, including monthly or quarterly data.
- Adding inflation, refining costs, transportation costs, global petroleum product prices, and other relevant indicators.
- Comparing linear regression, Ridge Regression, Lasso, Random Forest, and other regression algorithms.
- Tuning the polynomial degree and Lasso alpha using validation data.
- Implementing chronological train/test splitting.
- Using time-series cross-validation.
- Comparing models against simple baselines, such as predicting the most recent observed price.
- Examining residuals and prediction errors across different periods.
- Investigating prediction intervals and uncertainty.
- Deploying the trained model through a web application or REST API.

## Learning Outcomes

By working through this project, learners can gain practical experience with:

1. Data preparation using Pandas.
2. Numerical feature scaling using `StandardScaler`.
3. Categorical encoding using `OneHotEncoder`.
4. Data preprocessing pipelines using scikit-learn.
5. Polynomial feature engineering.
6. Lasso regression and regularization.
7. Model evaluation using MAE, MSE, and R².
8. Overfitting and generalization.
9. K-Fold and Leave-One-Out cross-validation.
10. Model coefficient inspection.
11. Data visualization using Matplotlib.
12. Interactive predictions using a trained model.
13. The importance of validation design for forecasting.

## Project Purpose

**Fuel_Price_ML_Model is a practical demonstration of the machine learning development and evaluation process.**

It shows how a model can be trained on historical data, assessed using MAE, MSE, and R², examined for potential overfitting, and used to generate predictions from user-provided inputs.

The project's central learning objective is to understand how to evaluate a model's predictive accuracy against available data and how to investigate improvements through preprocessing, regularization, validation, and model selection.

The predicted fuel price is an output of the fitted model under the assumptions supplied by the user—not a guarantee of future market prices.