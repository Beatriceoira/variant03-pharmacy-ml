# Variant 03 — Pharmacy Stock-Out Risk Prediction & Medicine Demand Forecasting

## Overview

This project implements two machine learning tasks using pharmacy-related datasets:

1. **Classification:** Predict whether a medicine is at risk of stock-out.
2. **Forecasting:** Predict daily medicine demand for the next **7 days**.

The project demonstrates data preprocessing, exploratory data analysis, feature engineering, machine learning classification, threshold adjustment, hyperparameter tuning, and time-series forecasting using Python.

## Project Objectives

### Classification Task

Predict the `Stock_Out_Risk` of medicines using inventory, demand, supplier, category, and cost-related information.

### Forecasting Task

Forecast daily medicine demand for the next **7 days** using historical daily demand data.

## Datasets

### 1. Pharmacy Stock-Out Risk Dataset

**File:**

```text
P03_pharmacy_stockout_risk.csv
```

The dataset contains **480 records** and **9 columns**.

| Feature                   | Description                                |
| ------------------------- | ------------------------------------------ |
| `Medicine_ID`             | Unique medicine identifier                 |
| `Current_Stock_Units`     | Current available stock                    |
| `Average_Daily_Demand`    | Average daily medicine demand              |
| `Supplier_Lead_Time_Days` | Expected supplier delivery time            |
| `Medicine_Category`       | Medicine category                          |
| `Critical_Medicine`       | Indicates whether the medicine is critical |
| `Pending_Order_Units`     | Units currently on order                   |
| `Unit_Cost_PHP`           | Cost per medicine unit in Philippine pesos |
| `Stock_Out_Risk`          | Target variable indicating stock-out risk  |

The target variable is:

```text
Stock_Out_Risk
```

with two classes:

```text
Yes
No
```

### 2. Daily Medicine Demand Dataset

**File:**

```text
P03_daily_medicine_demand.csv
```

The dataset contains historical daily medicine demand records with:

| Feature                 | Description                       |
| ----------------------- | --------------------------------- |
| `Date`                  | Date of observation               |
| `Medicine_Demand_Units` | Number of medicine units demanded |

The records are ordered chronologically before time-series analysis and forecasting.

## Technologies Used

* Python 3
* Jupyter Notebook
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Statsmodels

### Machine Learning

* Random Forest Classifier
* Train/Test Split
* One-Hot Encoding
* Missing-value Imputation
* Probability-based classification
* Decision-threshold adjustment
* Hyperparameter tuning

### Forecasting

* Holt-Winters Exponential Smoothing
* Additive trend
* Weekly seasonality
* 7-day forecast horizon

## Project Workflow

### Classification

The classification workflow consists of:

1. Load the pharmacy stock-out dataset.
2. Inspect the dataset dimensions, columns, data types, and missing values.
3. Perform descriptive statistical analysis.
4. Examine the distribution of the target variable.
5. Investigate relationships and correlations between variables.
6. Identify duplicates, invalid values, and potential outliers.
7. Clean missing and invalid data programmatically.
8. Remove inappropriate or redundant features such as the medicine identifier.
9. Encode categorical variables.
10. Split the dataset into training and testing sets using a reproducible random state.
11. Train a Random Forest classifier.
12. Generate predictions on the test set.
13. Evaluate the model using:

    * Accuracy
    * Precision
    * Recall
    * F1-score
    * Confusion Matrix
14. Generate prediction probabilities.
15. Test a modified decision threshold.
16. Compare the default and modified thresholds.
17. Tune the Random Forest hyperparameters.
18. Compare the original and tuned models.
19. Select the final model based on the evaluation results.

### Forecasting

The forecasting workflow consists of:

1. Load the daily medicine demand dataset.
2. Convert the `Date` column into a datetime format.
3. Check for missing and duplicate dates.
4. Handle missing demand values.
5. Sort the data chronologically.
6. Visualize historical medicine demand.
7. Examine rolling demand behavior.
8. Reserve the most recent observations for validation.
9. Train a Holt-Winters Exponential Smoothing model.
10. Evaluate the validation forecast using:

    * Mean Absolute Error (MAE)
    * Root Mean Squared Error (RMSE)
11. Fit the final model using the complete historical dataset.
12. Forecast medicine demand for the next **7 days**.
13. Present the final forecast in a table containing:

    * Future Period
    * Forecasted Value

## Model Evaluation

The classification model is evaluated using multiple metrics rather than accuracy alone.

### Accuracy

Measures the proportion of correctly classified observations.

### Precision

Measures how many observations predicted as stock-out risk actually belong to the positive class.

### Recall

Measures how many actual stock-out-risk observations were successfully identified.

### F1-Score

Provides a balance between precision and recall.

### Confusion Matrix

Shows the number of:

* True Negatives
* False Positives
* False Negatives
* True Positives

---

## Decision Threshold Analysis

The classification model produces a probability of stock-out risk.

The default classification threshold is:

```text
0.50
```

A second, lower threshold is tested to investigate how changing the decision boundary affects precision, recall, and F1-score.

This allows the project to demonstrate the trade-off between identifying more potential stock-out cases and increasing false-positive predictions.

## Hyperparameter Tuning

The Random Forest model is also trained with modified hyperparameters.

The tuned model changes parameters including:

```python
n_estimators=400
max_depth=10
min_samples_split=5
```

The original and tuned models are evaluated using the same test dataset so their performance can be compared consistently.

## Forecasting Model

The forecasting component uses the **Holt-Winters Exponential Smoothing** method with:

* Additive trend
* Additive weekly seasonality
* Seasonal period of 7 days

The model is used to capture daily demand patterns and produce a seven-day forecast.

## Repository Structure

```text
variant03-pharmacy-ml/
│
├── P03_pharmacy_stockout_risk.csv
├── P03_daily_medicine_demand.csv
├── VARIANT03_Pharmacy_ML.ipynb
├── README.md
└── .gitignore
```

The Python virtual environment should **not** be committed to GitHub.

```text
.venv/
```

should be excluded using `.gitignore`.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/variant03-pharmacy-ml.git
cd variant03-pharmacy-ml
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

### 3. Activate the environment

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels jupyter openpyxl
```

### 5. Start Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
VARIANT03_Pharmacy_ML.ipynb
```

and run the notebook cells sequentially.

---

## Reproducibility

The classification train/test split uses a fixed random state:

```python
random_state=42
```

This allows the experiment to be reproduced consistently.

The machine learning preprocessing and model training are implemented programmatically within the notebook.

## Expected Outputs

The notebook produces:

* Dataset previews
* Dataset dimensions
* Column and data-type information
* Missing-value summaries
* Descriptive statistics
* Target-class distribution
* Correlation analysis
* Relationship visualizations
* Outlier investigation
* Preprocessed ML-ready data
* Classification predictions
* Classification metrics
* Confusion matrices
* Prediction probabilities
* Decision-threshold comparison
* Original vs. tuned model comparison
* Historical demand visualization
* Forecast validation results
* Seven-day demand forecast
* Final forecast table

## Conclusion

This project demonstrates an end-to-end machine learning workflow for pharmacy inventory analysis. The classification component identifies medicines that may be at risk of stock-out, while the forecasting component estimates future medicine demand. Together, these techniques can provide useful analytical support for inventory monitoring, replenishment planning, and demand management.
