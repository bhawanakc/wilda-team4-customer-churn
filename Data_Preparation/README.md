# Data preparation

Stage 2 data preparation for the customer churn analysis. Owner: Rohan Sharma Kharel.

## Overview

[`data_preparation.py`](data_preparation.py) reads the raw dataset at
`data/Dataset_ATS_v2.csv` (7,043 rows, 10 columns) and writes every file the
clustering and modelling stages need. The script runs in this order:

1. Check the data for duplicates, stray whitespace and invalid values.
2. Fill any missing values, using the median for numbers and the mode for categories.
3. Encode the categorical columns and save the encoded dataset.
4. Split the rows 80/20 into training and test sets, stratified on `Churn`.
5. Fit a `StandardScaler` on the training set only, then use it to scale both sets.

To rebuild every output, run this from the repository root:

```bash
python Data_Preparation/data_preparation.py
```

[`build_docs.py`](build_docs.py) writes this README and the PDF from the numbers the
pipeline saves, so nobody types the figures by hand:

```bash
python Data_Preparation/build_docs.py
```

## Files produced

| File | Contents |
|---|---|
| `preprocessed_dataset.csv` | All 7,043 rows with missing data handled and categories encoded. Not scaled, so tenure is in months and charges are in dollars. |
| `train_set.csv` | The 5,634 training rows, scaled. Same 11 columns, with `Churn` last. |
| `test_set.csv` | The 1,409 test rows, scaled with the scaler fitted on the training set. |
| `scaler.pkl` | The fitted `StandardScaler`. Later stages load it to scale new data the same way. |
| `encoding_map.json` | The number each category was mapped to. |
| `preparation_summary.json` | The row counts, churn rates and scaler statistics used in this README and the PDF. |
| `scaling_techniques.pdf` | Which scaler we used and why, which columns it scaled, how we avoided data leakage, and the results. |
| `figures/scaling_before_after.png` | Histograms of the training set before and after scaling. |
| `data_preparation.py` | The pipeline. |
| `build_docs.py` | Builds this README and the PDF from `preparation_summary.json`. |

## Size and composition

We split the data with `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)`.
Stratifying on `Churn` gives all three rows below the same churn rate.

| Set | Rows | % of total | Churners | Churn rate | Month-to-month | One year | Two year |
|---|---|---|---|---|---|---|---|
| Full dataset | 7,043 | 100.00% | 1,869 | 26.54% | 3,875 | 1,473 | 1,695 |
| Training set | 5,634 | 79.99% | 1,495 | 26.54% | 3,094 | 1,175 | 1,365 |
| Test set | 1,409 | 20.01% | 374 | 26.54% | 781 | 298 | 330 |

The raw data has 302 duplicate rows, and the pipeline keeps them. It has 0 missing values.

The scaler learned these values from the training set:

| Column | Mean | Standard deviation |
|---|---|---|
| `tenure` | 32.3603 | 24.5639 |
| `MonthlyCharges` | 64.5351 | 30.2583 |

After scaling, both training columns have a mean of 0 and a standard deviation of 1.
The test columns come out close to those values but not equal to them. `tenure` has
mean 0.0022 and standard deviation 0.9987, and `MonthlyCharges` has
mean 0.0369 and standard deviation 0.9712. The small gap shows the scaler
never saw the test rows.

## Column dictionary

The 11 columns, in file order, with the target last.

| Column | Meaning | Type | Encoding |
|---|---|---|---|
| `gender` | Customer's gender | int (0/1) | Female = 0, Male = 1 |
| `SeniorCitizen` | Whether the customer is a senior citizen | int (0/1) | No = 0, Yes = 1 |
| `Dependents` | Whether the customer has dependents | int (0/1) | No = 0, Yes = 1 |
| `tenure` | Months the customer has been with the company | float (scaled) | original units in `preprocessed_dataset.csv`, z-scores in the train and test sets |
| `PhoneService` | Whether the customer has phone service | int (0/1) | No = 0, Yes = 1 |
| `MultipleLines` | Whether the customer has multiple lines | int (0/1) | No = 0, Yes = 1 |
| `InternetService` | Type of internet service | int (0/1) | DSL = 0, Fiber optic = 1 |
| `MonthlyCharges` | Amount charged per month, in dollars | float (scaled) | original units in `preprocessed_dataset.csv`, z-scores in the train and test sets |
| `Contract_One_year` | Contract is a one year contract | int (0/1) | not a one year contract = 0, one year contract = 1 |
| `Contract_Two_year` | Contract is a two year contract | int (0/1) | not a two year contract = 0, two year contract = 1 |
| `Churn` | Target. Whether the customer left | int (0/1) | No = 0, Yes = 1 |

The pipeline scales only `tenure` and `MonthlyCharges`. Every other feature is a 0/1
flag, and the target stays as 0/1.

## Decisions and reasons

- **We kept the 302 duplicate rows.** The dataset has no customer ID and only
  10 columns, most with two or three possible values. Two different customers on the
  same plan, with the same tenure and the same monthly charge, produce identical rows.
  Dropping the duplicates would remove 302 real customers from the analysis.
- **We fill missing numbers with the median and missing categories with the mode.** Extreme
  values move the mean more than the median. The mode is the most common category. This dataset has 0 missing values, so the step changes
  nothing today. It stays in the pipeline because the assignment requires it and a later
  version of the data may have gaps.
- **We label encode the two-value columns and one-hot encode `Contract`.** A column with two
  values maps to 0 and 1 without implying any order. `Contract` has three values. Numbering
  them 0, 1 and 2 would tell the model that a two year contract is twice a one year
  contract. One-hot encoding with `drop_first=True` creates two 0/1 columns instead. A
  month-to-month customer has 0 in both.
- **We stratified the 80/20 split on `Churn`.** Only 26.54% of customers churned,
  so a plain random split could give the test set a different churn rate. Stratifying gives
  26.54% in the training set and 26.54% in the test set. `random_state=42`
  makes the split the same on every run.
- **We fitted the scaler on the training set only.** If we fitted it on all 7,043 rows,
  the test set's mean and standard deviation would change how the training data is scaled.
  The test results would then look better than the model would do on new customers. The
  pipeline scales the test set with the training scaler, the same way it would scale new
  customer data.
