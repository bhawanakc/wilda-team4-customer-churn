# Data Preparation

Stage 2 deliverable for the customer churn analysis - owner: Rohan Sharma Kharel.

## Overview

One script, [`data_preparation.py`](data_preparation.py), takes the raw dataset at
`data/Dataset_ATS_v2.csv` (7,043 rows, 10 columns) and
produces everything the clustering and modelling stages need. It runs the integrity
checks, handles missing data with median/mode imputation, encodes the categorical
variables, saves the encoded dataset, splits it 80/20 stratified on `Churn`, fits a
`StandardScaler` **on the training set only** and applies it to both sets. Rerun the
whole thing from the repository root with:

```bash
python Data_Preparation/data_preparation.py
```

The two documents in this folder are regenerated from the pipeline's own output, so
none of the numbers below are typed by hand:

```bash
python Data_Preparation/build_docs.py
```

## Files produced

| File | Contents |
|---|---|
| `preprocessed_dataset.csv` | All 7,043 rows, missing data handled and categorical variables encoded. **Not scaled**, so the values stay readable. |
| `train_set.csv` | The 5,634 training rows, scaled. Same 11 columns, `Churn` last. |
| `test_set.csv` | The 1,409 test rows, scaled with the scaler fitted on the training set. |
| `scaler.pkl` | The fitted `StandardScaler`, so later stages apply the identical transformation. |
| `encoding_map.json` | Every category-to-number mapping used, for decoding results. |
| `preparation_summary.json` | The row counts, churn rates and scaler statistics behind this README and the PDF. |
| `scaling_techniques.pdf` | The scaling techniques document: method, columns, leakage and results. |
| `figures/scaling_before_after.png` | Histograms of the training set before and after scaling. |
| `data_preparation.py` | The pipeline itself. |
| `build_docs.py` | Builds this README and the PDF from `preparation_summary.json`. |

## Size and composition

Split with `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)`.
Stratifying on `Churn` is what keeps the churn rate identical across the three rows below.

| Set | Rows | % of total | Churners | Churn rate | Month-to-month | One year | Two year |
|---|---|---|---|---|---|---|---|
| Full dataset | 7,043 | 100.00% | 1,869 | 26.54% | 3,875 | 1,473 | 1,695 |
| Training set | 5,634 | 79.99% | 1,495 | 26.54% | 3,094 | 1,175 | 1,365 |
| Test set | 1,409 | 20.01% | 374 | 26.54% | 781 | 298 | 330 |

Duplicate rows found and kept: **302**. Missing values found: **0**.

Scaled columns, fitted on the training set: `tenure` mean 32.3603, std 24.5639; `MonthlyCharges` mean 64.5351, std 30.2583.
After scaling, the training columns have mean 0 and standard deviation 1 exactly, while the test columns land close by but not exactly (`tenure` mean 0.0022, std 0.9987; `MonthlyCharges` mean 0.0369, std 0.9712).
That small gap is the proof that no test information leaked into the scaler.

## Column dictionary

All 11 columns, in file order, target last.

| Column | Meaning | Type | Encoding |
|---|---|---|---|
| `gender` | Customer's gender | int (0/1) | Female = 0, Male = 1 |
| `SeniorCitizen` | Whether the customer is a senior citizen | int (0/1) | No = 0, Yes = 1 |
| `Dependents` | Whether the customer has dependents | int (0/1) | No = 0, Yes = 1 |
| `tenure` | Months the customer has been with the company | float (scaled) | original units in `preprocessed_dataset.csv`, z-score standardised in the train/test sets |
| `PhoneService` | Whether the customer has phone service | int (0/1) | No = 0, Yes = 1 |
| `MultipleLines` | Whether the customer has multiple lines | int (0/1) | No = 0, Yes = 1 |
| `InternetService` | Type of internet service | int (0/1) | DSL = 0, Fiber optic = 1 |
| `MonthlyCharges` | Amount charged per month, in dollars | float (scaled) | original units in `preprocessed_dataset.csv`, z-score standardised in the train/test sets |
| `Contract_One_year` | Contract is a one year contract | int (0/1) | not a one year contract = 0, one year contract = 1 |
| `Contract_Two_year` | Contract is a two year contract | int (0/1) | not a two year contract = 0, two year contract = 1 |
| `Churn` | TARGET - whether the customer left | int (0/1) | No = 0, Yes = 1 |

Only `tenure` and `MonthlyCharges` are scaled. Every other column is already a 0/1 flag,
and the target is never scaled.

## Decisions and reasons

- **The 302 identical rows are kept.** The dataset has no customer ID and only
  10 fairly coarse columns, so two customers on the same plan with the same tenure and
  the same monthly charge produce identical rows without being the same person. Dropping
  them would throw away real customers and bias the churn rate.
- **Median for numeric, mode for categorical imputation.** The median is not dragged around
  by extreme charges the way the mean is, and the mode is the only sensible fill for a
  category. This dataset has 0 missing values, but the step is part of the deliverable and
  stays in the pipeline so it handles a future refresh of the data.
- **Label encoding for the binary columns, one-hot for `Contract`.** A two-value column maps
  cleanly to 0/1 with no ordering implied. `Contract` has three levels, and numbering them
  0/1/2 would tell the model that a two year contract is "twice" a one year one. One-hot
  with `drop_first=True` avoids that and leaves Month-to-month as the baseline, which is
  both levels set to 0.
- **Stratified 80/20 split.** Churn is imbalanced at 26.54%, so a plain random split could
  hand the test set a noticeably different churn rate. Stratifying holds it at
  26.54% in training and 26.54% in the test set. `random_state=42` makes the split reproducible.
- **The scaler is fitted on the training data only.** Fitting on all 7,043 rows would let the
  test set's mean and standard deviation shape the training data, so the model would be
  evaluated on data it had already been told something about. The test set is transformed
  with the training scaler, exactly as unseen data would be in production.
