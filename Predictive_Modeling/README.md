# Predictive Modeling (Stage 3)

Artificial Neural Network (ANN) model for predicting customer churn in a telecommunications dataset.

Role: Data Analyst-Predictive Modeling
Deliverable:Stage 3-Predictive Modeling (3-1) and Final Report inputs (3-2)

---

## Contents

| File | Description |
|---|---|
| `ANN_Architecture.pdf` | Full specification of the ANN architecture (input, hidden, output layers, training configuration). |
| `Stage3_ANN_Churn.ipynb` | Colab notebook containing all code: data loading, model definition, training, evaluation, permutation importance, and segment-level churn analysis. |
| `ann_churn_model.keras` | Trained ANN model file (321 trainable parameters), ready for inference. |
| `training_history.png` | Training and validation loss/accuracy curves over epochs. |
| `confusion_matrix.png` | Confusion matrix on the held-out test set. |
| `roc_curve.png` | ROC curve (AUC = 0.808). |
| `churn_by_segment.png` | Bar chart of churn rate by customer segment. |
| `metrics.csv` | Summary of test-set performance metrics. |
| `permutation_importance.csv` | Feature importance ranking (ROC-AUC drop when shuffled). |
| `churn_by_segment.csv` | Churn rate and predicted probabilities per customer segment. |
| `test_predictions.csv` | Per-customer churn probabilities and predicted labels on the test set. |

---

## Model Summary

- **Type:** Binary classification (churn / no churn)
- **Input:** 10 features
- **Architecture:** Dense(16, ReLU) → Dropout(0.3) → Dense(8, ReLU) → Dense(1, Sigmoid)
- **Parameters:** 321 trainable
- **Optimiser:** Adam | **Loss:** Binary cross-entropy
- **Regularisation:** Dropout, early stopping (patience = 10), class weighting to handle 73.5 / 26.5 class imbalance

## Key Results (Test Set, n = 1,409)

- Accuracy: **0.723**
- Precision (Churn): **0.486**
- Recall (Churn): **0.738**
- F1 (Churn): **0.586**
- ROC-AUC: **0.808**

## Key Findings

- **Tenure** and **MonthlyCharges** are the dominant churn drivers (permutation importance: 0.208 and 0.111 respectively).
- **New high spenders** churn at **50.6%** — the highest-risk segment.
- **Long-term low spenders** churn at just **5.2%** — the safest segment.
- New high spenders represent ~32% of the customer base but account for **61% of all churners**.

---

## Usage

To load the trained model:

```python
from tensorflow.keras.models import load_model

model = load_model('ann_churn_model.keras')
probs = model.predict(X_test)   # returns churn probability per customer
