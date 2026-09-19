# Clustering analysis

Stage 2 clustering for the customer churn analysis. Owner: Ramesh Kunwar.

## Overview

[`clustering_analysis.py`](clustering_analysis.py) reads the training set that the data
preparation stage wrote and groups the customers into 4 segments with K-Means. The
script runs in this order:

1. Load `Data_Preparation/train_set.csv` and split `Churn` off the 10 features.
2. Fit K-Means for every k from 1 to 10 and record the inertia, the fall in inertia and the silhouette score.
3. Pick k = 4 with the elbow method.
4. Train the final model, then assign the 1,409 test rows with `model.predict`.
5. Profile each cluster and name it from its own average tenure and monthly charge.
6. Draw the five charts in `cluster_visualisations/`.

The model never sees `Churn`. Stage 3 predicts it, so a cluster built from it would hand
the answer to the model. The pipeline uses `Churn` only after clustering, to measure the
churn rate of each finished segment.

To rebuild every output, run this from the repository root:

```bash
python Clustering_Analysis/clustering_analysis.py
```

[`build_docs.py`](build_docs.py) writes this README and the PDF from the numbers the
pipeline saves, so nobody types the figures by hand:

```bash
python Clustering_Analysis/build_docs.py
```

## Files produced

| File | Contents |
|---|---|
| `clustering_analysis.py` | The pipeline. Chooses k, trains the model, profiles and names the segments, and draws the charts. |
| `clustering_analysis.ipynb` | A notebook that runs the same script and explains each step. Both write identical files. |
| `build_docs.py` | Builds this README and `elbow_method_results.pdf` from `clustering_summary.json`. |
| `kmeans_model.pkl` | The trained K-Means model, fitted on the 5,634 training rows and the 10 features. |
| `cluster_selection_metrics.csv` | Inertia, the percentage fall in inertia and the silhouette score for every k from 1 to 10. |
| `cluster_profiles.csv` | One row per segment, sorted by churn rate. Tenure and charges are in months and dollars. |
| `cluster_assignments.csv` | The segment of all 7,043 customers, for the Stage 3 model to join on. |
| `clustering_summary.json` | Every number used in this README and the PDF. |
| `elbow_method_results.pdf` | How k was chosen, the metrics table, the elbow chart and the trained model settings. |
| `README.md` | This file. The segments, their profiles and the action for each one. |
| `cluster_visualisations/elbow_method.png` | The inertia curve with the elbow at k = 4 marked, and the silhouette scores below it. |
| `cluster_visualisations/segments_tenure_vs_charges.png` | Every training customer in months and dollars, coloured and labelled by segment. |
| `cluster_visualisations/segments_pca.png` | The 10 features projected into 2D, coloured and labelled by segment. |
| `cluster_visualisations/churn_rate_by_segment.png` | Churn rate per segment against the 26.54% overall rate. |
| `cluster_visualisations/segment_profiles.png` | The profile table as a heatmap, with the real months, dollars and percentages shown. |

## How we chose k

We used the elbow method, which the assignment requires. Inertia is the total squared
distance from each customer to its own cluster centre, and it falls every time k rises, so
the choice comes from the size of each fall rather than from the value. Inertia falls
25.2% at k = 2, 18.0% at k = 3 and 12.1% at k = 4, then only 5.1% at k = 5 and less after
that, so k = 4 is the last cluster that buys a large improvement. The silhouette score is a
supporting check and it peaks at k = 2 (0.227), but two clusters split the base into
little more than new and long standing customers, which is too coarse to act on. At k = 4
the churn rates run from 4.8% to 48.7%, which separates the customers worth
contacting from the ones who stay. [`elbow_method_results.pdf`](elbow_method_results.pdf)
sets out the full reasoning.

| k | Inertia | Fall in inertia | Silhouette |
|---|---|---|---|
| 1 | 19,845.66 |  |  |
| 2 | 14,844.21 | 25.20% | 0.2274 |
| 3 | 12,178.36 | 17.96% | 0.2116 |
| 4 | 10,700.86 | 12.13% | 0.2031 |
| 5 | 10,152.73 | 5.12% | 0.1801 |
| 6 | 9,712.96 | 4.33% | 0.1565 |
| 7 | 9,382.54 | 3.40% | 0.1397 |
| 8 | 9,011.45 | 3.96% | 0.1393 |
| 9 | 8,826.27 | 2.05% | 0.1328 |
| 10 | 8,594.34 | 2.63% | 0.1287 |

## The segments

| Segment | Customers (train) | % of customers | Churn rate | Avg tenure (months) | Avg monthly charge ($) | % on fibre | % on a one or two year contract |
|---|---|---|---|---|---|---|---|
| New high spenders | 1,775 | 31.5% | 48.7% | 14.9 | $81.10 | 41.6% | 44.3% |
| New low spenders | 1,381 | 24.5% | 24.8% | 10.5 | $32.10 | 43.5% | 46.0% |
| Long-term high spenders | 1,512 | 26.8% | 16.1% | 58.7 | $93.80 | 45.4% | 44.3% |
| Long-term low spenders | 966 | 17.1% | 4.8% | 54.5 | $34.80 | 44.5% | 46.4% |

Cluster ID numbers can move between runs and between library versions, so every document
refers to the segments by name. The names come from code, not from a person reading the
table. The pipeline converts each cluster centre back into months and dollars and compares
it with the training means, 32.36 months and $64.54. Tenure below the mean gives
"New" and above gives "Long-term". A monthly charge below the mean gives "low spenders"
and above gives "high spenders".

### New high spenders

This segment holds 1,775 of the 5,634 training customers, 31.5% of the base. 864 of them left, a churn rate of 48.7%, against 26.54% across all customers. The average customer here has been with the company 14.9 months and pays $81.10 a month. 55.7% are on a month to month plan. In the held out test set the same segment holds 449 customers and churns at 50.6%. Two numbers in the profile line up with the churn rate. These customers are well below the 32.4 month average tenure and well above the $64.54 average bill, so they pay a lot early. Nothing holds them either, because more than half have no fixed term and can leave at the end of any month. The segment also carries the largest share of senior citizens, 23.7%.

What the company can do:

1. Put the retention budget here first. This segment produced 864 of the 1,495 churners in the training data, more than the other three segments added together.
2. Offer a fixed term plan to the 55.7% on month to month, priced at or below the $81.10 they already pay, within the first 15 months.

### New low spenders

This segment holds 1,381 of the 5,634 training customers, 24.5% of the base. 342 of them left, a churn rate of 24.8%, against 26.54% across all customers. The average customer here has been with the company 10.5 months and pays $32.10 a month. 54.0% are on a month to month plan. In the held out test set the same segment holds 345 customers and churns at 22.9%. This is the newest segment and the cheapest of the four. Its churn rate sits just below the 26.54% overall rate, so these customers leave at close to the average pace. The low bill is what separates them from the New high spenders, who pay $49.00 more a month and churn at 48.7%. This segment also takes the smallest bundle, with only 79.5% on phone service against 99.8% in the New high spenders.

What the company can do:

1. Target the first year. The average customer is 10.5 months in, so a fixed term offer reaches them well before the two long tenure segments, which sit at 54.5 months and above.
2. Test a phone service add-on with the 20.5% who do not have one, and hold the bill near the $32.10 they pay now.

### Long-term high spenders

This segment holds 1,512 of the 5,634 training customers, 26.8% of the base. 243 of them left, a churn rate of 16.1%, against 26.54% across all customers. The average customer here has been with the company 58.7 months and pays $93.80 a month. 55.7% are on a month to month plan. In the held out test set the same segment holds 402 customers and churns at 14.2%. These customers pay the most and have stayed the longest of the four segments. Long tenure goes with the churn rate of 16.1%, below the 26.54% overall rate. The bills are what makes the segment matter. The 243 who left were billing about $22,793 a month between them, using the segment average.

What the company can do:

1. Watch this segment for revenue rather than for counts. Each departure costs about $93.80 a month, 2.7 times the bill of a Long-term low spender.
2. Review the bills of the 45.4% on fibre optic, the highest share of the four segments, since this segment pays $12.70 more a month than the New high spenders.

### Long-term low spenders

This segment holds 966 of the 5,634 training customers, 17.1% of the base. 46 of them left, a churn rate of 4.8%, against 26.54% across all customers. The average customer here has been with the company 54.5 months and pays $34.80 a month. 53.6% are on a month to month plan. In the held out test set the same segment holds 213 customers and churns at 5.2%. This segment churns least of the four. Long tenure on a small bill is the pattern, and it holds the largest share of customers with dependents, 45.9%, and the smallest share of senior citizens, 7.4%. Its churn rate is 43.9% below the New high spenders.

What the company can do:

1. Spend no retention budget here. Only 46 of 966 customers left, so the segment returns the least per dollar of the four.
2. Treat it as the upgrade list instead. These customers pay $34.80 against $93.80 in the Long-term high spenders, and they have stayed almost as long, 54.5 months against 58.7.

## Which segment to act on first

Size multiplied by churn rate gives the number of churners a segment produces, which is
the order to work through:

| Priority | Segment | Customers | Churn rate | Churners |
|---|---|---|---|---|
| 1 | New high spenders | 1,775 | 48.7% | 864 |
| 2 | New low spenders | 1,381 | 24.8% | 342 |
| 3 | Long-term high spenders | 1,512 | 16.1% | 243 |
| 4 | Long-term low spenders | 966 | 4.8% | 46 |

**New high spenders** come first. They are both the largest segment, at 1,775
customers, and the one with the highest churn rate, at 48.7%. Those two
together produced 864 churners, 57.8% of the
1,495 in the training data. The same segment churns at 50.6% in
the test set, which the model never saw, so the pattern is not an artefact of the rows it
was fitted on.

One qualifier on the order. **Long-term high spenders** produce fewer churners than
**New low spenders**, 243 against 342, but each one
pays $93.80 a month against $32.10. Ranked by
the monthly billing that leaves with them, the two swap places.

## Assigning new customers to a segment

New data must be scaled with the same scaler the training data used, then passed to the
model. Load both files and call `predict`:

```python
import joblib
import pandas as pd

scaler = joblib.load("Data_Preparation/scaler.pkl")
model = joblib.load("Clustering_Analysis/kmeans_model.pkl")

SCALE_COLS = ["tenure", "MonthlyCharges"]

new_customers = pd.read_csv("new_customers.csv")      # encoded, not yet scaled
new_customers[SCALE_COLS] = scaler.transform(new_customers[SCALE_COLS])

clusters = model.predict(new_customers[list(model.feature_names_in_)])
```

The columns must arrive encoded the same way as the training data, which
`Data_Preparation/encoding_map.json` records, and in the order `model.feature_names_in_`
gives. `cluster_profiles.csv` maps each cluster number back to its segment name.

## Limitations

The segments split on tenure and monthly charge. The other eight features barely move
between them. Fibre optic runs from 41.6% to
45.4% across the four segments, and the share on a one or two year
contract runs from 44.3% to 46.4%.

This happens because of how K-Means measures distance. `tenure` and `MonthlyCharges` are
z-scores, so they spread roughly from -2 to 3. The other eight columns hold only 0 or 1.
The two wider columns therefore drive most of the distance between customers, and the
model groups mainly on them.

Two points follow from this. The segment names describe tenure and spending, and nothing
else, so they should not be read as statements about contract type or internet type.
Stage 3 should keep using contract type and internet type as predictors in their own
right, because this clustering does not capture them. The segment in
`cluster_assignments.csv` is one extra feature for that model, not a replacement for the
columns it is built from.

The silhouette score at k = 4 is 0.2031. A score near 0 means the clusters sit close
together rather than in separate groups, which the tenure and charge chart shows. The
segments are useful for sorting customers into action lists. They are not evidence that
four distinct customer types exist in the data.
