# WILDA Team 4 - Customer Churn Analysis

ACS Work Integrated Learning, Data Analytics. Stage 2 deliverables for S2W8A2.

## Team

| Member | Role |
|---|---|
| Bhawana K C | Project Manager |
| Rohan Sharm Kharel | Data preparation |
| Ramesh Kunwar | Exploratory analysis and clustering |
| Shradha Malla | Predictive modelling |
| Himal Kunwar | Findings and recommendations |

## Repository structure

### Data_Preparation/
Everything covering the preparation of the dataset for analysis and modelling.

- `preprocessed_dataset.csv` - the cleaned dataset, with missing values handled and categorical variables encoded
- `train_set.csv` and `test_set.csv` - the training and testing split used for model validation
- `scaling_techniques.pdf` - documentation of the scaling method applied, the columns it was applied to, why it suits this data, and the split ratio used
- `data_preparation.ipynb` - the notebook that produced the files above

### Clustering_Analysis/
Everything covering the customer segmentation work.

- `elbow_method_results.pdf` - the elbow method analysis and the resulting optimal number of clusters
- `kmeans_model.pkl` - the trained K-Means clustering model
- `clustering_analysis.ipynb` - the code that determined k, trained the model, and produced the plots
- `cluster_visualisations/` - the cluster plots, each segment labelled in plain English

## Video demonstration

A 10 to 15 minute walkthrough of the data preparation and clustering analysis is submitted separately on Canvas as part of S2W8A2.