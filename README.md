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

## Running the analysis

```bash
pip install -r requirements.txt
```

Place the raw dataset at `data/Dataset_ATS_v2.csv`, then run the notebooks in
order — the clustering notebook reads the training set the preparation notebook
writes:

1. `Data_Preparation/data_preparation.ipynb`
2. `Clustering_Analysis/clustering_analysis.ipynb`

The preparation notebook is a walkthrough of `Data_Preparation/data_preparation.py`,
so step 1 can equally be run from the command line:

```bash
python Data_Preparation/data_preparation.py
```

## Repository structure

### data/
The raw dataset, unmodified. The single input the whole pipeline depends on.

### Data_Preparation/
Everything covering the preparation of the dataset for analysis and modelling.

- `data_preparation.py` - the pipeline that produces the files below
- `data_preparation.ipynb` - the notebook walkthrough, which runs that same script, so both routes produce identical files
- `preprocessed_dataset.csv` - the cleaned dataset, with missing values handled and categorical variables encoded
- `train_set.csv` and `test_set.csv` - the training and testing split used for model validation
- `README.md` - the size and composition of those two sets, the column dictionary, and the reasoning behind each preparation decision
- `scaler.pkl` and `encoding_map.json` - the fitted scaler and the category mappings, so the clustering notebook reuses the exact same transformation
- `preparation_summary.json` - the figures behind the scaling documentation
- `scaling_techniques.pdf` - documentation of the scaling method applied, the columns it was applied to, why it suits this data, and the split ratio used
- `build_docs.py` - regenerates the two documents above from `preparation_summary.json`
- `figures/` - the before and after scaling histograms used in that document

### Clustering_Analysis/
Everything covering the customer segmentation work.

- `clustering_analysis.ipynb` - the code that determined k, trained the model, and produced the plots
- `cluster_selection_metrics.csv` - inertia and silhouette score for every k tested
- `elbow_method_results.pdf` - the elbow method analysis and the resulting optimal number of clusters
- `kmeans_model.pkl` - the trained K-Means clustering model
- `cluster_profiles.csv` - the per-segment figures the labels are derived from
- `cluster_assignments.csv` - each customer's segment, for the Stage 3 model to use
- `cluster_visualisations/` - the cluster plots, each segment labelled in plain English

## Method notes

Three decisions worth raising in the video walkthrough:

- **The scaler is fitted on the training set only**, then applied to the test set. Fitting on the full dataset leaks test information into training and inflates the results.
- **The optimal k is cross-checked.** Inertia falls forever as k rises, so the elbow is read alongside the silhouette score rather than from the inertia curve alone.
- **Cluster labels are derived from the cluster profiles**, not assigned by eye. `cluster_profiles.csv` is the evidence behind every name.

## Video demonstration

A 10 to 15 minute walkthrough of the data preparation and clustering analysis is
submitted separately on Canvas as part of S2W8A2.
