# WILDA Team 4, customer churn analysis

ACS Work Integrated Learning, Data Analytics. Stage 2 deliverables for S2W8A2.

## Team

| Member | Role |
|---|---|
| Bhawana K C | Project Manager |
| Rohan Sharma Kharel | Data preparation |
| Ramesh Kunwar | Exploratory analysis and clustering |
| Shradha Malla | Predictive modelling |
| Himal Kunwar | Findings and recommendations |

## Running the analysis

```bash
pip install -r requirements.txt
```

The raw dataset is at `data/Dataset_ATS_v2.csv`. Run the two notebooks in this order,
because the clustering notebook reads the training set that the preparation notebook
writes:

1. `Data_Preparation/data_preparation.ipynb`
2. `Clustering_Analysis/clustering_analysis.ipynb`

The preparation notebook runs `Data_Preparation/data_preparation.py`. You can also run
that script directly for step 1:

```bash
python Data_Preparation/data_preparation.py
```

## Repository structure

### data/

The raw dataset from Canvas, unchanged. Every other file in the repository is built
from it.

### Data_Preparation/

The files that prepare the dataset for clustering and modelling.

| File | Contents |
|---|---|
| `data_preparation.py` | The pipeline that writes the files below. |
| `data_preparation.ipynb` | A notebook that runs the same script and explains each step. Both write identical files. |
| `preprocessed_dataset.csv` | The cleaned dataset, with missing values handled and categorical variables encoded. |
| `train_set.csv`, `test_set.csv` | The training and test sets used for model validation. |
| `README.md` | The size and composition of the two sets, a column dictionary, and the reason for each preparation decision. |
| `scaler.pkl`, `encoding_map.json` | The fitted scaler and the category mappings. Later stages load these to transform data the same way. |
| `preparation_summary.json` | The numbers used in the scaling document. |
| `scaling_techniques.pdf` | The scaling method, the columns it applies to, why it suits this data, and the split ratio. |
| `build_docs.py` | Rebuilds `README.md` and `scaling_techniques.pdf` from `preparation_summary.json`. |
| `figures/` | Histograms of the scaled columns before and after scaling, used in the PDF. |

### Clustering_Analysis/

The files for the customer segmentation work.

| File | Contents |
|---|---|
| `clustering_analysis.ipynb` | The code that chooses k, trains the model and draws the plots. |
| `cluster_selection_metrics.csv` | Inertia and silhouette score for every k tested. |
| `elbow_method_results.pdf` | The elbow method analysis and the chosen number of clusters. |
| `kmeans_model.pkl` | The trained K-Means model. |
| `cluster_profiles.csv` | The average of each feature per segment, which the segment names are based on. |
| `cluster_assignments.csv` | Each customer's segment, for the Stage 3 model to use. |
| `cluster_visualisations/` | The cluster plots, with each segment named in plain English. |

## Method notes

Decisions to explain in the video:

- **We fit the scaler on the training set only** and then apply it to the test set. Fitting on the full dataset would let test data influence training and make the results look better than they are.
- **We check the choice of k with two measures.** Inertia always falls as k rises, so we read the elbow alongside the silhouette score.
- **We name each cluster from its profile.** `cluster_profiles.csv` shows the numbers behind every name.

## Video demonstration

We submit a 10 to 15 minute video of the data preparation and clustering analysis
separately on Canvas, as part of S2W8A2.
