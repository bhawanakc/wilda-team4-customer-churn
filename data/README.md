# Raw data

`Dataset_ATS_v2.csv` is the project dataset provided by ACS on Canvas
(page S1W1P2 - Project Scenario, Roles and Dataset), kept unmodified and under
its original filename so the source stays traceable.

- 7,043 customers, 10 columns
- Columns: gender, SeniorCitizen, Dependents, tenure, PhoneService,
  MultipleLines, InternetService, Contract, MonthlyCharges, Churn
- Target: `Churn` (Yes / No)

Everything in `Data_Preparation/` and `Clustering_Analysis/` is generated from
this file by running `Data_Preparation/data_preparation.ipynb` first.
