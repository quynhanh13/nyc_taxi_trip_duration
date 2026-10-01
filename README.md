# NYC Taxi Trip Duration Prediction

>An end-to-end data science project using Kaggle's **NYC Taxi Trip Duration** dataset (2016 NYC TLC data). The goal is to predict taxi trip duration in seconds from temporal information, pickup/dropoff coordinates, and other trip attributes.

## Objectives

- Explore the data-generating process and factors associated with trip duration.
- Analyze spatial and temporal patterns, outliers, and non-linear relationships.
- Engineer features from coordinates and pickup times; evaluate K-Means for trip grouping.
- Compare Linear Regression, Ridge, Random Forest, and an MLP neural network.
- Audit data leakage, evaluate prediction errors, and recommend a suitable model.

## Dataset

The original dataset contains approximately 1.46 million trips. Each row in `train.csv` includes the following key fields:

| Field | Description |
|---|---|
| `id` | Trip identifier |
| `vendor_id` | Provider identifier |
| `pickup_datetime` | Trip start time |
| `dropoff_datetime` | Trip end time |
| `passenger_count` | Number of passengers |
| `pickup_longitude`, `pickup_latitude` | Pickup coordinates |
| `dropoff_longitude`, `dropoff_latitude` | Dropoff coordinates |
| `store_and_fwd_flag` | Whether the trip record was stored in the vehicle before transmission |
| `trip_duration` | Trip duration in seconds; the target variable |

The notebook reads `train.csv` from the current working directory and creates its own 80/20 training/test split. Kaggle's `test.csv` is present in the project folder but is not used by the current workflow.

Download the data from [Kaggle: New York City Taxi Trip Duration](https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data) if the CSV files are not available. Place `train.csv` beside the notebook before running it.

## Analysis

The notebook [nyc_taxi_trip_duration.ipynb](nyc_taxi_trip_duration.ipynb) covers:

1. Data-generating process (DGP) hypotheses and limitations of the observed data.
2. Exploratory data analysis (EDA), including duration distributions, temporal patterns, distance, and geography.
3. Data cleaning and feature engineering: Haversine and Manhattan distances, pickup hour/day, time of day, pickup/dropoff boroughs, and borough-to-borough routes.
4. K-Means clustering on distance and temporal features, followed by an interpretation of the cluster labels.
5. Training and comparison of Linear Regression, Ridge Regression, Random Forest, and MLP models, with a `log1p` transformation of the target.
6. Data-leakage auditing, cross-validation, hyperparameter tuning, test-set evaluation, and residual analysis.

## Setup and Usage

Python 3.10 or later is recommended. From the project directory, create an environment and install the libraries used by the notebook:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install pandas numpy matplotlib seaborn scipy scikit-learn statsmodels jupyter
jupyter lab
```

Open `DataScience_FinalProject_0513_2.ipynb` in JupyterLab or VS Code, select the `.venv` kernel, and run the cells sequentially from top to bottom. Cross-validation and grid-search cells may take several minutes because of the dataset size.

## Key Results

According to the results documented in the notebook:

- Non-linear models perform substantially better than linear models on this dataset.
- The tuned Random Forest achieves approximately **302 seconds RMSE**, **206 seconds MAE**, and **0.72 R²** on the test set. The MLP performs similarly.
- Linear models achieve approximately **0.42–0.43 R²** on the test set.
- Adding K-Means cluster labels provides almost no aggregate score improvement; clustering is primarily useful for interpreting trip structure.
- Residual analysis shows that the model tends to underpredict longer trips, and that error varies with predicted duration.

> These figures are taken from results recorded in the notebook; they were not re-run while creating this README.

## Leakage Controls and Limitations

- The train/test split is created before analysis and preprocessing; transformers are fitted on training data through pipelines.
- `dropoff_datetime` and `speed_kmh` are excluded because they are derived directly or indirectly from the target.
- Cleaning thresholds and clustering models are determined from the training set; the test set is only transformed or predicted.
- The current split is random. A temporal split is recommended to estimate performance when predicting future trips.
- The data covers 2016 only and does not include real-time traffic, weather, or road-network features; these are important limitations on generalization.

## Project Structure

```text
NYC-Taxi/
├── DataScience_FinalProject_0513_2.ipynb
├── README.md
├── train.csv
└── test.csv
```