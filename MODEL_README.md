# Model Workflow Guide (for interns)

This document explains how the sales forecast model works end‑to‑end: data → features → training → prediction.

## 1. Data sources

Training data comes from the CSV files in the project root:

- `sales_history.csv`: historical sales per location/year (target column `Sales`).
- `locations_master.csv`: Tesla locations with `TRT ID`, `Name`, `Address`, `Latitude`, `Longitude`, etc.
- `competitors.csv`: competitor locations with `latitude`, `longitude`, and optionally `name`/`brand`.
- `demographics_censustracts.csv`: census/demographic features per tract (`latitude`, `longitude`, `population`, `median_income`, `ev_adoption_rate`, etc.).

These files are combined in `train_model.py` to build the training table.

## 2. Feature engineering at training time (`train_model.py`)

High‑level steps in `train_model.py`:

1. **Load and clean data**
   - Read all four CSVs with `pandas.read_csv`.
   - Convert `Sales` to numeric by stripping commas and casting to `float`.
   - Optionally clean `Zip` in `locations_master.csv` to be a string without commas.

2. **Join sales with locations**
   - `sales` is merged with `locations_master` on `["TRT ID", "Name"]`.
   - Each row now has: year, sales, plus the store’s lat/long and address info.

3. **Tesla network feature (`add_nearest_tesla_features`)**
   - For each Tesla store, compute distances to all other Tesla stores using a haversine distance function.
   - Ignore distance to itself, then take the minimum.
   - Add feature: `dist_to_nearest_tesla_miles`.

4. **Competition features (`compute_comp_features`)**
   - For each Tesla store, compute distances to all competitors in `competitors.csv`.
   - Create two features:
     - `dist_to_nearest_competitor_miles`: minimum distance to any competitor.
     - `num_competitors_5miles`: count of competitors within 5 miles.

5. **Demographic features (`attach_demo`)**
   - For each Tesla store, compute distance to every census tract centroid in `demographics_censustracts.csv`.
   - Find the nearest tract and copy all non‑lat/lon columns (e.g. `population`, `median_income`, `ev_adoption_rate`, `population_density`, `median_age`, etc.) into that row.

After these steps, each training row contains:

- Time: `Year`
- Location: `Latitude`, `Longitude`
- Tesla network: `dist_to_nearest_tesla_miles`
- Competition: `dist_to_nearest_competitor_miles`, `num_competitors_5miles`
- Demographics: multiple columns from `demographics_censustracts.csv`
- Target: `Sales`

## 3. Feature selection strategy

In `train_model.py`, features are chosen using a simple “exclude list”:

- Target column: `Sales`.
- Columns excluded from features: `["Sales", "TRT ID", "Name", "Address", "City", "Zip"]`.
- All remaining columns are treated as features:
  - `features = [c for c in df.columns if c not in exclude]`

The exact list of feature names is written to `model_metadata.json` under the key `feature_columns`. This is important because prediction scripts use the same ordered list to build their input DataFrames.

## 4. Model training

Also in `train_model.py`:

- Model type: `RandomForestRegressor` from scikit‑learn.
  - `n_estimators=200` trees.
  - `min_samples_leaf=2` to reduce overfitting on tiny leaves.
  - `random_state=42` for reproducibility.
- Training pipeline:
  - `X = df[features]`
  - `y = df["Sales"]`
  - `model.fit(X, y)`
- Saved artifacts:
  - `tesla_sales_model.joblib`: the trained model.
  - `model_metadata.json`: JSON file with the feature column list.

At the moment, there is no explicit train/validation split; the model is fitted on all rows in the dataset. To add proper evaluation, you would introduce a split (e.g. by time or random) and compute metrics like MAE or RMSE on the hold‑out set.

## 5. Prediction‑time workflow (CLI scripts)

All `predict_*.py` scripts follow a common pattern:

1. Parse `--address` and `--year` from the command line.
2. Geocode the address using `geopy.Nominatim` to get latitude and longitude.
3. Build a one‑row `pandas.DataFrame` with whatever subset of features that script is responsible for.
4. Load model and metadata:
   - `tesla_sales_model.joblib`
   - `model_metadata.json`
5. Align features with training:
   - For each column in `feature_columns` that is missing in the DataFrame, add it and fill with 0.
   - Reorder columns to exactly match `feature_columns`.
6. Call `model.predict(X)` and print the numeric sales prediction.

The differences between scripts are in step 3 – which features they compute:

- `predict_baseline.py`
  - Builds features with only `Latitude`, `Longitude`, and `Year`.
  - After alignment, all competition and demographic features are set to 0.
  - Interpreted as a “location + time only” baseline.

- `predict_competition.py`
  - Computes competition features for the candidate address:
    - Distance to each competitor in `competitors.csv`.
    - Adds “distance to nearest competitor” and “competitors within 5 miles” to the row.
  - Demonstrates the incremental value of competition features.

- `predict_demographics.py`
  - Finds the nearest census tract to the candidate address and copies its key demographic values (e.g. `population`, `median_income`, `ev_adoption_rate`).
  - Focuses on the impact of demographics alone.

- `predict_fullstack.py`
  - Computes both competition and demographic features for the candidate site.
  - Represents the “all features” view available at prediction time.

- `predict_for_address.py`
  - Reconstructs features in a way that closely mirrors training:
    - Creates a temporary one‑row DataFrame for the candidate address.
    - Applies competition and demographic feature functions similar to those in `train_model.py`.
  - Uses the same `feature_columns` from `model_metadata.json` to align the final feature vector.

All scripts rely on the metadata‑driven alignment step to stay compatible even if the feature set changes in future training runs.

## 6. Streamlit app integration

`streamlit_app.py` is the UI layer that calls the prediction scripts and visualizes results:

- The sidebar lets you choose one or more model variants (baseline, competition, demographics, full‑stack, unified).
- For a given input address and forecast year range:
  - The app calls the selected `predict_*.py` scripts via `subprocess.run`.
  - It parses each script’s stdout to extract the numeric prediction.
  - It builds tables comparing scenarios and years, and applies a configurable growth rate adjustment.
- The app also reuses geocoding and geographic helpers to:
  - Draw maps of existing Tesla locations, competitors, and the entered address.
  - Compute simple cannibalization risk and competitor density around the selected site.

## 7. How to safely extend the model

If you are modifying the model as an intern, follow this pattern:

1. **Change feature engineering** in `train_model.py` (e.g. add new demographics, new distance metrics).
2. **Retrain** the model by running `python train_model.py` to regenerate `tesla_sales_model.joblib` and `model_metadata.json`.
3. **Update prediction scripts** to compute any new features at prediction time (or accept them as 0 if they are training‑only).
4. **Check alignment**:
   - Make sure columns in `model_metadata.json["feature_columns"]` match what you expect to compute in `predict_*.py`.
5. **Test end‑to‑end** using the Streamlit app with a few addresses to confirm predictions are produced without errors and are numerically sensible.

This workflow keeps training, prediction, and the UI in sync as the feature set evolves.

