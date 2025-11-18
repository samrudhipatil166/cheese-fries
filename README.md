# Sales Forecast Playground

Simple Streamlit UI and scripts for training and running sales-forecast models for Tesla locations (or similar retail locations).

## 1. Setup

1. Install Python (3.9+ recommended).
2. In this folder (`Forecast sales`), create/activate a virtual env (optional but recommended), for example on Windows:
   - `py -m venv venv`
   - `venv\Scripts\activate`
3. Install all required packages:
   - `py -m pip install -r requirements.txt`

## 2. Data files

Keep these CSVs in the same folder as the scripts:

- `sales_history.csv` – historical sales by location/year.
- `locations_master.csv` – existing Tesla locations (with `Latitude`/`Longitude`).
- `competitors.csv` – competitor locations.
- `demographics_censustracts.csv` – census/demographic features.

## 3. Train the model

Run once (or whenever you change data/logic) to create the model file and metadata:

- `py train_model.py`

This produces:

- `tesla_sales_model.joblib` – trained RandomForest model.
- `model_metadata.json` – list of feature columns used at prediction time.

## 4. Run the Streamlit app

From the same folder and environment:

- `streamlit run streamlit_app.py`

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## 5. Using the UI

- **Choose model variant** (sidebar):
  - Unified model (metadata-driven) → `predict_for_address.py`
  - Baseline / Competition / Demographics / Full-stack → other `predict_*.py` scripts.
- **Enter inputs**:
  - Address (free text, geocoded via `geopy`).
  - Target year (numeric).
- **Run prediction**:
  - Click **Run sales forecast**.
  - The app shows the exact command used, stdout (prediction text), and any errors/warnings.

## 6. Map visualization

The app uses `locations_master.csv` to show existing locations on a map and compares them to the entered address:

- Existing locations → blue points.
- Entered address → red point.

If geocoding fails or `locations_master.csv` is missing, the app shows a message instead of the map.

## 7. Main scripts

- `train_model.py` – train and save the unified model + metadata.
- `predict_for_address.py` – unified prediction using all features (competition + demographics).
- `predict_baseline.py` – baseline model using only latitude/longitude/year.
- `predict_competition.py` – competition-only model.
- `predict_demographics.py` – demographics-only model.
- `predict_fullstack.py` – competition + demographics model.
- `streamlit_app.py` – Streamlit UI (model selection, prediction, and map).

