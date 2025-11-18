import argparse
import sys
import json
import joblib
import pandas as pd
from geopy.geocoders import Nominatim

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", required=True)
    parser.add_argument("--year", required=True, type=int)
    args = parser.parse_args()

    geo = Nominatim(user_agent="tesla_baseline")
    try:
        loc = geo.geocode(args.address, timeout=5)
    except Exception:
        loc = None
    if loc is None:
        print("Could not geocode address; please enter a more complete address.", file=sys.stderr)
        sys.exit(1)
    lat, lon = loc.latitude, loc.longitude

    model = joblib.load("tesla_sales_model.joblib")

    X = pd.DataFrame([{
        "Latitude": lat,
        "Longitude": lon,
        "Year": args.year
    }])

    # Align features with training metadata
    meta = json.load(open("model_metadata.json"))
    feats = meta["feature_columns"]
    for col in feats:
        if col not in X.columns:
            X[col] = 0
    X = X[feats]

    pred = model.predict(X)[0]
    print(f"Predicted annual sales (Baseline): {pred:,.0f}")

if __name__ == "__main__":
    main()
