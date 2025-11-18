import argparse, sys, json, joblib, pandas as pd, numpy as np
from geopy.geocoders import Nominatim

def haversine_miles(lat1, lon1, lat2, lon2):
    R=6371.0
    lat1=np.radians(lat1); lat2=np.radians(lat2)
    dphi=lat2-lat1
    dl=np.radians(lon2-lon1)
    a=np.sin(dphi/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dl/2)**2
    km=2*np.arctan2(np.sqrt(a), np.sqrt(1-a))*R
    return km*0.621371

def competitor_features(lat, lon):
    comp=pd.read_csv("competitors.csv")
    d=haversine_miles(lat, lon, comp["latitude"].values, comp["longitude"].values)
    return {
        "dist_to_nearest_comp_miles": float(d.min()),
        "num_competitors_5miles": int((d<=5).sum())
    }

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--address",required=True)
    parser.add_argument("--year",required=True,type=int)
    args=parser.parse_args()

    geo = Nominatim(user_agent="tesla_competition")
    try:
        loc = geo.geocode(args.address, timeout=5)
    except Exception:
        loc = None
    if loc is None:
        print("Could not geocode address; please enter a more complete address.", file=sys.stderr)
        sys.exit(1)
    lat, lon = loc.latitude, loc.longitude

    feats = competitor_features(lat, lon)

    X = pd.DataFrame([{"Latitude": lat, "Longitude": lon, "Year": args.year, **feats}])

    model = joblib.load("tesla_sales_model.joblib")

    # Align features with training metadata
    meta = json.load(open("model_metadata.json"))
    feat_cols = meta["feature_columns"]
    for col in feat_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[feat_cols]

    pred = model.predict(X)[0]

    print(f"Predicted annual sales (Competition Model): {pred:,.0f}")

if __name__=="__main__":
    main()
