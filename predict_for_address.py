
import argparse, json, sys, numpy as np, pandas as pd
from joblib import load
from geopy.geocoders import Nominatim

def haversine_miles(lat1, lon1, lat2, lon2):
    R=6371.0
    lat1=np.radians(lat1); lat2=np.radians(lat2)
    dphi=lat2-lat1
    dl=np.radians(lon2-lon1)
    a=np.sin(dphi/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dl/2)**2
    km=2*np.arctan2(np.sqrt(a),np.sqrt(1-a))*R
    return km*0.621371

def comp_features(df, comp):
    clat=comp["latitude"].values
    clon=comp["longitude"].values
    nearest=[]; num5=[]
    for i,row in df.iterrows():
        d=haversine_miles(row["Latitude"],row["Longitude"],clat,clon)
        nearest.append(d.min())
        num5.append((d<=5).sum())
    df["dist_to_nearest_competitor_miles"]=nearest
    df["num_competitors_5miles"]=num5
    return df

def demo_features(df, demo):
    tlat=demo["latitude"].values
    tlon=demo["longitude"].values
    idx=[]
    for i,row in df.iterrows():
        d=haversine_miles(row["Latitude"],row["Longitude"],tlat,tlon)
        idx.append(d.argmin())
    for col in demo.columns:
        if col not in ["latitude","longitude"]:
            df[col]=[demo.iloc[j][col] for j in idx]
    return df

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--address",required=True)
    parser.add_argument("--year",type=int,required=True)
    args=parser.parse_args()

    geo = Nominatim(user_agent="tesla_sales")
    try:
        loc = geo.geocode(args.address, timeout=5)
    except Exception:
        loc = None
    if loc is None:
        print("Could not geocode address; please enter a more complete address.", file=sys.stderr)
        sys.exit(1)
    lat, lon = loc.latitude, loc.longitude

    model=load("tesla_sales_model.joblib")
    meta=json.load(open("model_metadata.json"))
    feats=meta["feature_columns"]

    comp=pd.read_csv("competitors.csv")
    demo=pd.read_csv("demographics_censustracts.csv")

    df=pd.DataFrame([{"Year":args.year,"Latitude":lat,"Longitude":lon}])
    df=comp_features(df,comp)
    df=demo_features(df,demo)

    for col in feats:
        if col not in df:
            df[col]=0

    df=df[feats]
    pred=model.predict(df)[0]
    print(f"Predicted annual sales: {pred:,.0f}")

if __name__=="__main__":
    main()
