
import pandas as pd
import numpy as np
import json
import math
from sklearn.ensemble import RandomForestRegressor
from joblib import dump

def haversine_miles(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1 = np.radians(lat1); lat2 = np.radians(lat2)
    dphi = lat2 - lat1
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlam/2)**2
    km = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a)) * R
    return km * 0.621371

def add_nearest_tesla_features(df, lat_col, lon_col, id_col):
    lats = df[lat_col].values
    lons = df[lon_col].values
    ids  = df[id_col].values
    nearest=[]
    for i in range(len(df)):
        d=haversine_miles(lats[i],lons[i],lats,lons)
        d[i]=np.inf
        nearest.append(d.min())
    df["dist_to_nearest_tesla_miles"]=nearest
    return df

def compute_comp_features(df, comp):
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

def attach_demo(df, demo):
    tlat=demo["latitude"].values
    tlon=demo["longitude"].values
    idx=[]
    for i,row in df.iterrows():
        d=haversine_miles(row["Latitude"],row["Longitude"],tlat,tlon)
        idx.append(int(d.argmin()))
    for col in demo.columns:
        if col not in ["latitude","longitude"]:
            df[col]=[demo.iloc[j][col] for j in idx]
    return df

sales=pd.read_csv("sales_history.csv")
loc=pd.read_csv("locations_master.csv")
comp=pd.read_csv("competitors.csv")
demo=pd.read_csv("demographics_censustracts.csv")

sales["Sales"]=sales["Sales"].astype(str).str.replace(",","").astype(float)
if "Zip" in loc.columns:
    loc["Zip"]=loc["Zip"].astype(str).str.replace(",","")

df=sales.merge(loc,on=["TRT ID","Name"],how="left")
df=add_nearest_tesla_features(df,"Latitude","Longitude","TRT ID")
df=compute_comp_features(df,comp)
df=attach_demo(df,demo)

target="Sales"
exclude=[target,"TRT ID","Name","Address","City","Zip"]
features=[c for c in df.columns if c not in exclude]

X=df[features]
y=df[target]

model=RandomForestRegressor(n_estimators=200,random_state=42,min_samples_leaf=2)
model.fit(X,y)

dump(model,"tesla_sales_model.joblib")
json.dump({"feature_columns":features},open("model_metadata.json","w"),indent=2)

print("Model trained and saved.")
