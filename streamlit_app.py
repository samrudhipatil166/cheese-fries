import math
import os
import re
import sys
import subprocess

import numpy as np
import pandas as pd
import streamlit as st
from geopy.geocoders import Nominatim
import pydeck as pdk


st.set_page_config(page_title="Sales Forecast Playground", layout="wide")

st.title("Sales Forecast Prediction UI")
st.write(
    "Use this app to run your sales forecast models for any address. "
    "Pick the prediction variant at runtime and see the output below."
)

st.markdown("---")


# -------------------------
# Helper functions
# -------------------------
def script_exists(path: str) -> bool:
    return os.path.exists(path) and path.endswith(".py")


def run_prediction_script(script: str, address: str, year: int):
    cmd = [sys.executable, script, "--address", address, "--year", str(year)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout, result.stderr, result.returncode, " ".join(cmd)


@st.cache_data(show_spinner=False)
def load_locations():
    if not os.path.exists("locations_master.csv"):
        return None
    df = pd.read_csv("locations_master.csv")
    df = df.dropna(subset=["Latitude", "Longitude"])
    return df


@st.cache_data(show_spinner=False)
def load_competitors():
    if not os.path.exists("competitors.csv"):
        return None
    df = pd.read_csv("competitors.csv")
    df = df.dropna(subset=["latitude", "longitude"])
    return df


@st.cache_data(show_spinner=False)
def load_sales_history():
    if not os.path.exists("sales_history.csv"):
        return None
    df = pd.read_csv("sales_history.csv")
    if "Sales" not in df.columns:
        return None
    df["Sales"] = df["Sales"].astype(str).str.replace(",", "").astype(float)
    return df


@st.cache_data(show_spinner=False)
def load_demographics():
    if not os.path.exists("demographics_censustracts.csv"):
        return None
    df = pd.read_csv("demographics_censustracts.csv")
    df = df.dropna(subset=["latitude", "longitude"])
    return df


@st.cache_resource(show_spinner=False)
def get_geocoder():
    return Nominatim(user_agent="sales_forecast_app")


def geocode_address(address: str):
    geo = get_geocoder()
    try:
        loc = geo.geocode(address, timeout=5)
        if loc is None:
            return None, None
        return float(loc.latitude), float(loc.longitude)
    except Exception:
        return None, None


def build_radius_polygon(lat: float, lon: float, radius_miles: float = 5.0, n_points: int = 60):
    """Approximate a radius (in miles) around a point as a polygon."""
    R = 3958.8  # Earth radius in miles
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    ang_dist = radius_miles / R

    coords = []
    for i in range(n_points + 1):
        bearing = 2 * math.pi * i / n_points
        lat2 = math.asin(
            math.sin(lat_rad) * math.cos(ang_dist)
            + math.cos(lat_rad) * math.sin(ang_dist) * math.cos(bearing)
        )
        lon2 = lon_rad + math.atan2(
            math.sin(bearing) * math.sin(ang_dist) * math.cos(lat_rad),
            math.cos(ang_dist) - math.sin(lat_rad) * math.sin(lat2),
        )
        coords.append([math.degrees(lon2), math.degrees(lat2)])
    return coords


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in miles between two lat/lon points."""
    R = 3958.8
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def count_within_radius(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    center_lat: float,
    center_lon: float,
    radius_miles: float,
):
    """Return (count within radius, (nearest_distance, nearest_name_or_empty))."""
    count = 0
    nearest = None
    for row in df.itertuples():
        lat = getattr(row, lat_col)
        lon = getattr(row, lon_col)
        dist = haversine_miles(center_lat, center_lon, lat, lon)
        if dist <= radius_miles:
            count += 1
        if nearest is None or dist < nearest[0]:
            name = getattr(row, "Name", "") if hasattr(row, "Name") else ""
            nearest = (dist, name)
    return count, nearest


# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Prediction Variants")

prediction_options = {
    "Baseline only": "predict_baseline.py",
    "Competition model": "predict_competition.py",
    "Demographics model": "predict_demographics.py",
    "Full-stack model": "predict_fullstack.py",
}

variant_descriptions = {
    "Baseline only": "Uses only latitude/longitude/year; ignores competitors and demographics.",
    "Competition model": "Adds competitor distance and count features on top of baseline.",
    "Demographics model": "Adds nearest census-tract demographics on top of baseline.",
    "Full-stack model": "Combines both competition and demographic features with the baseline.",
}

available_options = {
    label: script for label, script in prediction_options.items() if script_exists(script)
}

if not available_options:
    st.sidebar.error("No prediction scripts found.")
else:
    st.sidebar.write("Available models:")
    for label in available_options.keys():
        desc = variant_descriptions.get(label, "")
        st.sidebar.markdown(f"- **{label}**  \n  {desc}")


# -------------------------
# Inputs & forecast (side-by-side scenarios)
# -------------------------
st.markdown("### 1. Enter address")
address = st.text_input("Address", placeholder="e.g. 500 E St Elmo Rd, Austin, TX")

col_cfg1, col_cfg2 = st.columns(2)
with col_cfg1:
    st.markdown("#### Scenario 1")
    location_type_1 = st.selectbox(
        "Location type (Scenario 1)",
        ["Tesla Center", "Popup"],
        index=0,
        help="Popup is more temporary; Tesla Center is a permanent store.",
        key="loc_type_1",
    )
    models_1 = st.multiselect(
        "Models for Scenario 1",
        options=list(available_options.keys()),
        default=list(available_options.keys())[:1],
        key="models_1",
    )

with col_cfg2:
    st.markdown("#### Scenario 2")
    location_type_2 = st.selectbox(
        "Location type (Scenario 2)",
        ["Tesla Center", "Popup"],
        index=1,
        help="Popup is more temporary; Tesla Center is a permanent store.",
        key="loc_type_2",
    )
    models_2 = st.multiselect(
        "Models for Scenario 2",
        options=list(available_options.keys()),
        default=list(available_options.keys())[1:2],
        key="models_2",
    )

growth_pct = st.slider(
    "Assumed annual growth rate (%)",
    min_value=-10.0,
    max_value=20.0,
    value=3.0,
    step=0.5,
)
growth_rate = growth_pct / 100.0

can_run = bool(address.strip()) and (len(models_1) > 0 or len(models_2) > 0)

st.markdown("### 2. Run 2025-2030 forecast (compare scenarios)")
run_clicked = st.button("Run 5-year sales forecast", disabled=not can_run)

if run_clicked:
    years = list(range(2025, 2031))  # 2025-2030 inclusive
    results = []

    scenarios = [
        ("Scenario 1", location_type_1, models_1),
        ("Scenario 2", location_type_2, models_2),
    ]

    for scenario_name, loc_type, model_list in scenarios:
        for model_label in model_list:
            script = available_options[model_label]

            for yr in years:
                stdout, stderr, code, _ = run_prediction_script(
                    script, address.strip(), yr
                )
                results.append(
                    {
                        "Scenario": scenario_name,
                        "Model": model_label,
                        "Year": yr,
                        "Location type": loc_type,
                        "Return code": code,
                        "Stdout": stdout.strip() if stdout else "",
                        "Stderr": stderr.strip() if stderr else "",
                    }
                )

    st.write("#### Forecast results (2025-2030)")
    results_df = pd.DataFrame(results)

    # Extract numeric prediction from stdout
    def extract_pred(s: str):
        if not s:
            return None
        m = re.search(r"([\d,]+(?:\.\d+)?)", s)
        if not m:
            return None
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            return None

    results_df["Base prediction"] = results_df["Stdout"].apply(extract_pred)

    # Use the first year (2025) as the baseline for growth adjustment,
    # computed separately per (scenario, model) so you can compare trajectories.
    base_year = years[0]

    def adjust_row(row, base_value):
        if base_value is None or pd.isna(row["Base prediction"]):
            return None
        years_ahead = row["Year"] - base_year
        return base_value * ((1.0 + growth_rate) ** years_ahead)

    adjusted = []
    for (scenario_name, model_label), group in results_df.groupby(["Scenario", "Model"]):
        base_rows = group[group["Year"] == base_year]["Base prediction"].dropna()
        base_value = base_rows.iloc[0] if not base_rows.empty else None
        for idx, row in group.iterrows():
            adjusted.append((idx, adjust_row(row, base_value)))

    adjusted_dict = {idx: val for idx, val in adjusted}
    results_df["Adjusted prediction (growth)"] = results_df.index.map(adjusted_dict.get)

    # Long-form table plus side-by-side comparison per scenario
    st.write("##### Detailed results")
    st.dataframe(results_df, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.write("##### Scenario 1: model comparison")
        df1 = results_df[results_df["Scenario"] == "Scenario 1"]
        if not df1.empty:
            base_pivot_1 = df1.pivot_table(
                index="Year", columns="Model", values="Base prediction"
            )
            adj_pivot_1 = df1.pivot_table(
                index="Year", columns="Model", values="Adjusted prediction (growth)"
            )
            st.write("Base predictions")
            st.dataframe(base_pivot_1)
            st.write("Growth-adjusted predictions")
            st.dataframe(adj_pivot_1)
        else:
            st.write("No models selected for Scenario 1.")

    with col2:
        st.write("##### Scenario 2: model comparison")
        df2 = results_df[results_df["Scenario"] == "Scenario 2"]
        if not df2.empty:
            base_pivot_2 = df2.pivot_table(
                index="Year", columns="Model", values="Base prediction"
            )
            adj_pivot_2 = df2.pivot_table(
                index="Year", columns="Model", values="Adjusted prediction (growth)"
            )
            st.write("Base predictions")
            st.dataframe(base_pivot_2)
            st.write("Growth-adjusted predictions")
            st.dataframe(adj_pivot_2)
        else:
            st.write("No models selected for Scenario 2.")

    any_error = any(row["Return code"] != 0 for row in results)
    if any_error:
        st.warning(
            "One or more years returned a non-zero exit code. "
            "Check the 'Return code' and 'Stderr' columns above for details."
        )
    else:
        st.success("Forecast completed successfully for all years 2025-2030.")

    # Suggestions & cannibalization risk (shared cannibalization/competition, per-scenario format)
    st.markdown("#### Suggestions & cannibalization risk")

    lat, lon = geocode_address(address.strip())
    if lat is None or lon is None:
        st.write("Could not compute suggestions; address could not be geocoded.")
    else:
        # Cannibalization risk from existing Tesla locations
        loc_df = load_locations()
        risk_text = None
        if loc_df is not None:
            cnt, nearest = count_within_radius(
                loc_df,
                "Latitude",
                "Longitude",
                lat,
                lon,
                radius_miles=5.0,
            )
            if cnt == 0:
                risk = "Low"
            elif cnt <= 2:
                risk = "Medium"
            else:
                risk = "High"

            risk_text = (
                f"Cannibalization risk (existing Tesla locations within 5 miles): "
                f"**{risk}** ({cnt} existing site(s) nearby)."
            )
        else:
            risk_text = (
                "Cannibalization risk: unknown (no existing locations data loaded)."
            )

        # Competitor intensity around this site
        comp_df = load_competitors()
        if comp_df is not None:
            comp_cnt, _ = count_within_radius(
                comp_df,
                "latitude",
                "longitude",
                lat,
                lon,
                radius_miles=5.0,
            )
            comp_text = f"Competitor intensity: {comp_cnt} competitors within 5 miles."
        else:
            comp_text = "Competitor intensity: unknown (no competitor data loaded)."

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.write("**Scenario 1**")
            st.markdown(f"- {risk_text}")
            st.markdown(f"- {comp_text}")
            if location_type_1 == "Popup":
                st.markdown(
                    "- Popup format is ideal to test demand or handle seasonal peaks "
                    "with lower long-term commitment, especially if cannibalization "
                    "risk is medium or high."
                )
            else:
                st.markdown(
                    "- Tesla Center format is better for long-term demand and service. "
                    "It is most attractive when cannibalization risk is low and "
                    "competitor intensity is moderate."
                )

        with col_s2:
            st.write("**Scenario 2**")
            st.markdown(f"- {risk_text}")
            st.markdown(f"- {comp_text}")
            if location_type_2 == "Popup":
                st.markdown(
                    "- Popup format is ideal to test demand or handle seasonal peaks "
                    "with lower long-term commitment, especially if cannibalization "
                    "risk is medium or high."
                )
            else:
                st.markdown(
                    "- Tesla Center format is better for long-term demand and service. "
                    "It is most attractive when cannibalization risk is low and "
                    "competitor intensity is moderate."
                )


# -------------------------
# Map: existing vs entered location
# -------------------------
st.markdown("---")
st.markdown("### 3. Map: existing locations vs entered address")

locations_df = load_locations()
competitors_df = load_competitors()

if locations_df is None and competitors_df is None:
    st.info(
        "No `locations_master.csv` or `competitors.csv` files found, so the map of "
        "existing locations cannot be displayed."
    )
else:
    entered_lat = entered_lon = None
    if address.strip():
        entered_lat, entered_lon = geocode_address(address.strip())

    if entered_lat is None or entered_lon is None:
        st.warning(
            "Enter a valid address (and run a prediction) to see your selected "
            "location alongside existing locations on the map."
        )
    else:
        layers = []
        map_frames = []

        existing = None
        if locations_df is not None:
            existing = locations_df.rename(
                columns={"Latitude": "lat", "Longitude": "lon"}
            )
            existing["type"] = "Existing location"
            existing["brand"] = "Tesla"
            map_frames.append(existing[["lat", "lon"]])

            existing_layer = pdk.Layer(
                "ScatterplotLayer",
                data=existing,
                get_position="[lon, lat]",
                get_radius=250,
                get_fill_color=[0, 102, 204, 160],  # blue
                pickable=True,
            )
            layers.append(existing_layer)

        competitors = None
        if competitors_df is not None:
            competitors = competitors_df.rename(
                columns={"latitude": "lat", "longitude": "lon", "name": "Name"}
            )
            competitors["type"] = "Competitor"
            if "brand" in competitors_df.columns:
                competitors["brand"] = competitors_df["brand"]
            else:
                competitors["brand"] = ""
            map_frames.append(competitors[["lat", "lon"]])

            competitors_layer = pdk.Layer(
                "ScatterplotLayer",
                data=competitors,
                get_position="[lon, lat]",
                get_radius=250,
                get_fill_color=[255, 255, 0, 255],  # yellow
                pickable=True,
            )
            layers.append(competitors_layer)

        entered = pd.DataFrame(
            [
                {
                    "lat": entered_lat,
                    "lon": entered_lon,
                    "Name": "Entered address",
                    "type": "Selected location",
                    "brand": "",
                }
            ]
        )
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=entered,
                get_position="[lon, lat]",
                get_radius=400,
                get_fill_color=[255, 0, 0, 220],  # red
                pickable=True,
            )
        )
        map_frames.append(entered[["lat", "lon"]])

        map_df = pd.concat(map_frames, ignore_index=True)
        midpoint = (map_df["lat"].mean(), map_df["lon"].mean())

        radius_polygon = build_radius_polygon(entered_lat, entered_lon, radius_miles=5.0)
        radius_layer = pdk.Layer(
            "PolygonLayer",
            data=[{"polygon": radius_polygon, "Name": "5-mile radius"}],
            get_polygon="polygon",
            get_fill_color=[255, 255, 0, 40],
            get_line_color=[255, 255, 0],
            line_width_min_pixels=2,
            pickable=False,
        )
        layers.append(radius_layer)

        deck = pdk.Deck(
            layers=layers,
            initial_view_state=pdk.ViewState(
                latitude=midpoint[0],
                longitude=midpoint[1],
                zoom=10,
                pitch=0,
            ),
            tooltip={"text": "{type}\n{Name}\n{brand}"},
        )

        st.pydeck_chart(deck)
