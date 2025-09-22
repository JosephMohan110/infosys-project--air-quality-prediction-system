# streamlit/milestone3.py
"""
Milestone 3 Streamlit UI for AirAware
Reads data from: results/, results/forecasts/, models/, reports/
Shows: AQI gauge, 7-day forecast table, pollutant time-series, active alerts, model performance.
Designed for cities: Ahmedabad, Bengaluru, Chennai, Delhi, Kolkata, Mumbai
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import math
import io

# ----------------------------
# Configuration
# ----------------------------
ROOT = Path.cwd()
RESULTS_DIR = ROOT / "results"
FORECASTS_SUBDIR = RESULTS_DIR / "forecasts"   # optional per-city forecast CSVs
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# Fixed city list requested
CITIES = ["Ahmedabad", "Bengaluru", "Chennai", "Delhi", "Kolkata", "Mumbai"]

# Pollutants known
POLLUTANTS_COMMON = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]

# Make sure basic dirs exist (don't create optional subfolder)
for d in (RESULTS_DIR, MODELS_DIR, REPORTS_DIR):
    d.mkdir(exist_ok=True)

# ----------------------------
# Utility functions
# ----------------------------
def find_file_by_variants(folder: Path, variants):
    """Return first existing file in folder matching any variant snippet (case-insensitive)."""
    if not folder.exists():
        return None
    for v in variants:
        p = folder / v
        if p.exists():
            return p
    # fallback: search filenames containing the variant
    for f in folder.glob("*.csv"):
        name = f.name.lower()
        for v in variants:
            if v.lower() in name:
                return f
    return None

def safe_read_csv(path: Path, parse_dates=True):
    try:
        if parse_dates:
            return pd.read_csv(path, parse_dates=True, infer_datetime_format=True)
        else:
            return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path)
        except Exception:
            return None

def detect_date_column(df: pd.DataFrame):
    candidates = [c for c in df.columns if c.lower() in ("date","datetime","time","timestamp")]
    if candidates:
        return candidates[0]
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    return None

def detect_value_column(df: pd.DataFrame):
    candidates = [c for c in df.columns if c.lower() in ("forecast","value","prediction","pred","aqi","reading","conc","concentration","mean")]
    if candidates:
        return candidates[0]
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None

def normalize_df_date_and_value(df, date_aliases=None, value_aliases=None):
    """Return df with 'Date' (datetime) and 'Value' numeric if possible."""
    df = df.copy()
    date_col = detect_date_column(df)
    if date_col:
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.rename(columns={date_col:"Date"})
        except Exception:
            pass
    val_col = detect_value_column(df)
    if val_col and val_col != "Value":
        try:
            df = df.rename(columns={val_col:"Value"})
            df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
        except Exception:
            pass
    return df

# ----------------------------
# Cached loader
# ----------------------------
@st.cache_data
def load_data():
    data = {}

    # forecasts_aqi: try several variants
    fa_file = find_file_by_variants(RESULTS_DIR, ["forecasts_aqi.csv", "forecast_aqi.csv", "forecast_aqi", "forecasts_aqi", "forecast-aqi.csv", "forecasts-aqi.csv", "forecast_aqi.csv"])
    if fa_file:
        dfa = safe_read_csv(fa_file)
        if dfa is not None:
            # try standardize Date column name
            date_col = detect_date_column(dfa)
            if date_col:
                try:
                    dfa[date_col] = pd.to_datetime(dfa[date_col], errors="coerce")
                    dfa = dfa.rename(columns={date_col:"Date"})
                except Exception:
                    pass
            data["forecasts_aqi"] = dfa

    # aggregated forecasts.csv
    fagg = find_file_by_variants(RESULTS_DIR, ["forecasts.csv","forecast.csv","forecasts", "forecast"])
    if fagg:
        dff = safe_read_csv(fagg)
        if dff is not None:
            date_col = detect_date_column(dff)
            if date_col:
                try:
                    dff[date_col] = pd.to_datetime(dff[date_col], errors="coerce")
                    dff = dff.rename(columns={date_col:"Date"})
                except Exception:
                    pass
            data["forecasts"] = dff

    # per-city forecasts folder (optional)
    percity = []
    if FORECASTS_SUBDIR.exists():
        for p in sorted(FORECASTS_SUBDIR.glob("*.csv")):
            dfp = safe_read_csv(p)
            if dfp is None: 
                continue
            # normalize
            dfp = normalize_df_date_and_value(dfp)
            # infer city/pollutant from filename patterns: forecast_Ahmedabad_PM2.5.csv or Ahmedabad_PM2.5.csv
            stem = p.stem
            parts = stem.split("_")
            city = None
            pollutant = None
            if len(parts) >= 3 and parts[0].lower().startswith("forecast"):
                city = parts[1]
                pollutant = "_".join(parts[2:])
            elif len(parts) >= 2:
                city = parts[0]
                pollutant = "_".join(parts[1:])
            else:
                # fallback: parent folder name if exists
                city = p.parent.name if p.parent != RESULTS_DIR else "Unknown"
                pollutant = stem
            percity.append({"file":p.name, "path":p, "city":city, "pollutant":pollutant, "df":dfp})
    data["percity_forecasts"] = percity

    # alerts
    alerts_f = find_file_by_variants(RESULTS_DIR, ["alerts.csv","alert.csv","alerts_summary.csv","alert_summary.csv"])
    if alerts_f:
        data["alerts"] = safe_read_csv(alerts_f)

    # model performance
    perf_f = find_file_by_variants(RESULTS_DIR, ["model_performance.csv","model_summary.csv","forecast_metrics_summary.csv","forecast_accuracy_matrics.csv"])
    if perf_f:
        data["performance"] = safe_read_csv(perf_f)

    # best_models
    best_f = RESULTS_DIR / "best_models.csv"
    if best_f.exists():
        data["best_models"] = safe_read_csv(best_f)

    # model list and reports
    data["model_files"] = sorted([p.name for p in MODELS_DIR.glob("*")]) if MODELS_DIR.exists() else []
    data["report_images"] = sorted([p.name for p in REPORTS_DIR.glob("*.png")]) if REPORTS_DIR.exists() else []

    return data

data = load_data()

# ----------------------------
# Streamlit layout
# ----------------------------
st.set_page_config(page_title="AirAware — Milestone 3", layout="wide")
st.title("AirAware — Milestone 3: Forecasts & Alerts")

# Sidebar controls
st.sidebar.header("Controls")
# ensure cities shown in order requested and only them
selected_city = st.sidebar.selectbox("City", CITIES)
# pollutant choices based on data (fallback to common list)
poll_choices = POLLUTANTS_COMMON.copy()
# if aggregated forecasts exist and have pollutant column, use those
if "forecasts" in data and data["forecasts"] is not None:
    if "Pollutant" in data["forecasts"].columns:
        poll_choices = sorted(data["forecasts"]["Pollutant"].dropna().unique().tolist())
elif data["percity_forecasts"]:
    # collect pollutants from filenames
    pset = sorted({x["pollutant"] for x in data["percity_forecasts"] if x["pollutant"]})
    if pset:
        poll_choices = pset
selected_pollutant = st.sidebar.selectbox("Pollutant", ["All"] + poll_choices)
show_models = st.sidebar.checkbox("Show model files", value=False)
st.sidebar.markdown("---")
st.sidebar.write("Files found in results/:")
for f in sorted([p.name for p in RESULTS_DIR.glob("*")]):
    st.sidebar.text(f)
st.sidebar.write("Models found:")
for m in data.get("model_files", [])[:30]:
    st.sidebar.text(m)

# ----------------------------
# AQI Gauge & 7-day Forecast table
# ----------------------------
st.subheader("AQI Gauge & 7-day Forecast")

if "forecasts_aqi" in data and data["forecasts_aqi"] is not None:
    df_aqi = data["forecasts_aqi"].copy()
    # normalize common column names
    if "City" not in df_aqi.columns:
        for c in df_aqi.columns:
            if c.lower() == "city":
                df_aqi = df_aqi.rename(columns={c:"City"})
                break
    if "Date" not in df_aqi.columns:
        for c in df_aqi.columns:
            if c.lower() in ("date","datetime","time"):
                df_aqi = df_aqi.rename(columns={c:"Date"})
                break
    # filter selected city
    df_city = df_aqi[df_aqi["City"].astype(str).str.lower() == selected_city.lower()]
    if df_city.empty:
        st.info(f"No AQI rows for {selected_city} in forecasts_aqi.csv")
    else:
        # ensure Date parsed
        if "Date" in df_city.columns:
            df_city["Date"] = pd.to_datetime(df_city["Date"], errors="coerce")
            df_city = df_city.sort_values("Date")
        # find AQI column name
        aqi_col = next((c for c in df_city.columns if c.lower()=="aqi"), None)
        bucket_col = next((c for c in df_city.columns if "bucket" in c.lower() or "category" in c.lower()), None)
        color_col = next((c for c in df_city.columns if "color" in c.lower() or "hex" in c.lower()), None)
        # latest gauge
        latest = df_city.iloc[-1]
        aqi_val = float(latest[aqi_col]) if aqi_col else float(latest.get("AQI", np.nan))
        category = latest[bucket_col] if bucket_col else latest.get("AQI_Bucket", latest.get("Category", "Unknown"))
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=aqi_val,
            title={"text": f"{selected_city} — {category}"},
            gauge={"axis":{"range":[0,500]},
                   "steps":[
                       {"range":[0,50],"color":"#2ecc71"},
                       {"range":[51,100],"color":"#f1c40f"},
                       {"range":[101,200],"color":"#e67e22"},
                       {"range":[201,300],"color":"#e74c3c"},
                       {"range":[301,500],"color":"#8e44ad"}
                   ]}
        ))
        st.plotly_chart(fig, use_container_width=True)
        st.write("7-day AQI forecast (top 7 rows):")
        cols_show = ["Date"]
        if aqi_col: cols_show.append(aqi_col)
        elif "AQI" in df_city.columns: cols_show.append("AQI")
        if bucket_col: cols_show.append(bucket_col)
        elif "AQI_Bucket" in df_city.columns: cols_show.append("AQI_Bucket")
        if color_col: cols_show.append(color_col)
        st.dataframe(df_city[cols_show].head(7))
else:
    st.info("No forecasts_aqi.csv found in results/. The UI will attempt to show pollutant forecasts instead.")

# ----------------------------
# Pollutant concentration plots
# ----------------------------
st.subheader("Pollutant Concentrations (Forecasts / Readings)")

def build_city_pollutant_df(city, pollutant):
    # priority 1: aggregated forecasts.csv
    if "forecasts" in data and data["forecasts"] is not None:
        df = data["forecasts"].copy()
        # filter city
        if "City" in df.columns:
            df = df[df["City"].astype(str).str.lower() == city.lower()]
        # filter pollutant
        if pollutant != "All" and "Pollutant" in df.columns:
            df = df[df["Pollutant"] == pollutant]
        # normalize Date & Forecast column
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        # detect forecast column name
        for col in df.columns:
            if col.lower() in ("forecast","value","prediction","pred"):
                df = df.rename(columns={col:"Forecast"})
                break
        # keep useful columns
        keep = [c for c in ["Date","Pollutant","Model","Forecast","Actual"] if c in df.columns]
        return df[keep].sort_values("Date")
    # priority 2: per-city files under results/forecasts/
    recs = []
    for rec in data.get("percity_forecasts", []):
        if rec["city"].lower() == city.lower():
            dfp = rec["df"].copy()
            dfp = normalize_df_date_and_value(dfp)
            # rename Value to Forecast
            if "Value" in dfp.columns:
                dfp = dfp.rename(columns={"Value":"Forecast"})
            # use pollutant from filename
            pol = rec.get("pollutant", None)
            dfp["Pollutant"] = pol
            if pollutant != "All" and pol and pol != pollutant:
                continue
            # keep Date & Forecast
            recs.append(dfp[["Date","Pollutant","Forecast"]])
    if recs:
        return pd.concat(recs, ignore_index=True).sort_values("Date")
    # fallback: empty df
    return pd.DataFrame(columns=["Date","Pollutant","Forecast"])

city_poll_df = build_city_pollutant_df(selected_city, selected_pollutant)
if city_poll_df.empty:
    st.info("No pollutant forecast/reading data found for this city/pollutant combination.")
else:
    # if multiple models present, aggregate mean across models for plotting when pollutant selected
    if "Model" in city_poll_df.columns and selected_pollutant != "All":
        plotdf = city_poll_df.groupby(["Date","Model"])["Forecast"].mean().reset_index()
        fig = px.line(plotdf, x="Date", y="Forecast", color="Model", markers=True, title=f"{selected_city} — {selected_pollutant} forecasts by model")
    elif "Model" in city_poll_df.columns and selected_pollutant == "All":
        plotdf = city_poll_df.groupby(["Date","Pollutant"])["Forecast"].mean().reset_index()
        fig = px.line(plotdf, x="Date", y="Forecast", color="Pollutant", markers=True, title=f"{selected_city} — pollutants (mean across models)")
    else:
        # no Model column: straightforward
        if selected_pollutant == "All":
            if "Pollutant" in city_poll_df.columns:
                plotdf = city_poll_df.groupby(["Date","Pollutant"])["Forecast"].mean().reset_index()
                fig = px.line(plotdf, x="Date", y="Forecast", color="Pollutant", markers=True, title=f"{selected_city} — pollutants")
            else:
                fig = px.line(city_poll_df, x="Date", y="Forecast", markers=True, title=f"{selected_city} — values")
        else:
            fig = px.line(city_poll_df, x="Date", y="Forecast", markers=True, title=f"{selected_city} — {selected_pollutant} forecast")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Active Alerts
# ----------------------------
st.subheader("Active Alerts")
if "alerts" in data and data["alerts"] is not None:
    df_alerts = data["alerts"].copy()
    # try to filter by City if possible
    if "City" in df_alerts.columns:
        df_alerts = df_alerts[df_alerts["City"].astype(str).str.lower() == selected_city.lower()]
    if df_alerts.empty:
        st.info("No active alerts for this city.")
    else:
        # detect message-like column
        msg_col = next((c for c in df_alerts.columns if any(k in c.lower() for k in ("alert","message","warning","status","type"))), None)
        show_cols = df_alerts.columns.tolist()
        if msg_col:
            show_cols = [c for c in show_cols if c in (msg_col,"Date","Time","City")]
        st.dataframe(df_alerts[show_cols].sort_values(show_cols[0] if show_cols else df_alerts.columns[0], ascending=False).head(50))
else:
    st.info("No alerts.csv in results/")

# ----------------------------
# Model performance & best models
# ----------------------------
st.subheader("Model Performance & Best Models")
if "performance" in data and data["performance"] is not None:
    st.dataframe(data["performance"].head(200))
else:
    st.info("No model performance CSV found in results/")

if "best_models" in data and data["best_models"] is not None:
    st.dataframe(data["best_models"])
else:
    st.info("No best_models.csv found in results/")

# ----------------------------
# Models list & reports
# ----------------------------
if show_models:
    st.subheader("Model files (models/)")
    if data.get("model_files"):
        for m in data["model_files"]:
            st.text(m)
    else:
        st.info("No model files found in models/")

st.subheader("Reports (images)")
if data.get("report_images"):
    for r in data["report_images"]:
        st.markdown(f"- {r}")
else:
    st.info("No images found in reports/")

# ----------------------------
# Download CSVs
# ----------------------------
st.subheader("Download CSVs (results/)")
csv_files = sorted([p for p in RESULTS_DIR.glob("*.csv")])
if csv_files:
    for p in csv_files:
        with open(p, "rb") as f:
            st.download_button(label=f"Download {p.name}", data=f, file_name=p.name)
else:
    st.info("No CSV files in results/ to download.")

# Footer
st.markdown("---")
st.caption("If something is missing, confirm the CSV names in results/ and per-city CSVs in results/forecasts/. This app tolerates common variants but requires Date and Forecast/AQI columns to be present.")
