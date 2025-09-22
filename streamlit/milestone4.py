# streamlit/milestone4.py
"""
Streamlit dashboard for Milestone 4 (AirAware)
Run: streamlit run streamlit/milestone4.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import pickle
from datetime import datetime

# ---------------------------
# Config
# ---------------------------
ROOT = Path.cwd()
RESULTS_DIR = ROOT / "results"
PER_CITY_FORECASTS_DIR = RESULTS_DIR / "forecasts"
MODELS_DIR = ROOT / "models"

CITIES = ["Ahmedabad", "Bengaluru", "Chennai", "Delhi", "Kolkata", "Mumbai"]
POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]

# ---------------------------
# Utilities
# ---------------------------
def safe_read_csv(path: Path):
    try:
        return pd.read_csv(path, parse_dates=["Date"])
    except Exception:
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()

def aqi_category(aqi: float):
    """Return AQI category name and color"""
    try:
        aqi = float(aqi)
    except:
        return "Unknown", "#cccccc"
    if aqi <= 50: return "Good", "#2ecc71"
    if aqi <= 100: return "Satisfactory", "#f1c40f"
    if aqi <= 200: return "Moderate", "#e67e22"
    if aqi <= 300: return "Poor", "#e74c3c"
    if aqi <= 400: return "Very Poor", "#8e44ad"
    return "Severe", "#7f0000"

def filter_time(df):
    """Filter dataframe for last 30 days"""
    if df.empty or "Date" not in df.columns:
        return df
    now = pd.Timestamp.now()
    start = now - pd.Timedelta(days=30)
    return df[df["Date"] >= start]

def load_model(city, pollutant):
    """Load trained model for a city and pollutant"""
    model_file = MODELS_DIR / f"{city}_{pollutant}_model.pkl"
    if model_file.exists():
        with open(model_file, "rb") as f:
            return pickle.load(f)
    return None

def predict_pollutant(model, days_ahead):
    """Predict pollutant value for a future day"""
    X = np.array([[days_ahead]])
    pred = model.predict(X)
    return float(pred[0])

# ---------------------------
# Load Data
# ---------------------------
@st.cache_data
def load_all():
    out = {}

    # forecasts_aqi
    fa = RESULTS_DIR / "forecasts_aqi.csv"
    if fa.exists():
        out["forecasts_aqi"] = safe_read_csv(fa)

    # forecasts
    fagg = RESULTS_DIR / "forecasts.csv"
    if fagg.exists():
        out["forecasts"] = safe_read_csv(fagg)

    # per-city forecast files
    percity = []
    if PER_CITY_FORECASTS_DIR.exists():
        for csv in PER_CITY_FORECASTS_DIR.glob("*.csv"):
            df = safe_read_csv(csv)
            if not df.empty:
                stem = csv.stem.split("_")
                city = stem[1] if len(stem) > 1 else "Unknown"
                pollutant = stem[2] if len(stem) > 2 else "Unknown"
                percity.append({"city": city, "pollutant": pollutant, "df": df})
    out["percity_forecasts"] = percity

    # alerts
    af = RESULTS_DIR / "alerts.csv"
    if af.exists():
        out["alerts"] = safe_read_csv(af)

    # model performance
    pf = RESULTS_DIR / "model_performance.csv"
    if pf.exists():
        out["performance"] = safe_read_csv(pf)

    return out

data = load_all()

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="AirAware — Milestone 4", layout="wide")
st.markdown("<h1 style='color:#1b2b6f;'>AirAware — Milestone 4</h1>", unsafe_allow_html=True)

# ---------------------------
# Controls (Left Panel)
# ---------------------------
left, middle, right = st.columns([1.2, 2.4, 2.4])
with left:
    st.markdown("### Controls")
    station = st.selectbox("City", CITIES)
    # horizon_choice = st.selectbox("Forecast Horizon", ["24 Hours", "48 Hours", "72 Hours", "7 Days"], index=3)
    admin_mode = st.checkbox("Admin Mode", value=False)

# ---------------------------
# Current AQI
# ---------------------------
with middle:
    st.markdown("### Current Air Quality")
    if "forecasts_aqi" in data:
        df_city = data["forecasts_aqi"][data["forecasts_aqi"]["City"].str.lower() == station.lower()]
        if df_city.empty:
            st.warning(f"No AQI forecast for {station}")
        else:
            latest = df_city.sort_values("Date").iloc[-1]
            aqi_val = float(latest.get("AQI", np.nan))
            cat, color = aqi_category(aqi_val)

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=aqi_val,
                title={"text": f"{station} — {cat}"},
                gauge={
                    "axis": {"range": [0, 500]},
                    "bar": {"color": "#ff9900"},
                    "bgcolor": "white",
                    "steps": [
                        {"range": [0, 50], "color": "#2ecc71"},
                        {"range": [51, 100], "color": "#f1c40f"},
                        {"range": [101, 200], "color": "#e67e22"},
                        {"range": [201, 300], "color": "#e74c3c"},
                        {"range": [301, 500], "color": "#8e44ad"}
                    ]
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                f"<div style='padding:12px;border-radius:8px;background:{color};color:white;text-align:center;'>"
                f"<b>AQI {aqi_val:.1f}</b> — {cat}</div>", unsafe_allow_html=True
            )
    else:
        st.info("⚠️ No forecasts_aqi.csv found in results/")

# ---------------------------
# Forecast Plot
# ---------------------------
with right:
    st.markdown("### Forecast Plot")
    selected_pollutant = st.selectbox("Choose Pollutant for Forecast", POLLUTANTS, index=0)
    model_options = ["RandomForest", "XGB", "Prophet", "LSTM"]
    selected_model = st.selectbox("Choose Model", model_options, index=0)

    df_plot = pd.DataFrame()
    if "forecasts" in data:
        df_all = data["forecasts"]
        if "Pollutant" in df_all.columns:
            df_plot = df_all[
                (df_all["City"].str.lower() == station.lower()) &
                (df_all["Pollutant"].str.upper() == selected_pollutant.upper())
            ]
    if df_plot.empty and data.get("percity_forecasts"):
        for rec in data["percity_forecasts"]:
            if rec["city"].lower() == station.lower() and selected_pollutant.lower() in rec["pollutant"].lower():
                df_plot = rec["df"]

    if df_plot.empty:
        st.warning(f"No forecast found for {station}, {selected_pollutant}")
    else:
        df_plot = df_plot.sort_values("Date")
        fig = px.line(df_plot, x="Date", y="Forecast",
                      title=f"{selected_pollutant} Forecast in {station} ({selected_model})")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Pollutant Trends (Last 30 Days)
# ---------------------------
st.markdown("### Pollutant Trends")
col1, col2 = st.columns(2)
with col1:
    pollutant_choice = st.selectbox("Pollutant", POLLUTANTS + ["All"], index=0)
with col2:
    st.markdown("**Time Range:** Last 30 Days")

trend_df = pd.DataFrame()
if "forecasts" in data:
    df_all = data["forecasts"]
    df_city = df_all[df_all["City"].str.lower() == station.lower()]
    if pollutant_choice != "All":
        df_city = df_city[df_city["Pollutant"] == pollutant_choice]
    trend_df = df_city[["Date", "Pollutant", "Forecast"]]

if trend_df.empty and data.get("percity_forecasts"):
    recs = []
    for rec in data["percity_forecasts"]:
        if rec["city"].lower() == station.lower():
            df = rec["df"].copy()
            df["Pollutant"] = rec["pollutant"]
            recs.append(df[["Date", "Pollutant", "Forecast"]])
    if recs:
        trend_df = pd.concat(recs)
        if pollutant_choice != "All":
            trend_df = trend_df[trend_df["Pollutant"] == pollutant_choice]

if trend_df.empty:
    st.warning(f"No pollutant trend data for {station}.")
else:
    trend_df = filter_time(trend_df)
    fig = px.line(trend_df, x="Date", y="Forecast", color="Pollutant",
                  title=f"Pollutant Trends — {station}", markers=True)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Calendar Prediction (Dynamic using Prophet CSVs)
# ---------------------------
st.markdown("### 📅 Predict Air Quality by Date")
selected_date = st.date_input("Select a Date", pd.Timestamp.today() + pd.Timedelta(days=1))
today = pd.Timestamp.today().date()
days_ahead = (selected_date - today).days

if days_ahead < 0:
    st.warning("Please select a future date.")
else:
    predictions = []
    for pollutant in POLLUTANTS:
        file_path = PER_CITY_FORECASTS_DIR / f"forecast_{station}_{pollutant}.csv"
        
        if file_path.exists():
            try:
                df = pd.read_csv(file_path, parse_dates=["ds"]).sort_values("ds")

                # Calculate difference in days from first date
                base_date = df["ds"].min()
                days_since_start = (pd.Timestamp(selected_date) - base_date).days

                # Prophet components: trend + weekly + yearly + multiplicative_terms
                trend = df["trend"].values[-1] if "trend" in df.columns else 0
                weekly = df["weekly"].values[-1] if "weekly" in df.columns else 0
                yearly = df["yearly"].values[-1] if "yearly" in df.columns else 0
                multiplicative = df["multiplicative_terms"].values[-1] if "multiplicative_terms" in df.columns else 0

                # Linear trend extrapolation
                trend_slope = (df["trend"].values[-1] - df["trend"].values[0]) / max((len(df)-1),1)
                predicted_trend = df["trend"].values[0] + trend_slope * days_since_start

                # Add seasonal components
                predicted_val = predicted_trend + weekly + yearly + multiplicative
                predicted_val = max(predicted_val, 0)  # non-negative

                predictions.append({"Pollutant": pollutant, "Predicted Value": round(predicted_val, 2)})

            except Exception as e:
                predictions.append({"Pollutant": pollutant, "Predicted Value": f"Error: {str(e)}"})
        else:
            predictions.append({"Pollutant": pollutant, "Predicted Value": "No forecast available"})

    df_pred = pd.DataFrame(predictions)
    st.subheader(f"Predicted Air Quality in {station} for {selected_date}")
    st.dataframe(df_pred)


# ---------------------------
# Alerts (Next 7 Days)  Alerts are dynamically shown for the next 7 days from the current system date.
# ---------------------------
st.markdown("### Alert Notifications (Next 7 Days)")
if "alerts" in data:
    df_alerts = data["alerts"][data["alerts"]["City"].str.lower() == station.lower()]

    if df_alerts.empty:
        st.info("No alerts for this station.")
    else:
        # Ensure Date column is datetime
        df_alerts["Date"] = pd.to_datetime(df_alerts["Date"], errors='coerce')

        # Filter alerts for next 7 days from today
        today = pd.Timestamp.today().normalize()
        next_7_days = today + pd.Timedelta(days=7)
        df_alerts_future = df_alerts[
            (df_alerts["Date"] >= today) & (df_alerts["Date"] <= next_7_days)
        ]

        if df_alerts_future.empty:
            st.info("No alerts in the next 7 days.")
        else:
            # Sort by date
            df_alerts_future = df_alerts_future.sort_values("Date")

            level_column_candidates = ["Level", "Alert_Level", "Severity"]
            level_colors = {
                "Moderate": "#f1c40f",
                "Hard": "#e74c3c",
                "Severe": "#8e44ad"
            }

            for _, row in df_alerts_future.iterrows():
                text = str(row.get("Alert", "Alert"))
                date = row["Date"].strftime("%Y-%m-%d")

                # Detect alert level dynamically
                level = None
                for col in level_column_candidates:
                    if col in df_alerts_future.columns:
                        level = str(row.get(col, "Moderate"))
                        break
                if not level:
                    level = "Moderate"

                color = level_colors.get(level, "#ff9933")

                st.markdown(
                    f"<div style='padding:12px;border-radius:10px;font-weight:bold;"
                    f"background: linear-gradient(90deg, {color}, #ffffff); color:white;'>"
                    f"{text} — {level} — {date}</div>", unsafe_allow_html=True
                )



# # ---------------------------
# # Admin Mode
# # ---------------------------
# if admin_mode:
#     st.markdown("### Admin: Model Performance")
#     if "performance" in data:
#         st.dataframe(data["performance"])
#     else:
#         st.info("No model performance CSV found.")
