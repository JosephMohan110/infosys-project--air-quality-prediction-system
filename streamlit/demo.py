import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
results_path = Path("results")
models_path = Path("models")

# -----------------------------
# Load Data Function
# -----------------------------
@st.cache_data
def load_data():
    data = {}
    
    # Forecast files
    for name in ["forecast_aqi.csv", "forecast.csv"]:
        file = results_path / name
        if file.exists():
            data["forecast"] = pd.read_csv(file)
            break

    # Pollutant concentrations (search inside results subfolders)
    pollutant_data = []
    for sub in results_path.rglob("*.csv"):
        if sub.stem.lower() in ["co", "no", "no2", "pm2.5", "pm10", "o3", "so2"]:
            df = pd.read_csv(sub)
            df["Pollutant"] = sub.stem.upper()
            df["City"] = sub.parent.stem  # city from folder name
            pollutant_data.append(df)
    data["pollutants"] = pd.concat(pollutant_data, ignore_index=True) if pollutant_data else pd.DataFrame()

    # Alerts
    for name in ["alert_summary.csv", "alert.csv"]:
        file = results_path / name
        if file.exists():
            data["alerts"] = pd.read_csv(file)
            break

    # Model performance
    for name in ["model_performance.csv", "forecast_metrics_summary.csv", "forecast_accuracy_matrics.csv"]:
        file = results_path / name
        if file.exists():
            data["performance"] = pd.read_csv(file)
            break

    # Best models summary
    file = results_path / "best_models.csv"
    if file.exists():
        data["best_models"] = pd.read_csv(file)

    return data

data = load_data()

# -----------------------------
# Streamlit Layout
# -----------------------------
st.set_page_config(page_title="Air Quality Alert System", layout="wide")
st.markdown("<h1 style='color:#d35400;'>Air Quality Alert System</h1>", unsafe_allow_html=True)

# -----------------------------
# Current AQI & Forecast
# -----------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Current AQI Status")
    if "forecast" in data and not data["forecast"].empty:
        df_forecast = data["forecast"]
        current_aqi = df_forecast.iloc[0].get("AQI", 0)
        category = df_forecast.iloc[0].get("Category", "Unknown")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=current_aqi,
            title={"text": category},
            gauge={
                "axis": {"range": [0, 500]},
                "bar": {"color": "orange"},
                "steps": [
                    {"range": [0, 50], "color": "green"},
                    {"range": [51, 100], "color": "yellow"},
                    {"range": [101, 150], "color": "orange"},
                    {"range": [151, 200], "color": "red"},
                    {"range": [201, 300], "color": "purple"},
                    {"range": [301, 500], "color": "maroon"},
                ]
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)
    else:
        st.info("No forecast data available.")

with col2:
    st.subheader("7-Day Forecast")
    if "forecast" in data and not data["forecast"].empty:
        st.dataframe(data["forecast"].head(7))
    else:
        st.warning("Forecast file missing!")

# -----------------------------
# Pollutant Concentrations
# -----------------------------
st.subheader("Pollutant Concentrations")
if "pollutants" in data and not data["pollutants"].empty:
    cities = data["pollutants"]["City"].unique()
    selected_city = st.selectbox("Select City", cities)
    city_data = data["pollutants"][data["pollutants"]["City"] == selected_city]

    if "Time" in city_data.columns:
        fig_pollutants = px.line(
            city_data,
            x="Time",
            y="Value",  # assumes pollutant files have Value column
            color="Pollutant",
            title=f"Pollutant Levels in {selected_city}"
        )
        st.plotly_chart(fig_pollutants, use_container_width=True)
    else:
        st.warning("No Time column in pollutant data.")
else:
    st.info("No pollutant data available.")

# -----------------------------
# Active Alerts
# -----------------------------
st.subheader("Active Alerts")
if "alerts" in data and not data["alerts"].empty:
    df_alerts = data["alerts"]
    # Detect alert column automatically
    alert_col = None
    for col in df_alerts.columns:
        if any(word in col.lower() for word in ["alert", "type", "message", "status", "warning"]):
            alert_col = col
            break

    if alert_col:
        for _, row in df_alerts.iterrows():
            text = str(row[alert_col])
            color = "orange" if "Moderate" in text else "red" if "High" in text or "Unhealthy" in text else "yellow"
            st.markdown(
                f"<div style='padding:10px; border-radius:8px; background-color:{color}; color:white;'>"
                f"<b>{text}</b><br>{row.get('Date','')} @ {row.get('Time','')}"
                "</div><br>",
                unsafe_allow_html=True
            )
    else:
        st.warning("Could not detect alert column.")
else:
    st.info("No alerts data available.")

# -----------------------------
# Model Performance
# -----------------------------
st.subheader("Model Performance")
if "performance" in data and not data["performance"].empty:
    st.dataframe(data["performance"].head(10))
else:
    st.info("No model performance data available.")

# -----------------------------
# Best Models
# -----------------------------
st.subheader("Best Models")
if "best_models" in data and not data["best_models"].empty:
    st.dataframe(data["best_models"])
else:
    st.info("No best models file found.")

# -----------------------------
# Admin Section - List Model Files
# -----------------------------
st.subheader("Available Models")
if models_path.exists():
    model_files = [f.name for f in models_path.glob("*.*")]
    if model_files:
        st.write(model_files)
    else:
        st.info("No model files in models/ folder.")
else:
    st.info("Models folder not found.")
