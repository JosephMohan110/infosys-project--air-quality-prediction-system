# Milestone 2 Streamlit App with Two-City Forecast Comparison
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# -------------------------
# Page config
# -------------------------
st.set_page_config(layout="wide", page_title="Milestone 2 - Forecast Models")

# -------------------------
# Paths
# -------------------------
RESULTS_DIR = Path("results")
FORECAST_DIR = RESULTS_DIR / "forecasts"

files = {
    "model_perf": RESULTS_DIR / "model_performance.csv",
    "best_models": RESULTS_DIR / "best_models.csv",
    "comparison": RESULTS_DIR / "model_comparison_per_pollutant.csv",
    "alerts": RESULTS_DIR / "alerts_summary.csv"
}

# -------------------------
# Load CSV safely
# -------------------------
@st.cache_data
def load_csv(path):
    if path.exists():
        return pd.read_csv(path)
    else:
        return pd.DataFrame()

model_perf = load_csv(files["model_perf"])
best_models = load_csv(files["best_models"])
comparison = load_csv(files["comparison"])
alerts = load_csv(files["alerts"])

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.title("Controls")

# Dynamically list cities and pollutants from forecast files
forecast_files = list(FORECAST_DIR.glob("forecast_*.csv"))
cities = sorted(set(f.name.split("_")[1] for f in forecast_files))
pollutants = sorted(set(f.name.split("_")[2].split(".")[0] for f in forecast_files))

selected_city1 = st.sidebar.selectbox("Select City 1", cities)
selected_city2 = st.sidebar.selectbox("Select City 2 (for comparison)", cities, index=1)
selected_pollutant = st.sidebar.selectbox("Select Pollutant", pollutants)

# -------------------------
# Title
# -------------------------
st.title("🔮 Milestone 2: Forecast Dashboard")
st.caption("Weeks 3–4: Forecasts, Model Evaluation & Accuracy Metrics")

# -------------------------
# Function to load forecast CSV
# -------------------------
def load_forecast(city, pollutant):
    file_path = FORECAST_DIR / f"forecast_{city}_{pollutant}.csv"
    if file_path.exists():
        df = pd.read_csv(file_path)
        date_col = next((c for c in df.columns if "date" in c.lower() or "ds" in c.lower()), None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.rename(columns={date_col: "Datetime"})
        return df
    else:
        return None

# -------------------------
# Forecast Comparison Graph
# -------------------------
st.subheader(f"Forecast Comparison for {selected_pollutant}")

df1 = load_forecast(selected_city1, selected_pollutant)
df2 = load_forecast(selected_city2, selected_pollutant)

fig = go.Figure()

for df, city in zip([df1, df2], [selected_city1, selected_city2]):
    if df is not None:
        # Forecast
        forecast_col = next((c for c in df.columns if "yhat" in c.lower() or "forecast" in c.lower()), None)
        if forecast_col:
            fig.add_trace(go.Scatter(
                x=df['Datetime'], y=df[forecast_col],
                mode='lines', name=f'{city} Forecast', line=dict(dash='dot')
            ))
        # Actual
        if 'y_true' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['Datetime'], y=df['y_true'],
                mode='lines', name=f'{city} Actual'
            ))
fig.update_layout(
    xaxis_title="Date",
    yaxis_title=f"{selected_pollutant} (µg/m³)",
    height=500,
    margin=dict(t=20,b=20,l=20,r=20)
)
st.plotly_chart(fig, use_container_width=True)

st.sidebar.title("Forecast Accuracy Metrics")

# -------------------------
# Forecast Accuracy Metrics
# -------------------------
st.subheader("📊 Forecast Accuracy Metrics")

# File path
accuracy_file = RESULTS_DIR / "forecasts" / "forecast_accuracy_metrics.csv"

# Load CSV if it exists
if accuracy_file.exists():
    accuracy_df = pd.read_csv(accuracy_file)
else:
    accuracy_df = pd.DataFrame()

# Check if file is not empty
if not accuracy_df.empty:
    # Sidebar: select two cities for comparison
    cities = accuracy_df["City"].unique().tolist()
    selected_city1 = st.sidebar.selectbox("Select City 1", cities, index=0)
    selected_city2 = st.sidebar.selectbox("Select City 2", cities, index=1 if len(cities) > 1 else 0)

    # Filter by selected cities and pollutant
    filtered_df = accuracy_df[
        (accuracy_df["City"].isin([selected_city1, selected_city2])) &
        (accuracy_df["Pollutant"] == selected_pollutant)
    ]

    if not filtered_df.empty:
        st.dataframe(filtered_df)

        # Optional: Plot RMSE/MAE comparison
        metric_cols = [c for c in filtered_df.columns if c not in ["City", "Pollutant"]]
        if metric_cols:
            import plotly.express as px
            fig = px.bar(
                filtered_df.melt(id_vars=["City", "Pollutant"], value_vars=metric_cols, 
                                 var_name="Metric", value_name="Value"),
                x="Metric", y="Value", color="City", barmode="group",
                title=f"Forecast Accuracy Metrics for {selected_pollutant}"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No metric columns found in the accuracy file.")
    else:
        st.info("⚠️ No accuracy data available for the selected cities/pollutant.")
else:
    st.info("Forecast accuracy metrics file is missing or empty.")


# -------------------------
# Model Performance & Best Models
# -------------------------
col1, col2 = st.columns([1,1])

with col1:
    st.subheader("Model Performance")
    if not model_perf.empty:
        st.dataframe(model_perf)
    else:
        st.info("No model performance data found.")

with col2:
    st.subheader("Best Models by Pollutant")
    if not best_models.empty:
        st.dataframe(best_models)
    else:
        st.info("No best_models.csv available.")

# -------------------------
# Alerts
# -------------------------
st.subheader("⚠️ Alerts Summary")
if not alerts.empty:
    st.dataframe(alerts)
else:
    st.info("No alerts summary available.")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption("Milestone 2 — Forecasting & Model Evaluation Dashboard")
