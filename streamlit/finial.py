# streamlit/upload_predictor_all_gases_readable.py
import re
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.linear_model import LinearRegression

# ---------------------------
# AQI Calculation Helpers
# ---------------------------
def aqi_category(aqi):
    try:
        a = float(aqi)
    except:
        return "Unknown", "#cccccc"
    if a <= 50: return "Good 😊", "#2ecc71"
    if a <= 100: return "Satisfactory 🙂", "#f1c40f"
    if a <= 200: return "Moderate 😐", "#e67e22"
    if a <= 300: return "Poor 😷", "#e74c3c"
    if a <= 400: return "Very Poor 🤢", "#8e44ad"
    return "Severe ☠️", "#7f0000"

def plot_aqi_gauge(aqi_val, title="🌬️ Current AQI"):
    cat, color = aqi_category(aqi_val)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi_val,
        title={"text": f"{title} — {cat}", "font": {"size": 20, "color":"#111111"}},
        gauge={
            "axis": {"range": [0, 500], "tickcolor":"black"},
            "bar": {"color": "#ff6600"},  # bright orange bar
            "steps": [
                {"range": [0, 50], "color": "#66ff66"},      # bright green
                {"range": [51, 100], "color": "#ffff66"},    # bright yellow
                {"range": [101, 200], "color": "#ffcc66"},   # light orange
                {"range": [201, 300], "color": "#ff6666"},   # red
                {"range": [301, 400], "color": "#cc66ff"},   # purple
                {"range": [401, 500], "color": "#7f0000"}    # dark red
            ]
        }
    ))
    fig.update_layout(
        paper_bgcolor="#ffffff",  # light background
        font={"color":"#111111", "family":"Arial"},
        margin=dict(t=40, b=40, l=40, r=40)
    )
    return fig, color

def parse_to_numeric(val):
    try:
        return float(val)
    except:
        pass
    if isinstance(val, str):
        m = re.search(r"(-?\d+(\.\d+)?)", val.strip())
        if m:
            return float(m.group(1))
    return np.nan

def compute_aqi_from_pollutants(row):
    aqi_vals = []
    for col in pollutant_cols:
        if pd.notna(row[col]):
            val = parse_to_numeric(row[col])
            aqi_vals.append(val)
    return max(aqi_vals) if aqi_vals else np.nan

def show_current_aqi(df, station_name):
    if "AQI" in df.columns:
        vals = df["AQI"].dropna()
        if vals.empty: return
        raw_val = vals.iloc[-1]
        aqi_val = parse_to_numeric(raw_val)
    else:
        df["AQI_calc"] = df.apply(compute_aqi_from_pollutants, axis=1)
        vals = df["AQI_calc"].dropna()
        if vals.empty: return
        aqi_val = vals.iloc[-1]

    fig, color = plot_aqi_gauge(aqi_val, title=f"🌆 {station_name} — Current AQI")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        f"<div style='padding:15px;border-radius:12px;background:#4da6ff;color:white;text-align:center;font-size:20px;font-weight:bold;'>"
        f"🌟 AQI {aqi_val:.1f}</div>", 
        unsafe_allow_html=True
    )

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="🌬️ AirAware — All Pollutants Viewer", layout="wide")
st.markdown("## 🌈 AirAware — Upload CSV & View Pollutants", unsafe_allow_html=True)
st.markdown("<hr style='border:2px solid #4da6ff'>", unsafe_allow_html=True)

# File uploader
uploaded = st.file_uploader("📂 Upload CSV file", type=['csv'])
if not uploaded:
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"❌ Error reading CSV: {e}")
    st.stop()

# Ensure Date column
date_cols = df.filter(regex='date|ds|time|datetime', axis=1).columns
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
elif len(date_cols) > 0:
    df["Date"] = pd.to_datetime(df[date_cols[0]], errors="coerce")
else:
    df["Date"] = pd.to_datetime(df.index, errors="coerce")

# Detect City column
city_col = None
for c in df.columns:
    if any(x in c.lower() for x in ["city", "station"]):
        city_col = c
        break

# Detect pollutant columns
exclude_cols = ["date", "city", "station"]
pollutant_cols = [c for c in df.columns if c not in [city_col, "Date"] and pd.api.types.is_numeric_dtype(df[c]) and not any(x in c.lower() for x in exclude_cols)]

# Dropdowns
cities = ["All"] + (sorted(df[city_col].dropna().unique()) if city_col else [])
pollutants = ["All"] + pollutant_cols

col1, col2 = st.columns([1.5, 1.5])
with col1:
    station = st.selectbox("🏙️ Select City", options=cities)
with col2:
    selected_pollutant = st.selectbox("💨 Select Pollutant", options=pollutants)

# Filter by city
df_filtered = df.copy()
if city_col and station != "All":
    df_filtered = df_filtered[df_filtered[city_col] == station]

# ---------------------------
# Show Current AQI
# ---------------------------
st.markdown("### 🌟 Current Air Quality (AQI Gauge)")
show_current_aqi(df_filtered, station or "Site")

# ---------------------------
# Pollutant Graphs (Improved Date Visibility)
# ---------------------------
st.markdown("### 📈 Pollutant Graphs")
plot_bg = "#ffffff"  # light background

if selected_pollutant == "All":
    if city_col and station != "All":
        fig = px.line(
            df_filtered, x="Date", y=pollutant_cols,
            title=f"🌈 All Pollutants in {station}",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_traces(mode="lines+markers")
        fig.update_layout(
            paper_bgcolor=plot_bg, plot_bgcolor=plot_bg, font_color="#111111",
            title_font=dict(size=20, color="#111111"),
            xaxis_title_font=dict(size=16, color="#111111"),
            yaxis_title_font=dict(size=16, color="#111111"),
            xaxis_tickangle=-45,
            xaxis_tickfont=dict(size=12, color="#111111"),
            yaxis_tickfont=dict(size=12, color="#111111")
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Select a specific city to view all pollutants together.")
else:
    if city_col:
        fig = px.line(
            df_filtered, x="Date", y=selected_pollutant, color=city_col,
            title=f"💨 {selected_pollutant} Levels Across Cities",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
    else:
        fig = px.line(
            df_filtered, x="Date", y=selected_pollutant,
            title=f"💨 {selected_pollutant} Levels",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
    fig.update_traces(mode="lines+markers")
    fig.update_layout(
        paper_bgcolor=plot_bg, plot_bgcolor=plot_bg, font_color="#111111",
        title_font=dict(size=20, color="#111111"),
        xaxis_title_font=dict(size=16, color="#111111"),
        yaxis_title_font=dict(size=16, color="#111111"),
        xaxis_tickangle=-45,
        xaxis_tickfont=dict(size=12, color="#111111"),
        yaxis_tickfont=dict(size=12, color="#111111")
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Calendar-based Prediction
# ---------------------------
st.markdown("### 📅 Calendar-based Prediction")
target_date = st.date_input("🗓️ Select a future date to predict pollutant levels")
if target_date:
    df_pred = pd.DataFrame()
    X = df_filtered["Date"].map(datetime.toordinal).values.reshape(-1, 1)

    for col in pollutant_cols:
        y = df_filtered[col].dropna()
        if y.empty: 
            df_pred.loc[0, col] = np.nan
            continue
        valid_idx = y.index
        X_valid = X[valid_idx] if len(X) > max(valid_idx) else X[:len(y)]
        model = LinearRegression()
        model.fit(X_valid.reshape(-1, 1), y.values)
        pred_val = model.predict([[pd.to_datetime(target_date).toordinal()]])[0]
        df_pred.loc[0, col] = round(float(pred_val), 2)

    if not df_pred.empty:
        st.subheader(f"🎯 Predicted Pollutant Levels on {target_date}")
        st.markdown(
            f"<div style='padding:15px;border-radius:12px;background:#4da6ff;color:#111111;text-align:center;font-size:18px;font-weight:bold;'>"
            f"📊 Predictions</div>", unsafe_allow_html=True
        )
        st.dataframe(df_pred.astype(float), use_container_width=True)

        df_melted = df_pred.melt(var_name="Pollutant", value_name="Predicted Value")
        fig_pred = px.bar(
            df_melted, x="Pollutant", y="Predicted Value",
            color="Predicted Value", color_continuous_scale="Viridis",
            text="Predicted Value", title=f"📊 Predicted Pollutants on {target_date}"
        )
        fig_pred.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_pred.update_layout(
            paper_bgcolor=plot_bg, plot_bgcolor=plot_bg, font_color="#111111",
            title_font=dict(size=20, color="#111111", family="Arial"),
            xaxis_title_font=dict(size=16, color="#111111"),
            yaxis_title_font=dict(size=16, color="#111111"),
            xaxis_tickfont=dict(size=12, color="#111111"),
            yaxis_tickfont=dict(size=12, color="#111111"),
            coloraxis_colorbar=dict(title="Value")
        )
        st.plotly_chart(fig_pred, use_container_width=True)
    else:
        st.warning("⚠️ No pollutant data available for prediction.")

# ---------------------------
# Preview Data
# ---------------------------
with st.expander("👀 Preview uploaded CSV"):
    st.dataframe(df.head(100))
