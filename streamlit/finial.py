# streamlit/upload_predictor_all_gases_fixed.py
import re
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

# ---------------------------
# Helpers
# ---------------------------

def deduplicate_columns(df):
    cols = list(df.columns)
    counts, new_cols = {}, []
    for c in cols:
        if c in counts:
            counts[c] += 1
            new_cols.append(f"{c}__dup{counts[c]}")
        else:
            counts[c] = 0
            new_cols.append(c)
    df.columns = new_cols
    return df

def aqi_category(aqi):
    try: a = float(aqi)
    except: return "Unknown", "#cccccc"
    if a <= 50: return "Good", "#2ecc71"
    if a <= 100: return "Satisfactory", "#f1c40f"
    if a <= 200: return "Moderate", "#e67e22"
    if a <= 300: return "Poor", "#e74c3c"
    if a <= 400: return "Very Poor", "#8e44ad"
    return "Severe", "#7f0000"

def plot_aqi_gauge(aqi_val, title="Current AQI"):
    cat, color = aqi_category(aqi_val)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi_val,
        title={"text": f"{title} — {cat}"},
        gauge={
            "axis": {"range": [0, 500]},
            "bar": {"color": "#ff9900"},
            "steps": [
                {"range": [0, 50], "color": "#2ecc71"},
                {"range": [51, 100], "color": "#f1c40f"},
                {"range": [101, 200], "color": "#e67e22"},
                {"range": [201, 300], "color": "#e74c3c"},
                {"range": [301, 500], "color": "#8e44ad"}
            ]
        }
    ))
    return fig, color

def parse_to_numeric_aqi(raw):
    try: return float(raw)
    except: pass
    if isinstance(raw, str):
        m = re.search(r"(-?\d+(\.\d+)?)", raw.strip())
        if m: return float(m.group(1))
        cat_map = {"good":25,"satisfactory":75,"moderate":150,"poor":250,"very poor":350,"severe":450}
        return float(cat_map.get(raw.lower(), np.nan))
    return None

def detect_columns(df):
    city_col = pollutant_col = value_col = None
    for c in df.columns:
        low = c.lower()
        if any(x in low for x in ['city','station']) and not city_col: city_col = c
        if any(x in low for x in ['pollutant','param']) and not pollutant_col: pollutant_col = c
        if any(x in low for x in ['aqi','value','pm','co','no2','o3']) and not value_col: value_col = c
    if not value_col:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols: value_col = numeric_cols[0]
    return city_col, pollutant_col, value_col

def show_current_aqi(df, value_col, station_name):
    if value_col not in df.columns: return
    vals = df[value_col].dropna()
    if vals.empty: return
    raw_val = vals.iloc[-1]
    aqi_val = parse_to_numeric_aqi(raw_val)
    if aqi_val is None: return
    fig, color = plot_aqi_gauge(aqi_val, title=f"{station_name} — Current")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"<div style='padding:12px;border-radius:8px;background:{color};color:white;text-align:center;'><b>AQI {aqi_val:.1f}</b></div>", unsafe_allow_html=True)

def fit_linear_model(df, date_col='Date', value_col='Value'):
    df = df.dropna(subset=[date_col,value_col]).sort_values(date_col)
    df['days'] = (df[date_col]-df[date_col].min()).dt.days
    X = df[['days']].values; y = df[value_col].values
    model = LinearRegression()
    model.fit(X,y)
    return model, df

# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="AirAware — Simplified (All Gases)", layout="wide")
st.title("AirAware — Upload CSV & Predict (All Gases)")

uploaded = st.file_uploader("Upload CSV file", type=['csv'])
if not uploaded: st.stop()

try:
    df_raw = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Error reading CSV: {e}")
    st.stop()

df = deduplicate_columns(df_raw)
# Parse Date
date_cols = df.filter(regex='date|ds|time|datetime', axis=1).columns
df['Date'] = pd.to_datetime(df['Date'], errors='coerce') if 'Date' in df.columns else pd.to_datetime(df[date_cols[0]], errors='coerce') if len(date_cols)>0 else pd.to_datetime(df.index, errors='coerce')

city_col, pollutant_col, value_col = detect_columns(df)
cities = ['All'] + sorted(df[city_col].dropna().astype(str).unique()) if city_col else ['All']

# Pollutant options: ALL gases present in CSV
pollutant_options = sorted(df[pollutant_col].dropna().astype(str).unique()) if pollutant_col else ['All']

col1,col2 = st.columns([1.5,1.5])
with col1: station = st.selectbox("City", options=cities)
with col2: selected_pollutant = st.selectbox("Pollutant", options=pollutant_options)

df_filtered = df.copy()
if city_col and station!='All': df_filtered = df_filtered[df_filtered[city_col].astype(str)==station]
if pollutant_col and selected_pollutant!='All': df_filtered = df_filtered[df_filtered[pollutant_col].astype(str)==selected_pollutant]

# Show Current AQI
st.header("Current Air Quality")
show_current_aqi(df_filtered, value_col, station or "Site")

# Historical Plot
st.header("Historical / Forecast Plot")
plot_df = df_filtered[['Date', value_col]].dropna()
plot_df = plot_df.rename(columns={value_col:'Value'})
if not plot_df.empty:
    st.plotly_chart(px.line(plot_df, x='Date', y='Value', title="Historical AQI", markers=True), use_container_width=True)

# Predict AQI
st.header("Predict AQI by Date")
selected_date = st.date_input("Select a Date", pd.Timestamp.today().date()+pd.Timedelta(days=1))
predictions=[]
if not plot_df.empty:
    model, fit_df = fit_linear_model(plot_df, date_col='Date', value_col='Value')
    days = (pd.Timestamp(selected_date)-fit_df['Date'].min()).days
    pred_val = model.predict(np.array([[days]]))[0]
    predictions.append({'Date':selected_date,'Predicted AQI':round(float(pred_val),2)})

pred_df = pd.DataFrame(predictions)
st.table(pred_df)

with st.expander("Preview uploaded CSV"):
    st.dataframe(df.head(200))
