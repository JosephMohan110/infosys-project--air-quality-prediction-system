# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
import mysql.connector

st.set_page_config(layout="wide", page_title="Air Quality Data Explorer")

# -------------------------
# MySQL Configuration
# -------------------------
DB_CONFIG = {
    "host": "localhost",
    "user": "root",          # Default in XAMPP
    "password": "",          # Set if you configured one
    "database": "airaware",  # Make sure this database exists
    "port": 3306             # Default XAMPP port
}

# Table names (must match your loader script)
DATASETS = {
    "city_day": "city_day",
    "city_hour": "city_hour",
    "station_day": "station_day",
    "station_hour": "station_hour",
    "stations": "stations"
}

# Pollutants (normalized lowercase)
POLLUTANTS = ["pm2_5", "pm10", "no2", "o3", "so2", "co"]
COLORS = px.colors.qualitative.Dark24

# -------------------------
# Helpers
# -------------------------
@st.cache_data
def load_from_mysql(table_name, freq="H"):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        query = f"SELECT * FROM {table_name};"
        df = pd.read_sql(query, conn)
        conn.close()
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()

    # Normalize column names
    df.columns = [c.strip().replace(" ", "_").replace("-", "_").lower() for c in df.columns]

    # Ensure datetime column exists
    for col in ["datetime", "date", "timestamp"]:
        if col in df.columns:
            df["datetime"] = pd.to_datetime(df[col])
            break
    if "datetime" not in df.columns:
        df["datetime"] = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq=freq)

    return df


def extract_locations(df, dataset_type):
    key = "city" if dataset_type.startswith("city") else "station"
    if key in df.columns:
        return sorted(df[key].dropna().unique().tolist())
    return []


def filter_by_location_and_range(df, dataset_type, location, time_range):
    key = "city" if dataset_type.startswith("city") else "station"
    if key in df.columns and location is not None:
        df = df[df[key] == location].copy()
    df = df.sort_values("datetime")
    if time_range != "all" and not df.empty:
        last = df["datetime"].max()
        if time_range == "1d":
            start = last - timedelta(days=1)
        elif time_range == "7d":
            start = last - timedelta(days=7)
        elif time_range == "30d":
            start = last - timedelta(days=30)
        else:
            start = df["datetime"].min()
        df = df[(df["datetime"] >= start) & (df["datetime"] <= last)]
    return df


def compute_data_quality(df):
    if df.empty:
        return {"completeness": "--", "validity": "--"}
    total, present, valid, checked = 0, 0, 0, 0
    for p in POLLUTANTS:
        if p not in df.columns:
            continue
        total += df.shape[0]
        present += df[p].notna().sum()
        valid_mask = df[p].apply(lambda x: pd.notna(x) and np.isfinite(x) and x >= 0 and x < 1e5)
        checked += df[p].notna().sum()
        valid += valid_mask.sum()
    completeness = int(round((present/total)*100)) if total > 0 else "--"
    validity = int(round((valid/checked)*100)) if checked > 0 else "--"
    return {"completeness": f"{completeness}%", "validity": f"{validity}%"}

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.header("Data Controls")
    dataset_type = st.selectbox("Dataset", options=list(DATASETS.keys()), format_func=lambda x: x.replace("_"," ").title())
    raw_df = load_from_mysql(DATASETS[dataset_type], freq="H" if "hour" in dataset_type else "D")
    locations = extract_locations(raw_df, dataset_type)
    location = st.selectbox("Location", options=locations) if locations else st.text_input("Location")
    time_range = st.selectbox(
        "Time Range",
        options=["1d", "7d", "30d", "all"],
        format_func=lambda x: {"1d":"Last 24 Hours","7d":"Last 7 Days","30d":"Last 30 Days","all":"All"}[x]
    )
    pollutants = st.multiselect("Pollutants", options=POLLUTANTS, default=["pm2_5"])
    apply = st.button("Apply Filters")

    st.markdown("### Data Quality")
    dq_placeholder = st.empty()
    st.markdown("---")
    st.write("Loaded from MySQL table:")
    st.code(DATASETS[dataset_type])

# -------------------------
# Main Layout
# -------------------------
st.title("Air Quality Data Explorer")
col1, col2 = st.columns([2,1])

with col1:
    ts_col, corr_col = st.columns(2)
    with ts_col:
        st.subheader(f"{', '.join(pollutants)} Time Series")
        ts_card = st.container()
    with corr_col:
        st.subheader("Pollutant Correlations (bubble)")
        corr_card = st.container()

    sum_col, hist_col = st.columns([1,2])
    with sum_col:
        st.subheader("Statistical Summary")
        stats_card = st.container()
    with hist_col:
        st.subheader("Distribution Analysis")
        hist_card = st.container()

with col2:
    st.markdown("## Status")
    status = st.empty()
    st.markdown("## Data Quality")
    completeness_box = st.empty()
    validity_box = st.empty()
    st.markdown("---")
    st.markdown("**Legend / Notes**")
    st.write("This app loads data directly from MySQL (XAMPP).")

# -------------------------
# Data Filtering & Rendering
# -------------------------
if (apply is False) and ("last_filtered" not in st.session_state):
    st.session_state.last_filtered = True

filtered = filter_by_location_and_range(raw_df, dataset_type, location, time_range)

dq = compute_data_quality(filtered)
dq_placeholder.info(f"Completeness: {dq['completeness']} • Validity: {dq['validity']}")
completeness_box.metric("Completeness", dq["completeness"])
validity_box.metric("Validity", dq["validity"])
status.info(f"Rendered ({len(filtered)} rows) — dataset: {dataset_type}")

filtered["datetime"] = pd.to_datetime(filtered["datetime"])

# Time series
with ts_card:
    if filtered.empty or len(pollutants) == 0:
        st.info("No data to plot")
    else:
        fig = go.Figure()
        for i, p in enumerate(pollutants):
            if p not in filtered.columns:
                continue
            fig.add_trace(go.Scatter(x=filtered["datetime"], y=filtered[p],
                                     mode="lines", name=p,
                                     line=dict(color=COLORS[i % len(COLORS)])))
        fig.update_layout(margin=dict(t=10,b=10,l=10,r=10),
                          height=360, xaxis_title="Time", yaxis_title="Concentration (µg/m³)")
        st.plotly_chart(fig, use_container_width=True)

# Histogram
with hist_card:
    if filtered.empty or len(pollutants) == 0:
        st.info("No data")
    else:
        p = pollutants[0]
        if p not in filtered.columns:
            st.info(f"{p} not found in data")
        else:
            fig = px.histogram(filtered, x=p, nbins=8, marginal="box", title=f"{p} distribution")
            fig.update_layout(height=340, margin=dict(t=30,b=10,l=10,r=10))
            st.plotly_chart(fig, use_container_width=True)

# Stats
with stats_card:
    if filtered.empty or len(pollutants) == 0:
        st.write("No stats")
    else:
        p = pollutants[0]
        vals = filtered[p].dropna().astype(float) if p in filtered.columns else pd.Series([], dtype=float)
        if vals.empty:
            st.write("No valid values")
        else:
            stats = {
                "Mean": round(vals.mean(),2),
                "Median": round(vals.median(),2),
                "Max": round(vals.max(),2),
                "Min": round(vals.min(),2),
                "Std Dev": round(vals.std(),2),
                "Count": int(vals.count())
            }
            st.table(pd.DataFrame(stats, index=[p]))

# Correlation
with corr_card:
    present_cols = [c for c in POLLUTANTS if c in filtered.columns]
    corr_df = filtered[present_cols].corr(method="pearson")
    bubbles = []
    for i, a in enumerate(present_cols):
        for j, b in enumerate(present_cols):
            if j <= i:
                continue
            corr_val = corr_df.loc[a, b]
            if pd.isna(corr_val):
                continue
            size = max(6, abs(corr_val) * 80)
            bubbles.append(dict(x=i, y=j, size=size, corr=corr_val, a=a, b=b))

    if not bubbles:
        st.info("Not enough overlapping data for correlation plot")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[b["x"] for b in bubbles],
            y=[b["y"] for b in bubbles],
            mode="markers",
            marker=dict(size=[b["size"] for b in bubbles],
                        sizemode="area",
                        color=[b["corr"] for b in bubbles],
                        colorscale="RdYlGn",
                        showscale=True,
                        colorbar=dict(title="corr")),
            text=[f"{b['a']} / {b['b']}: {b['corr']:.2f}" for b in bubbles],
            hoverinfo="text"
        ))
        fig.update_layout(
            xaxis=dict(tickmode="array", tickvals=list(range(len(present_cols))), ticktext=present_cols),
            yaxis=dict(tickmode="array", tickvals=list(range(len(present_cols))), ticktext=present_cols),
            height=360, margin=dict(t=10,b=10,l=10,r=10)
        )
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Design & layout same as Milestone 1 — now loading data directly from MySQL (XAMPP).")
