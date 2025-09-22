# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import timedelta

st.set_page_config(layout="wide", page_title="Air Quality Data Explorer")

# -------------------------
# Helpers & config
# -------------------------
DATASETS = {
    "city_day": Path("../data/processed/city_day_processed.csv"),
    "city_hour": Path("../data/processed/city_hour_processed.csv"),
    "station_day": Path("../data/processed/station_day_processed.csv"),
    "station_hour": Path("../data/processed/station_hour_processed.csv"),
}
POLLUTANTS = ["PM2.5", "PM10", "NO2", "O3", "SO2", "CO"]
COLORS = px.colors.qualitative.Dark24

@st.cache_data
def load_or_generate(path: Path, freq="H"):
    if path.exists():
        df = pd.read_csv(path)
        for col in ["Datetime", "Date", "timestamp", "datetime"]:
            if col in df.columns:
                df["Datetime"] = pd.to_datetime(df[col])
                break
        if "Datetime" not in df.columns:
            df["Datetime"] = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq=freq)
        return df

    # generate synthetic data
    rng = pd.date_range(end=pd.Timestamp.now(), periods=24*30 if freq=="H" else 30, freq=freq)
    cities = ["Delhi", "Mumbai", "Chennai", "Kolkata", "Bengaluru"]
    stations = ["Delhi", "Mumbai", "Chennai", "Kolkata", "Bengaluru"]
    rows = []
    for loc in (cities if "city" in path.name else stations):
        base = np.random.uniform(10, 60)
        for dt in rng:
            row = {"Datetime": dt, "City": loc, "Station": loc}
            for p in POLLUTANTS:
                diurnal = 10 * np.sin((dt.hour/24) * 2 * np.pi) if freq == "H" else 0
                val = max(0, base + diurnal + np.random.normal(scale=8.0))
                row[p] = round(val + (np.random.rand() - 0.5) * 5, 2)
            rows.append(row)
    return pd.DataFrame(rows)

def extract_locations(df, dataset_type):
    key = "City" if dataset_type.startswith("city") else "Station"
    if key in df.columns:
        return sorted(df[key].dropna().unique().tolist())
    return sorted(df.columns.dropna().tolist())

def filter_by_location_and_range(df, dataset_type, location, time_range):
    key = "City" if dataset_type.startswith("city") else "Station"
    if key in df.columns and location is not None:
        df = df[df[key] == location].copy()
    df = df.sort_values("Datetime")
    if time_range != "all" and not df.empty:
        last = df["Datetime"].max()
        if time_range == "1d":
            start = last - timedelta(days=1)
        elif time_range == "7d":
            start = last - timedelta(days=7)
        elif time_range == "30d":
            start = last - timedelta(days=30)
        else:
            start = df["Datetime"].min()
        df = df[(df["Datetime"] >= start) & (df["Datetime"] <= last)]
    return df

def compute_data_quality(df):
    if df.empty:
        return {"completeness": "--", "validity": "--"}
    total, present, valid, checked = 0, 0, 0, 0
    for p in POLLUTANTS:
        if p not in df.columns: continue
        total += df.shape[0]
        present += df[p].notna().sum()
        valid_mask = df[p].apply(lambda x: pd.notna(x) and np.isfinite(x) and x >= 0 and x < 1e5)
        checked += df[p].notna().sum()
        valid += valid_mask.sum()
    completeness = int(round((present/total)*100)) if total>0 else "--"
    validity = int(round((valid/checked)*100)) if checked>0 else "--"
    return {"completeness": f"{completeness}%", "validity": f"{validity}%"}

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.header("Data Controls")
    dataset_type = st.selectbox("Dataset", options=list(DATASETS.keys()), format_func=lambda x: x.replace("_"," ").title())
    raw_df = load_or_generate(DATASETS[dataset_type], freq="H" if "hour" in dataset_type else "D")
    locations = extract_locations(raw_df, dataset_type)
    location = st.selectbox("Location", options=locations) if locations else st.text_input("Location")
    time_range = st.selectbox("Time Range", options=["1d","7d","30d","all"], format_func=lambda x: {"1d":"Last 24 Hours","7d":"Last 7 Days","30d":"Last 30 Days","all":"All"}[x])
    pollutants = st.multiselect("Pollutants", options=POLLUTANTS, default=["PM2.5"])
    apply = st.button("Apply Filters")

    st.markdown("### Data Quality")
    dq_placeholder = st.empty()
    st.markdown("---")
    st.write("Files loaded from:")
    st.code(str(DATASETS[dataset_type]))

# -------------------------
# Main layout
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
    st.write("If processed CSVs are not found the app uses generated sample data.")

# -------------------------
# Data filtering & rendering
# -------------------------
if (apply is False) and ("last_filtered" not in st.session_state):
    st.session_state.last_filtered = True

filtered = filter_by_location_and_range(raw_df, dataset_type, location, time_range)

dq = compute_data_quality(filtered)
dq_placeholder.info(f"Completeness: {dq['completeness']} • Validity: {dq['validity']}")
completeness_box.metric("Completeness", dq["completeness"])
validity_box.metric("Validity", dq["validity"])
status.info(f"Rendered ({len(filtered)} rows) — dataset: {dataset_type}")

filtered["Datetime"] = pd.to_datetime(filtered["Datetime"])

# Time series
with ts_card:
    if filtered.empty or len(pollutants) == 0:
        st.info("No data to plot")
    else:
        fig = go.Figure()
        for i,p in enumerate(pollutants):
            if p not in filtered.columns: continue
            fig.add_trace(go.Scatter(x=filtered["Datetime"], y=filtered[p],
                                     mode="lines", name=p,
                                     line=dict(color=COLORS[i % len(COLORS)])))
        fig.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=360, xaxis_title="Time", yaxis_title="Concentration (µg/m³)")
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
    if filtered.empty or len(pollutants)==0:
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
            if j <= i: continue
            corr_val = corr_df.loc[a,b]
            if pd.isna(corr_val): continue
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
st.caption("Design & layout inspired by uploaded dashboard (index.html).")
