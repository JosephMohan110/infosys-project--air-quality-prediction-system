# streamlit/upload_predictor_all_gases_readable.py
import re
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# COMPREHENSIVE AQI CALCULATION FOR ALL MAJOR POLLUTANTS & GASES
# =============================================================================

def calculate_pollutant_aqi(pollutant_name, concentration):
    """
    Calculate AQI for individual pollutants based on EPA standards
    Comprehensive coverage of all major air pollutants and gases
    """
    if pd.isna(concentration) or concentration <= 0:
        return np.nan
    
    concentration = float(concentration)
    poll_lower = pollutant_name.lower()
    
    # PM2.5 (μg/m³) - 24-hour average
    if any(x in poll_lower for x in ['pm2.5', 'pm25', 'pm2_5', 'fine particulate']):
        if concentration <= 12.0: return (50/12.0) * concentration
        elif concentration <= 35.4: return 51 + (49/23.4) * (concentration - 12.1)
        elif concentration <= 55.4: return 101 + (49/20.0) * (concentration - 35.5)
        elif concentration <= 150.4: return 151 + (49/95.0) * (concentration - 55.5)
        elif concentration <= 250.4: return 201 + (99/100.0) * (concentration - 150.5)
        else: return 301 + (199/149.6) * (concentration - 250.5)
    
    # PM10 (μg/m³) - 24-hour average
    elif any(x in poll_lower for x in ['pm10', 'pm_10', 'coarse particulate']):
        if concentration <= 54: return (50/54) * concentration
        elif concentration <= 154: return 51 + (49/100) * (concentration - 55)
        elif concentration <= 254: return 101 + (49/100) * (concentration - 155)
        elif concentration <= 354: return 151 + (49/100) * (concentration - 255)
        elif concentration <= 424: return 201 + (99/70) * (concentration - 355)
        else: return 301 + (199/175) * (concentration - 425)
    
    # Ozone O3 (ppb) - 8-hour average
    elif any(x in poll_lower for x in ['o3', 'ozone']):
        if concentration <= 54: return concentration
        elif concentration <= 70: return 55 + (45/16) * (concentration - 55)
        elif concentration <= 85: return 101 + (49/15) * (concentration - 71)
        elif concentration <= 105: return 151 + (49/20) * (concentration - 86)
        elif concentration <= 200: return 201 + (99/95) * (concentration - 106)
        else: return 301  # Very hazardous
    
    # Nitrogen Dioxide NO2 (ppb) - 1-hour average
    elif any(x in poll_lower for x in ['no2', 'nitrogen dioxide', 'nitrogen']):
        if concentration <= 53: return concentration
        elif concentration <= 100: return 54 + (46/47) * (concentration - 54)
        elif concentration <= 360: return 101 + (49/260) * (concentration - 101)
        elif concentration <= 649: return 151 + (49/289) * (concentration - 361)
        elif concentration <= 1249: return 201 + (99/600) * (concentration - 650)
        elif concentration <= 1649: return 301 + (99/400) * (concentration - 1250)
        else: return 401 + (99/350) * (concentration - 1650)
    
    # Sulfur Dioxide SO2 (ppb) - 1-hour average
    elif any(x in poll_lower for x in ['so2', 'sulfur dioxide', 'sulfur']):
        if concentration <= 35: return concentration * 50/35
        elif concentration <= 75: return 51 + (49/40) * (concentration - 36)
        elif concentration <= 185: return 101 + (49/110) * (concentration - 76)
        elif concentration <= 304: return 151 + (49/119) * (concentration - 186)
        elif concentration <= 604: return 201 + (99/300) * (concentration - 305)
        else: return 301 + (199/396) * (concentration - 605)
    
    # Carbon Monoxide CO (ppm) - 8-hour average
    elif any(x in poll_lower for x in ['co', 'carbon monoxide', 'carbon']):
        if concentration <= 4.4: return concentration * 50/4.4
        elif concentration <= 9.4: return 51 + (49/5) * (concentration - 4.5)
        elif concentration <= 12.4: return 101 + (49/3) * (concentration - 9.5)
        elif concentration <= 15.4: return 151 + (49/3) * (concentration - 12.5)
        elif concentration <= 30.4: return 201 + (99/15) * (concentration - 15.5)
        elif concentration <= 40.4: return 301 + (99/10) * (concentration - 30.5)
        else: return 401 + (99/9.6) * (concentration - 40.5)
    
    # Carbon Dioxide CO2 (ppm) - Not standard AQI but included for reference
    elif any(x in poll_lower for x in ['co2', 'carbon dioxide']):
        # CO2 doesn't have standard AQI, but we can create a relative scale
        if concentration <= 500: return 50 * (concentration / 500)
        elif concentration <= 1000: return 50 + 50 * ((concentration - 500) / 500)
        elif concentration <= 2000: return 100 + 50 * ((concentration - 1000) / 1000)
        elif concentration <= 5000: return 150 + 50 * ((concentration - 2000) / 3000)
        else: return 200 + 100 * min(1, (concentration - 5000) / 5000)
    
    # Ammonia NH3 (ppb)
    elif any(x in poll_lower for x in ['nh3', 'ammonia']):
        if concentration <= 200: return (50/200) * concentration
        elif concentration <= 400: return 51 + (49/200) * (concentration - 201)
        elif concentration <= 800: return 101 + (49/400) * (concentration - 401)
        elif concentration <= 1200: return 151 + (49/400) * (concentration - 801)
        elif concentration <= 1800: return 201 + (99/600) * (concentration - 1201)
        else: return 301 + (199/1800) * (concentration - 1801)
    
    # Volatile Organic Compounds VOC (ppb)
    elif any(x in poll_lower for x in ['voc', 'volatile organic', 'hydrocarbons']):
        if concentration <= 50: return (50/50) * concentration
        elif concentration <= 100: return 51 + (49/50) * (concentration - 51)
        elif concentration <= 200: return 101 + (49/100) * (concentration - 101)
        elif concentration <= 400: return 151 + (49/200) * (concentration - 201)
        elif concentration <= 800: return 201 + (99/400) * (concentration - 401)
        else: return 301 + (199/1200) * (concentration - 801)
    
    # Methane CH4 (ppm)
    elif any(x in poll_lower for x in ['ch4', 'methane']):
        if concentration <= 2: return (50/2) * concentration
        elif concentration <= 5: return 51 + (49/3) * (concentration - 2.1)
        elif concentration <= 10: return 101 + (49/5) * (concentration - 5.1)
        elif concentration <= 20: return 151 + (49/10) * (concentration - 10.1)
        elif concentration <= 50: return 201 + (99/30) * (concentration - 20.1)
        else: return 301 + (199/50) * (concentration - 50.1)
    
    # Benzene C6H6 (ppb)
    elif any(x in poll_lower for x in ['c6h6', 'benzene']):
        if concentration <= 5: return (50/5) * concentration
        elif concentration <= 10: return 51 + (49/5) * (concentration - 5.1)
        elif concentration <= 20: return 101 + (49/10) * (concentration - 10.1)
        elif concentration <= 30: return 151 + (49/10) * (concentration - 20.1)
        elif concentration <= 40: return 201 + (99/10) * (concentration - 30.1)
        else: return 301 + (199/60) * (concentration - 40.1)
    
    # Formaldehyde HCHO (ppb)
    elif any(x in poll_lower for x in ['hcho', 'formaldehyde']):
        if concentration <= 10: return (50/10) * concentration
        elif concentration <= 20: return 51 + (49/10) * (concentration - 10.1)
        elif concentration <= 30: return 101 + (49/10) * (concentration - 20.1)
        elif concentration <= 40: return 151 + (49/10) * (concentration - 30.1)
        elif concentration <= 50: return 201 + (99/10) * (concentration - 40.1)
        else: return 301 + (199/50) * (concentration - 50.1)
    
    # Hydrogen Sulfide H2S (ppb)
    elif any(x in poll_lower for x in ['h2s', 'hydrogen sulfide']):
        if concentration <= 5: return (50/5) * concentration
        elif concentration <= 10: return 51 + (49/5) * (concentration - 5.1)
        elif concentration <= 30: return 101 + (49/20) * (concentration - 10.1)
        elif concentration <= 50: return 151 + (49/20) * (concentration - 30.1)
        elif concentration <= 100: return 201 + (99/50) * (concentration - 50.1)
        else: return 301 + (199/100) * (concentration - 100.1)
    
    # Ozone O3 (1-hour) - Different scale for 1-hour average
    elif any(x in poll_lower for x in ['o3_1hr', 'ozone_1hr']):
        if concentration <= 124: return (100/124) * concentration
        elif concentration <= 164: return 101 + (49/40) * (concentration - 125)
        elif concentration <= 204: return 151 + (49/40) * (concentration - 165)
        elif concentration <= 404: return 201 + (99/200) * (concentration - 205)
        else: return 301 + (199/196) * (concentration - 405)
    
    # Default for unknown pollutants - use generic scale
    else:
        return min(concentration * 2, 500)  # Generic conversion

def get_pollutant_units(pollutant_name):
    """
    Get proper units for each pollutant
    """
    poll_lower = pollutant_name.lower()
    
    if any(x in poll_lower for x in ['pm2.5', 'pm25', 'pm2_5', 'pm10', 'pm_10']):
        return "μg/m³"
    elif any(x in poll_lower for x in ['o3', 'no2', 'so2', 'nh3', 'voc', 'c6h6', 'hcho', 'h2s']):
        return "ppb"
    elif any(x in poll_lower for x in ['co', 'ch4']):
        return "ppm"
    elif any(x in poll_lower for x in ['co2']):
        return "ppm"
    else:
        return "units"

def get_pollutant_description(pollutant_name):
    """
    Get description for each pollutant
    """
    poll_lower = pollutant_name.lower()
    
    descriptions = {
        'pm2.5': 'Fine inhalable particles, 2.5 micrometers and smaller',
        'pm10': 'Inhalable particles, 10 micrometers and smaller', 
        'o3': 'Ground-level ozone formed by chemical reactions',
        'no2': 'Reddish-brown toxic gas from combustion processes',
        'so2': 'Colorless gas with sharp odor from burning fossil fuels',
        'co': 'Colorless, odorless gas from incomplete combustion',
        'co2': 'Primary greenhouse gas from burning fossil fuels',
        'nh3': 'Colorless gas with pungent smell, from agriculture',
        'voc': 'Volatile Organic Compounds from industrial processes',
        'ch4': 'Primary component of natural gas, potent greenhouse gas',
        'c6h6': 'Carcinogenic compound from petroleum and combustion',
        'hcho': 'Colorless gas with strong odor, used in building materials',
        'h2s': 'Colorless gas with rotten egg odor, from industrial processes'
    }
    
    for key, desc in descriptions.items():
        if key in poll_lower:
            return desc
    return 'Air pollutant'

def aqi_category(aqi):
    """
    Categorize AQI values into quality levels with colors and emojis
    Returns category name and color code based on AQI value ranges
    """
    try:
        a = float(aqi)
    except:
        return "Unknown", "#cccccc"
    if a <= 50: return "Good 😊", "#00e400"
    if a <= 100: return "Satisfactory 🙂", "#ffff00"
    if a <= 200: return "Moderate 😐", "#ff7e00"
    if a <= 300: return "Poor 😷", "#ff0000"
    if a <= 400: return "Very Poor 🤢", "#99004c"
    return "Severe ☠️", "#7e0023"

def plot_aqi_gauge(aqi_val, title="🌬️ Current AQI"):
    """
    Create a visual gauge meter showing AQI value with color-coded ranges
    Uses Plotly gauge indicator for professional visualization
    """
    cat, color = aqi_category(aqi_val)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi_val,
        title={"text": f"{title} — {cat}", "font": {"size": 20, "color":"#111111"}},
        gauge={
            "axis": {"range": [0, 500], "tickcolor":"black", "tickwidth": 2},
            "bar": {"color": "#ff6600", "line": {"color": "black", "width": 2}},
            "steps": [
                {"range": [0, 50], "color": "#00e400"},      # green
                {"range": [51, 100], "color": "#ffff00"},    # yellow
                {"range": [101, 200], "color": "#ff7e00"},   # orange
                {"range": [201, 300], "color": "#ff0000"},   # red
                {"range": [301, 400], "color": "#99004c"},   # purple
                {"range": [401, 500], "color": "#7e0023"}    # maroon
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": aqi_val
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor="#f8f9fa",
        font={"color":"#111111", "family":"Arial"},
        margin=dict(t=60, b=40, l=40, r=40),
        height=300
    )
    return fig, color

def enhanced_parse_to_numeric(val):
    """
    Enhanced numeric parsing that handles various data formats in CSV files
    """
    if pd.isna(val) or val is None:
        return np.nan
    
    # If already numeric, return as is
    if isinstance(val, (int, float)):
        return float(val)
    
    # Convert to string and clean
    if isinstance(val, str):
        # Remove extra spaces and convert to lowercase
        val = val.strip().lower()
        
        # Handle empty strings
        if val in ['', 'null', 'none', 'nan', 'na', '-', '--', 'n/a']:
            return np.nan
        
        # Handle negative values and ranges
        if 'to' in val or '-' in val and not val.startswith('-'):
            # Try to extract the first number from ranges like "10-20" or "10 to 20"
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", val)
            if numbers:
                return float(numbers[0])
        
        # Handle values with units: "15.6 μg/m³" -> 15.6
        # Remove common units and extract numbers
        clean_val = re.sub(r'[^\d.-]', ' ', val)
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", clean_val)
        if numbers:
            return float(numbers[0])
        
        # Handle scientific notation
        try:
            return float(val)
        except:
            pass
    
    # Final attempt with pandas
    try:
        return pd.to_numeric(val, errors='coerce')
    except:
        return np.nan

def compute_aqi_from_pollutants(row, pollutant_cols):
    """
    Calculate AQI from multiple pollutant values using proper AQI formulas
    Uses the maximum AQI value among all pollutants (standard approach)
    """
    aqi_vals = []
    for col in pollutant_cols:
        if pd.notna(row[col]):
            concentration = enhanced_parse_to_numeric(row[col])
            if concentration > 0 and not pd.isna(concentration):  # Only consider valid concentrations
                pollutant_aqi = calculate_pollutant_aqi(col, concentration)
                if not pd.isna(pollutant_aqi):
                    aqi_vals.append(pollutant_aqi)
    
    return max(aqi_vals) if aqi_vals else np.nan

def show_current_aqi(df, station_name, pollutant_cols):
    """
    Display current AQI gauge and value
    Handles both pre-calculated AQI columns and computed AQI from pollutants
    """
    if "AQI" in df.columns:
        vals = df["AQI"].dropna()
        if vals.empty: 
            st.warning("⚠️ No AQI data available")
            return
        raw_val = vals.iloc[-1]
        aqi_val = enhanced_parse_to_numeric(raw_val)
    else:
        df["AQI_calc"] = df.apply(lambda row: compute_aqi_from_pollutants(row, pollutant_cols), axis=1)
        vals = df["AQI_calc"].dropna()
        if vals.empty: 
            st.warning("⚠️ Cannot calculate AQI - insufficient pollutant data")
            return
        aqi_val = vals.iloc[-1]

    fig, color = plot_aqi_gauge(aqi_val, title=f"🌆 {station_name} — Current AQI")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        f"<div style='padding:15px;border-radius:12px;background:{color};color:white;text-align:center;font-size:20px;font-weight:bold;'>"
        f"🌟 Current AQI: {aqi_val:.1f}</div>", 
        unsafe_allow_html=True
    )

# =============================================================================
# RANDOM FOREST REGRESSION IMPLEMENTATION SECTION
# =============================================================================

def create_time_features(dates):
    """
    Create comprehensive time-based features for Random Forest
    """
    features = []
    for date in dates:
        if isinstance(date, (int, np.integer)):
            dt = datetime.fromordinal(date)
        else:
            dt = pd.to_datetime(date)
        
        # Convert to pandas Timestamp for proper attribute access
        ts = pd.Timestamp(dt)
        
        features.append([
            ts.year,
            ts.month,
            ts.day,
            ts.dayofweek,  # Monday=0, Sunday=6
            ts.dayofyear,
            ts.quarter,
            int(ts.is_month_start),
            int(ts.is_month_end),
            np.sin(2 * np.pi * ts.dayofyear / 365),
            np.cos(2 * np.pi * ts.dayofyear / 365),
            np.sin(2 * np.pi * ts.month / 12),
            np.cos(2 * np.pi * ts.month / 12),
        ])
    
    return np.array(features)

def random_forest_predictor(X_dates, y_values, target_date):
    """
    Random Forest prediction with proper time series features
    """
    if len(X_dates) < 5:
        return np.mean(y_values) if len(y_values) > 0 else np.nan
    
    try:
        # Create features for training data
        X_train = create_time_features(X_dates)
        y_train = np.array(y_values)
        
        # Remove any NaN values
        mask = ~np.isnan(y_train)
        X_train = X_train[mask]
        y_train = y_train[mask]
        
        if len(X_train) < 5:
            return np.mean(y_train) if len(y_train) > 0 else np.nan
        
        # Train Random Forest
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=3,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Create features for target date
        X_pred = create_time_features([target_date])
        
        # Make prediction
        prediction = model.predict(X_pred)[0]
        
        # Ensure prediction is reasonable
        historical_mean = np.mean(y_train)
        historical_std = np.std(y_train)
        
        # Apply bounds based on historical data
        lower_bound = max(0, historical_mean - 2 * historical_std)
        upper_bound = historical_mean + 2 * historical_std
        
        prediction = np.clip(prediction, lower_bound, upper_bound)
        
        return float(prediction)
        
    except Exception as e:
        st.warning(f"Random Forest prediction failed: {str(e)}")
        return np.mean(y_values) if len(y_values) > 0 else np.nan

def predict_pollutants(df_filtered, pollutant_cols, target_date):
    """
    IMPROVED PREDICTION FUNCTION WITH RANDOM FOREST
    """
    predictions = {}
    prediction_notes = {}
    prediction_data = {}  # Store data used for each prediction
    
    # Convert target date to ordinal for compatibility
    target_ordinal = pd.to_datetime(target_date).toordinal()
    
    for col in pollutant_cols:
        try:
            # Create clean dataset for this specific pollutant
            temp_df = df_filtered[['Date', col]].copy()
            temp_df = temp_df.dropna(subset=[col, 'Date'])
            
            # Use enhanced parsing for numeric conversion
            temp_df[col] = temp_df[col].apply(enhanced_parse_to_numeric)
            temp_df = temp_df[temp_df[col] > 0]  # Remove zero/negative values
            
            if len(temp_df) < 3:
                predictions[col] = np.nan
                prediction_notes[col] = f"Insufficient historical data ({len(temp_df)} points)"
                prediction_data[col] = {"data_points": len(temp_df), "available_data": "No sufficient data"}
                continue
            
            # Prepare features and target
            X_dates = temp_df['Date'].map(lambda x: x.toordinal() if hasattr(x, 'toordinal') else pd.to_datetime(x).toordinal()).values
            y_values = temp_df[col].values
            
            # Store data used for prediction with safe key access
            prediction_data[col] = {
                "data_points": len(temp_df),
                "date_range": f"{temp_df['Date'].min().strftime('%Y-%m-%d')} to {temp_df['Date'].max().strftime('%Y-%m-%d')}" if len(temp_df) > 0 else "No data",
                "values_range": f"{y_values.min():.2f} to {y_values.max():.2f}" if len(y_values) > 0 else "No values",
                "mean_value": f"{y_values.mean():.2f}" if len(y_values) > 0 else "No data",
                "recent_values": y_values[-5:].tolist() if len(y_values) >= 5 else y_values.tolist()
            }
            
            # Use Random Forest for prediction
            pred_val = random_forest_predictor(X_dates, y_values, target_ordinal)
            
            if pd.isna(pred_val):
                # Fallback: use recent average
                recent_avg = np.mean(y_values[-5:]) if len(y_values) >= 5 else np.mean(y_values)
                pred_val = recent_avg
                prediction_notes[col] = "Used recent average (RF failed)"
            else:
                prediction_notes[col] = "Random Forest prediction"
            
            # Final sanity check
            historical_avg = np.mean(y_values)
            if pred_val > historical_avg * 10:  # Unreasonable prediction
                pred_val = historical_avg * 1.5  # Conservative estimate
                prediction_notes[col] = "Adjusted unreasonable prediction"
            elif pred_val < 0:
                pred_val = max(0.1, historical_avg * 0.8)
                prediction_notes[col] = "Adjusted negative prediction"
            
            predictions[col] = round(float(pred_val), 2)
                
        except Exception as e:
            # Fallback to simple prediction
            try:
                temp_df = df_filtered[['Date', col]].copy().dropna()
                if len(temp_df) >= 2:
                    recent_avg = temp_df[col].tail(5).mean()
                    predictions[col] = round(float(recent_avg), 2)
                    prediction_notes[col] = "Used recent average (fallback)"
                    prediction_data[col] = {
                        "data_points": len(temp_df),
                        "date_range": f"{temp_df['Date'].min().strftime('%Y-%m-%d')} to {temp_df['Date'].max().strftime('%Y-%m-%d')}" if len(temp_df) > 0 else "No data",
                        "method": "Recent average fallback"
                    }
                else:
                    predictions[col] = np.nan
                    prediction_notes[col] = f"Prediction error: {str(e)}"
                    prediction_data[col] = {"error": str(e)}
            except:
                predictions[col] = np.nan
                prediction_notes[col] = f"Prediction failed: {str(e)}"
                prediction_data[col] = {"error": str(e)}
    
    return predictions, prediction_notes, prediction_data

def predict_next_7_days(df_filtered, pollutant_cols):
    """
    IMPROVED 7-DAY PREDICTION WITH RANDOM FOREST
    """
    next_7_days = []
    today = datetime.now().date()
    
    for i in range(1, 8):  # Next 7 days
        target_date = today + timedelta(days=i)
        
        # Get predictions for all pollutants
        predictions, _, _ = predict_pollutants(df_filtered, pollutant_cols, target_date)
        
        # Calculate AQI from predicted pollutants
        aqi_vals = []
        for poll, concentration in predictions.items():
            if not pd.isna(concentration) and concentration > 0:
                poll_aqi = calculate_pollutant_aqi(poll, concentration)
                if not pd.isna(poll_aqi):
                    aqi_vals.append(poll_aqi)
        
        if aqi_vals:
            predicted_aqi = max(aqi_vals)
            predicted_aqi = min(500, predicted_aqi)  # Cap at 500
            category, color = aqi_category(predicted_aqi)
        else:
            predicted_aqi = np.nan
            category, color = "Unknown", "#cccccc"
        
        next_7_days.append({
            'Date': target_date,
            'Day': target_date.strftime('%a'),
            'Full_Date': target_date.strftime('%d %b'),
            'AQI': predicted_aqi,
            'Category': category,
            'Color': color
        })
    
    return pd.DataFrame(next_7_days)

def get_rolling_7day_forecast(df_filtered, pollutant_cols):
    """
    Get rolling 7-day forecast that automatically updates
    Removes past dates and adds future dates to maintain 7-day window
    """
    today = datetime.now().date()
    
    # Generate predictions for next 7 days
    forecast_df = predict_next_7_days(df_filtered, pollutant_cols)
    
    return forecast_df

# =============================================================================
# STREAMLIT UI CONFIGURATION SECTION
# =============================================================================

# Configure the Streamlit page
st.set_page_config(page_title="🌬️ AirAware — All Pollutants Viewer", layout="wide")
st.markdown("## 🌈 AirAware — Comprehensive Air Quality Analysis", unsafe_allow_html=True)
st.markdown("<hr style='border:2px solid #4da6ff'>", unsafe_allow_html=True)

# =============================================================================
# ENHANCED FILE UPLOAD AND DATA PROCESSING SECTION
# =============================================================================

# File uploader widget
uploaded = st.file_uploader("📂 Upload CSV file", type=['csv'])
if not uploaded:
    st.stop()

# Initialize session state for data processing results
if 'processing_results' not in st.session_state:
    st.session_state.processing_results = {}

try:
    # Try different encodings to handle various CSV formats
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(uploaded, encoding=encoding)
            st.session_state.processing_results['encoding'] = f"Successfully loaded {len(df)} records using {encoding} encoding"
            st.session_state.processing_results['original_dataset'] = f"Original dataset: {len(df)} rows × {len(df.columns)} columns"
            break
        except UnicodeDecodeError:
            uploaded.seek(0)  # Reset file pointer
            continue
    
    if df is None:
        # Final attempt with error handling
        uploaded.seek(0)
        df = pd.read_csv(uploaded, encoding='utf-8', errors='ignore')
        st.session_state.processing_results['encoding'] = f"Successfully loaded {len(df)} records with error handling"
        st.session_state.processing_results['original_dataset'] = f"Original dataset: {len(df)} rows × {len(df.columns)} columns"
        
except Exception as e:
    st.error(f"❌ Error reading CSV: {e}")
    st.stop()

# Date column detection and processing
date_cols = df.filter(regex='date|ds|time|datetime', axis=1).columns
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
elif len(date_cols) > 0:
    df["Date"] = pd.to_datetime(df[date_cols[0]], errors="coerce")
else:
    # If no date column found, create one from index
    df["Date"] = pd.to_datetime(df.index, errors="coerce")
    st.session_state.processing_results['date_warning'] = "⚠️ No date column found. Using index as date."

# Remove rows with invalid dates
initial_count = len(df)
df = df[df["Date"].notna()]
if len(df) < initial_count:
    st.session_state.processing_results['date_cleaning'] = f"⚠️ Removed {initial_count - len(df)} rows with invalid dates"

# Detect City/Station column for filtering
city_col = None
for c in df.columns:
    if any(x in c.lower() for x in ["city", "station", "location", "site", "area"]):
        city_col = c
        break

if city_col:
    st.session_state.processing_results['location_column'] = f"📍 Using '{city_col}' as location column"
else:
    st.session_state.processing_results['location_warning'] = "📍 No city/station column detected. Showing all data together."

# Enhanced pollutant detection for all gases - MORE INCLUSIVE APPROACH
common_pollutants = [
    'pm2.5', 'pm25', 'pm2_5', 'pm10', 'pm_10',  # Particulate Matter
    'o3', 'ozone',  # Ozone
    'no2', 'nitrogen',  # Nitrogen Dioxide
    'so2', 'sulfur',  # Sulfur Dioxide  
    'co', 'carbon',  # Carbon Monoxide
    'co2', 'carbon dioxide',  # Carbon Dioxide
    'nh3', 'ammonia',  # Ammonia
    'voc', 'volatile',  # Volatile Organic Compounds
    'ch4', 'methane',  # Methane
    'c6h6', 'benzene',  # Benzene
    'hcho', 'formaldehyde',  # Formaldehyde
    'h2s', 'hydrogen sulfide',  # Hydrogen Sulfide
    'aqi', 'quality', 'pollution'  # General air quality indicators
]

# EXCLUDE only very specific columns, be more inclusive
exclude_cols = ["date", "time", "index", "id", "no", "code", "row", "serial"]

pollutant_cols = []
pollutant_details = {}

for c in df.columns:
    c_lower = c.lower()
    
    # Skip if it's in exclude list or is date/city column
    if any(exclude in c_lower for exclude in exclude_cols):
        continue
    if c == city_col or c == "Date":
        continue
    
    # Check if it matches known pollutants
    is_known_pollutant = any(poll in c_lower for poll in common_pollutants)
    
    # Check data type and content
    has_data = df[c].notna().sum() > 0
    
    # Enhanced data detection
    if has_data:
        # Try to convert to numeric to check if it contains numeric data
        sample_data = df[c].dropna().head(10)
        numeric_count = 0
        total_count = len(sample_data)
        
        for val in sample_data:
            parsed_val = enhanced_parse_to_numeric(val)
            if not pd.isna(parsed_val) and parsed_val > 0:
                numeric_count += 1
        
        # If most samples are numeric or it's a known pollutant, include it
        if (numeric_count / total_count > 0.3) or is_known_pollutant:
            pollutant_cols.append(c)
            
            # Store details about this pollutant
            pollutant_details[c] = {
                'known_pollutant': is_known_pollutant,
                'numeric_data': numeric_count / total_count,
                'total_records': len(df),
                'non_null_records': df[c].notna().sum(),
                'sample_values': sample_data.tolist()[:3]
            }

if not pollutant_cols:
    st.error("""
    ❌ No pollutant columns detected! 
    
    **Possible reasons:**
    - Your CSV might have different column names
    - Data might be in non-numeric format
    - File might be empty or corrupted
    
    **Please check:**
    - Column names contain pollutant names (PM2.5, O3, NO2, etc.)
    - Data contains numeric values
    - File is properly formatted as CSV
    """)
    
    # Show all columns for debugging
    st.write("### 📊 All Columns in Your CSV:")
    for col in df.columns:
        st.write(f"- **{col}**: {df[col].dtype}, {df[col].notna().sum()} non-null values")
    
    st.stop()

# Store detection results
st.session_state.processing_results['pollutant_detection'] = f"✅ Detected {len(pollutant_cols)} pollutant columns"
total_non_null = sum([df[col].notna().sum() for col in pollutant_cols])
st.session_state.processing_results['data_points'] = f"📊 Total pollutant data points: {total_non_null}"

# Enhanced data cleaning - DON'T REMOVE ROWS TOO AGGRESSIVELY
initial_row_count = len(df)

# Only remove rows where ALL pollutants are NaN (be more conservative)
df_clean = df.copy()
rows_before = len(df_clean)
df_clean = df_clean[df_clean[pollutant_cols].notna().any(axis=1)]
rows_after = len(df_clean)

if rows_after < rows_before:
    st.session_state.processing_results['data_cleaning'] = f"🧹 Kept {rows_after} rows with at least one pollutant value (removed {rows_before - rows_after} completely empty rows)"

# Use the cleaned dataframe
df = df_clean

# Enhanced data conversion for all pollutant columns
conversion_results = []
for col in pollutant_cols:
    original_non_null = df[col].notna().sum()
    
    # Convert using enhanced parser
    df[col] = df[col].apply(enhanced_parse_to_numeric)
    
    converted_non_null = df[col].notna().sum()
    conversion_rate = (converted_non_null / original_non_null * 100) if original_non_null > 0 else 0
    
    conversion_results.append({
        'Pollutant': col,
        'Original': original_non_null,
        'Converted': converted_non_null,
        'Success Rate': f"{conversion_rate:.1f}%"
    })

# Store conversion results
st.session_state.processing_results['conversion_data'] = conversion_results

# Sort by date to ensure proper time series
df = df.sort_values('Date')

# =============================================================================
# USER INTERFACE CONTROLS SECTION
# =============================================================================

# Create dropdown selection controls
cities = ["All"] + (sorted(df[city_col].dropna().unique()) if city_col else [])
pollutants = ["All"] + pollutant_cols

col1, col2 = st.columns([1.5, 1.5])
with col1:
    station = st.selectbox("🏙️ Select City/Station", options=cities)
with col2:
    selected_pollutant = st.selectbox("💨 Select Pollutant", options=pollutants)

# Filter data based on city selection
df_filtered = df.copy()
if city_col and station != "All":
    df_filtered = df_filtered[df_filtered[city_col] == station]
    st.success(f"📊 Showing data for: {station}")

# Show basic data info (minimal)
st.info(f"📈 Available data: {len(df_filtered)} records with {len(pollutant_cols)} pollutants")

# =============================================================================
# CURRENT AQI DISPLAY SECTION
# =============================================================================

st.markdown("### 🌟 Current Air Quality (AQI Gauge)")
show_current_aqi(df_filtered, station or "Site", pollutant_cols)

# =============================================================================
# POLLUTANT INFORMATION SECTION
# =============================================================================

st.markdown("### 🔍 Pollutant Information")

# Show pollutant details in expandable sections
if selected_pollutant != "All":
    with st.expander(f"📖 About {selected_pollutant}"):
        units = get_pollutant_units(selected_pollutant)
        description = get_pollutant_description(selected_pollutant)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Units", units)
        with col2:
            if not df_filtered.empty and df_filtered[selected_pollutant].notna().sum() > 0:
                current_val = df_filtered[selected_pollutant].iloc[-1]
                st.metric("Current Value", f"{current_val:.2f} {units}")
            else:
                st.metric("Current Value", "No data")
        with col3:
            if not df_filtered.empty and df_filtered[selected_pollutant].notna().sum() > 0:
                current_val = df_filtered[selected_pollutant].iloc[-1]
                aqi_impact = calculate_pollutant_aqi(selected_pollutant, current_val)
                st.metric("AQI Impact", f"{aqi_impact:.1f}")
            else:
                st.metric("AQI Impact", "N/A")
        
        st.write(f"**Description**: {description}")
        
        # Show data statistics
        if df_filtered[selected_pollutant].notna().sum() > 0:
            st.write("**Data Statistics**:")
            stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
            with stats_col1:
                st.metric("Mean", f"{df_filtered[selected_pollutant].mean():.2f}")
            with stats_col2:
                st.metric("Max", f"{df_filtered[selected_pollutant].max():.2f}")
            with stats_col3:
                st.metric("Min", f"{df_filtered[selected_pollutant].min():.2f}")
            with stats_col4:
                st.metric("Std Dev", f"{df_filtered[selected_pollutant].std():.2f}")

# =============================================================================
# POLLUTANT GRAPHS VISUALIZATION SECTION
# =============================================================================

st.markdown("### 📈 Pollutant Graphs")
plot_bg = "#f8f9fa"  # light background for charts

if selected_pollutant == "All":
    # Show all pollutants together for a specific city
    if city_col and station != "All":
        # Filter out pollutants with no data for this city
        available_pollutants = []
        for col in pollutant_cols:
            if df_filtered[col].notna().sum() > 0:
                available_pollutants.append(col)
        
        if available_pollutants:
            # Show top 8 pollutants to avoid clutter
            display_pollutants = available_pollutants[:8]
            fig = px.line(
                df_filtered, x="Date", y=display_pollutants,
                title=f"🌈 Multiple Pollutant Trends in {station}",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig.update_traces(mode="lines+markers", line=dict(width=2), marker=dict(size=4))
            fig.update_layout(
                paper_bgcolor=plot_bg, plot_bgcolor=plot_bg, font_color="#111111",
                title_font=dict(size=22, color="#111111", family="Arial"),
                xaxis_title_font=dict(size=16, color="#111111"),
                yaxis_title_font=dict(size=16, color="#111111"),
                xaxis_tickangle=-45,
                xaxis_tickfont=dict(size=12, color="#111111"),
                yaxis_tickfont=dict(size=12, color="#111111"),
                legend=dict(bgcolor="white", bordercolor="black", borderwidth=1),
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            if len(available_pollutants) > 8:
                st.info(f"📊 Showing first 8 of {len(available_pollutants)} pollutants. Select specific pollutant for detailed view.")
        else:
            st.warning("⚠️ No pollutant data available for the selected city.")
    else:
        st.warning("⚠️ Select a specific city to view all pollutants together.")
else:
    # Show individual pollutant trends
    if df_filtered[selected_pollutant].notna().sum() > 0:
        units = get_pollutant_units(selected_pollutant)
        
        if city_col and station == "All":
            fig = px.line(
                df_filtered, x="Date", y=selected_pollutant, color=city_col,
                title=f"💨 {selected_pollutant} Levels Across Cities ({units})",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
        else:
            fig = px.line(
                df_filtered, x="Date", y=selected_pollutant,
                title=f"💨 {selected_pollutant} Trend in {station if station != 'All' else 'All Cities'} ({units})",
                color_discrete_sequence=['#ff6b6b']
            )
        fig.update_traces(mode="lines+markers", line=dict(width=3), marker=dict(size=6))
        fig.update_layout(
            paper_bgcolor=plot_bg, plot_bgcolor=plot_bg, font_color="#111111",
            title_font=dict(size=22, color="#111111", family="Arial"),
            xaxis_title_font=dict(size=16, color="#111111"),
            yaxis_title_font=dict(size=16, color="#111111"),
            xaxis_tickangle=-45,
            xaxis_tickfont=dict(size=12, color="#111111"),
            yaxis_tickfont=dict(size=12, color="#111111"),
            legend=dict(bgcolor="white", bordercolor="black", borderwidth=1),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"⚠️ No data available for {selected_pollutant} in the selected location.")

# =============================================================================
# CALENDAR-BASED PREDICTION INTERFACE SECTION
# =============================================================================

st.markdown("### 📅 Calendar-based Prediction")
st.info("🔮 Select any future date (up to 10 years) to predict pollutant levels using Random Forest")

col1, col2 = st.columns([2, 1])
with col1:
    target_date = st.date_input(
        "🗓️ Select a future date to predict pollutant levels",
        min_value=datetime.now().date() + timedelta(days=1),
        max_value=datetime.now().date() + timedelta(days=3650),  # 10 years max
        value=datetime.now().date() + timedelta(days=30),
        help="Select any future date for prediction (up to 10 years)"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🚀 Predict Now", type="primary")

if target_date and predict_btn:
    if len(df_filtered) < 3:
        st.error("❌ Need at least 3 data points for prediction")
    else:
        # Calculate how far in the future we're predicting
        days_in_future = (target_date - datetime.now().date()).days
        years_in_future = days_in_future / 365.25
        
        with st.spinner(f"🤖 Generating Random Forest predictions for {target_date} ({years_in_future:.1f} years from now)..."):
            predictions, prediction_notes, prediction_data = predict_pollutants(df_filtered, pollutant_cols, target_date)
        
        # Filter out None predictions
        valid_predictions = {k: v for k, v in predictions.items() if not pd.isna(v)}
        
        if valid_predictions:
            st.subheader(f"🎯 Predicted Pollutant Levels on {target_date}")
            
            # Display prediction timeframe info
            if years_in_future > 2:
                st.warning(f"⚠️ Long-term prediction ({years_in_future:.1f} years from now) - Using Random Forest")
            elif years_in_future > 1:
                st.info(f"📅 Medium-term prediction ({years_in_future:.1f} years from now) - Using Random Forest")
            else:
                st.success(f"📅 Short-term prediction ({days_in_future} days from now) - Using Random Forest")
            
            # Calculate predicted AQI
            aqi_vals = []
            for poll, concentration in valid_predictions.items():
                poll_aqi = calculate_pollutant_aqi(poll, concentration)
                if not pd.isna(poll_aqi):
                    aqi_vals.append(poll_aqi)
            
            predicted_aqi = max(aqi_vals) if aqi_vals else np.nan
            
            # Create results dataframe for display
            results_data = []
            for poll, value in valid_predictions.items():
                poll_aqi = calculate_pollutant_aqi(poll, value)
                units = get_pollutant_units(poll)
                results_data.append({
                    'Pollutant': poll,
                    'Units': units,
                    'Predicted Concentration': value,
                    'Predicted AQI': round(poll_aqi, 1) if not pd.isna(poll_aqi) else "N/A",
                    'Notes': prediction_notes.get(poll, 'Random Forest prediction')
                })
            
            results_df = pd.DataFrame(results_data)
            
            # Display AQI summary
            if not pd.isna(predicted_aqi):
                category, color = aqi_category(predicted_aqi)
                st.markdown(
                    f"<div style='padding:15px;border-radius:12px;background:{color};color:black;text-align:center;font-size:20px;font-weight:bold;'>"
                    f"📊 Overall Predicted AQI: {predicted_aqi:.1f} — {category}</div>", 
                    unsafe_allow_html=True
                )
            
            # Show results table
            st.dataframe(results_df, width='stretch')
            
            # Visualize predictions as bar chart
            if not results_df.empty:
                fig_pred = px.bar(
                    results_df, x='Pollutant', y='Predicted Concentration',
                    color='Predicted AQI', 
                    color_continuous_scale="Viridis",
                    text='Predicted Concentration', 
                    title=f"📊 Predicted Pollutant Concentrations on {target_date}",
                    hover_data=['Units', 'Predicted AQI', 'Notes']
                )
                fig_pred.update_traces(
                    texttemplate='%{text:.2f}', 
                    textposition='outside',
                    marker_line_color='black', 
                    marker_line_width=1
                )
                fig_pred.update_layout(
                    paper_bgcolor=plot_bg, 
                    plot_bgcolor=plot_bg, 
                    font_color="#111111",
                    title_font=dict(size=22, color="#111111", family="Arial"),
                    xaxis_title_font=dict(size=16, color="#111111"),
                    yaxis_title_font=dict(size=16, color="#111111"),
                    xaxis_tickfont=dict(size=12, color="#111111"),
                    yaxis_tickfont=dict(size=12, color="#111111"),
                    coloraxis_colorbar=dict(title="AQI Impact"),
                    height=500
                )
                st.plotly_chart(fig_pred, use_container_width=True)
            
            # Show data used for predictions
            with st.expander("📊 Data Used for Predictions"):
                st.markdown("### 📈 Prediction Data Sources")
                st.info("This shows the historical data used to train the Random Forest model for each pollutant:")
                
                for poll, data_info in prediction_data.items():
                    if 'error' not in data_info:
                        st.markdown(f"**{poll}**")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Data Points", data_info.get('data_points', 'N/A'))
                        with col2:
                            st.metric("Date Range", data_info.get('date_range', 'No data'))
                        with col3:
                            st.metric("Value Range", data_info.get('values_range', 'No values'))
                        
                        if 'mean_value' in data_info:
                            st.metric("Historical Mean", data_info['mean_value'])
                        
                        if 'recent_values' in data_info and len(data_info['recent_values']) > 0:
                            st.write(f"Recent values: {[round(x, 2) for x in data_info['recent_values']]}")
                        
                        st.markdown("---")
                    else:
                        st.warning(f"**{poll}**: {data_info.get('error', 'Unknown error')}")
            
        else:
            st.error("❌ No valid predictions could be generated. Please check if you have sufficient historical data.")
        
elif target_date:
    st.info("👆 Click the 'Predict Now' button to generate Random Forest predictions")

# =============================================================================
# 7-DAY AQI FORECAST SECTION WITH ROLLING UPDATES
# =============================================================================

st.markdown("### 🌟 7-Day AQI Forecast (Auto-Updating)")

# Check if we have enough data for predictions
if len(df_filtered) < 3:
    st.warning("⚠️ Insufficient data for 7-day forecast. Need at least 3 data points.")
else:
    # Generate 7-day predictions automatically using RANDOM FOREST
    with st.spinner("🌤️ Generating 7-day forecast using Random Forest..."):
        forecast_df = get_rolling_7day_forecast(df_filtered, pollutant_cols)

    if not forecast_df.empty:
        # Display current date info
        today = datetime.now().date()
        st.info(f"📅 **Today**: {today.strftime('%A, %d %B %Y')} - Forecast updates automatically")
        
        # Display 7-day forecast as visual cards
        st.markdown("#### 📅 Daily AQI Forecast")
        
        # Create columns for the 7 days
        cols = st.columns(7)
        
        for idx, (_, day_data) in enumerate(forecast_df.iterrows()):
            with cols[idx]:
                if pd.isna(day_data['AQI']):
                    # No prediction available case
                    st.markdown(
                        f"<div style='padding:10px;border-radius:12px;background:#f8f9fa;text-align:center;border:2px solid #dee2e6;min-height:160px;box-shadow:0 4px 6px rgba(0,0,0,0.1);'>"
                        f"<div style='font-weight:bold;font-size:14px;color:#6c757d;'>{day_data['Full_Date']}</div>"
                        f"<div style='font-size:12px;color:#6c757d;margin-bottom:8px;'>{day_data['Day']}</div>"
                        f"<div style='margin:12px 0;font-size:24px;'>❓</div>"
                        f"<div style='font-size:12px;color:#6c757d;'>No Data</div>"
                        f"</div>", 
                        unsafe_allow_html=True
                    )
                else:
                    # Prediction available case with color coding
                    days_from_today = (day_data['Date'] - today).days
                    day_label = "Tomorrow" if days_from_today == 1 else f"In {days_from_today} days"
                    
                    emoji = "😊" if "Good" in day_data['Category'] else \
                           "🙂" if "Satisfactory" in day_data['Category'] else \
                           "😐" if "Moderate" in day_data['Category'] else \
                           "😷" if "Poor" in day_data['Category'] else \
                           "🤢" if "Very Poor" in day_data['Category'] else "☠️"
                    
                    text_color = "white" if day_data['AQI'] > 200 else "#111111"
                    
                    st.markdown(
                        f"<div style='padding:12px;border-radius:12px;background:{day_data['Color']};text-align:center;color:{text_color};border:3px solid {day_data['Color']}80;min-height:160px;box-shadow:0 4px 8px rgba(0,0,0,0.2);'>"
                        f"<div style='font-weight:bold;font-size:15px;'>{day_data['Full_Date']}</div>"
                        f"<div style='font-size:13px;opacity:0.9;margin-bottom:5px;'>📅 {day_data['Day']}</div>"
                        f"<div style='font-size:11px;opacity:0.8;margin-bottom:8px;'>{day_label}</div>"
                        f"<div style='margin:8px 0;font-size:28px;'>{emoji}</div>"
                        f"<div style='font-weight:bold;font-size:18px;'>AQI {day_data['AQI']:.0f}</div>"
                        f"<div style='font-size:12px;opacity:0.9;margin-top:5px;'>{day_data['Category'].split()[0]}</div>"
                        f"</div>", 
                        unsafe_allow_html=True
                    )
        
        # Show weekly summary statistics
        st.markdown("#### 📊 Week Summary")
        valid_predictions = forecast_df.dropna(subset=['AQI'])
        if not valid_predictions.empty:
            avg_aqi = valid_predictions['AQI'].mean()
            best_day = valid_predictions.loc[valid_predictions['AQI'].idxmin()]
            worst_day = valid_predictions.loc[valid_predictions['AQI'].idxmax()]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📈 Average AQI", f"{avg_aqi:.1f}", delta_color="off")
            with col2:
                st.metric("🎯 Best Day", f"{best_day['Day']} ({best_day['AQI']:.0f})", delta_color="off")
            with col3:
                st.metric("⚠️ Worst Day", f"{worst_day['Day']} ({worst_day['AQI']:.0f})", delta_color="off")
            with col4:
                days_with_data = len(valid_predictions)
                st.metric("📅 Days Predicted", f"{days_with_data}/7", delta_color="off")
        
        # Show detailed forecast table with additional info
        with st.expander("📋 Detailed 7-Day Forecast Table"):
            st.markdown("### 📊 Complete Forecast Details")
            
            # Create enhanced display dataframe
            display_df = forecast_df.copy()
            display_df['Date'] = display_df['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
            display_df['AQI'] = display_df['AQI'].round(1)
            
            # Add days from today column
            display_df['Days From Today'] = [(pd.to_datetime(row['Date']) - pd.to_datetime(today)).days for _, row in display_df.iterrows()]
            
            # Reorder columns for better display
            display_df = display_df[['Date', 'Day', 'Days From Today', 'AQI', 'Category']]
            
            st.dataframe(display_df, width='stretch')
            
            # Forecast notes
            st.markdown("---")
            st.markdown("#### 📝 Forecast Notes")
            st.info("""
            - **Forecast Period**: Next 7 days from today
            - **Update Frequency**: Forecast automatically updates daily
            - **Methodology**: Random Forest regression with time-series features
            - **Data Used**: Historical pollutant data from your uploaded dataset
            - **Accuracy**: Better for short-term predictions (1-3 days)
            """)
        
        # Show forecast trend visualization
        with st.expander("📈 Forecast Trend Visualization"):
            st.markdown("### 📊 AQI Forecast Trend")
            
            # Create trend line chart
            trend_df = forecast_df.dropna(subset=['AQI']).copy()
            if not trend_df.empty:
                fig_trend = px.line(
                    trend_df, x='Date', y='AQI',
                    title='📈 7-Day AQI Forecast Trend',
                    markers=True,
                    line_shape='linear'
                )
                
                # Add colored background based on AQI categories
                for i, row in trend_df.iterrows():
                    fig_trend.add_hrect(
                        y0=row['AQI']-5, y1=row['AQI']+5,
                        fillcolor=row['Color'], opacity=0.2,
                        line_width=0
                    )
                
                fig_trend.update_traces(
                    line=dict(width=4, color='#2c3e50'),
                    marker=dict(size=8, color='#2c3e50')
                )
                
                fig_trend.update_layout(
                    paper_bgcolor=plot_bg,
                    plot_bgcolor=plot_bg,
                    font_color="#111111",
                    title_font=dict(size=22, color="#111111", family="Arial"),
                    xaxis_title="Date",
                    yaxis_title="AQI",
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.warning("No valid forecast data available for trend visualization.")

# =============================================================================
# DATA PREVIEW SECTION
# =============================================================================

with st.expander("👀 Preview Data Used for Predictions"):
    st.markdown("### 📊 Data Available for Predictions")
    
    # Show data summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(df_filtered))
    with col2:
        st.metric("Date Range", f"{df_filtered['Date'].min().strftime('%Y-%m-%d')} to {df_filtered['Date'].max().strftime('%Y-%m-%d')}")
    with col3:
        st.metric("Time Span", f"{(df_filtered['Date'].max() - df_filtered['Date'].min()).days} days")
    
    # Show pollutant data availability
    st.markdown("#### 🔍 Pollutant Data Availability")
    availability_data = []
    for col in pollutant_cols:
        non_null = df_filtered[col].notna().sum()
        total = len(df_filtered)
        completeness = (non_null / total * 100) if total > 0 else 0
        
        if non_null > 0:
            avg_val = df_filtered[col].mean()
            units = get_pollutant_units(col)
        else:
            avg_val = np.nan
            units = "N/A"
        
        availability_data.append({
            'Pollutant': col,
            'Available Records': non_null,
            'Completeness': f"{completeness:.1f}%",
            'Average Value': f"{avg_val:.2f} {units}" if not pd.isna(avg_val) else "No data",
            'Suitable for Prediction': "✅" if non_null >= 3 else "❌"
        })

    availability_df = pd.DataFrame(availability_data)
    st.dataframe(availability_df, width='stretch')
    
    # Show sample of the data
    st.markdown("#### 📋 Sample of Available Data")
    preview_cols = ['Date']
    if city_col:
        preview_cols.append(city_col)
    preview_cols.extend(pollutant_cols)
    
    sample_df = df_filtered[preview_cols].copy()
    if len(sample_df) > 10:
        st.dataframe(sample_df.head(10), width='stretch')
        st.info(f"Showing first 10 of {len(sample_df)} records")
    else:
        st.dataframe(sample_df, width='stretch')

# =============================================================================
# FOOTER SECTION
# =============================================================================

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 20px;'>"
    "🌍 <b>AirAware</b> - Comprehensive Air Quality Analysis & Prediction System<br>"
    "<small>Enhanced data processing | All major pollutants | Random Forest Predictions | Auto-updating 7-Day Forecast</small>"
    "</div>",
    unsafe_allow_html=True
)