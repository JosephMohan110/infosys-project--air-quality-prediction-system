# streamlit/advanced_air_quality_predictor.py
import re
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Models
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Time Series Models
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Performance optimization
import concurrent.futures
from functools import lru_cache
import joblib
import gc

# =============================================================================
# PERFORMANCE OPTIMIZATION CONFIGURATION
# =============================================================================

# Configuration for performance tuning
MAX_DATA_POINTS = 50000  # Limit data points for training
CACHE_SIZE = 1000
PARALLEL_PROCESSING = True
MAX_WORKERS = 4

# Set TensorFlow for better performance
tf.config.optimizer.set_jit(True)

# =============================================================================
# OPTIMIZED AQI CALCULATION WITH CACHING
# =============================================================================

@lru_cache(maxsize=CACHE_SIZE)
def calculate_pollutant_aqi_cached(pollutant_name, concentration):
    """Cached version of AQI calculation for performance"""
    return calculate_pollutant_aqi(pollutant_name, concentration)

def calculate_pollutant_aqi(pollutant_name, concentration):
    """Calculate AQI for individual pollutants based on EPA standards"""
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
        else: return 301
    
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
    
    # Other pollutants (generic scale)
    else:
        return min(concentration * 2, 500)

def get_pollutant_units(pollutant_name):
    """Get proper units for each pollutant"""
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
    """Get description for each pollutant"""
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
    """Categorize AQI values into quality levels with colors and emojis"""
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
    """Create a visual gauge meter showing AQI value"""
    cat, color = aqi_category(aqi_val)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi_val,
        title={"text": f"{title} — {cat}", "font": {"size": 20, "color":"#111111"}},
        gauge={
            "axis": {"range": [0, 500], "tickcolor":"black", "tickwidth": 2},
            "bar": {"color": "#ff6600", "line": {"color": "black", "width": 2}},
            "steps": [
                {"range": [0, 50], "color": "#00e400"},
                {"range": [51, 100], "color": "#ffff00"},
                {"range": [101, 200], "color": "#ff7e00"},
                {"range": [201, 300], "color": "#ff0000"},
                {"range": [301, 400], "color": "#99004c"},
                {"range": [401, 500], "color": "#7e0023"}
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
    """Enhanced numeric parsing that handles various data formats"""
    if pd.isna(val) or val is None:
        return np.nan
    
    if isinstance(val, (int, float)):
        return float(val)
    
    if isinstance(val, str):
        val = val.strip().lower()
        
        if val in ['', 'null', 'none', 'nan', 'na', '-', '--', 'n/a']:
            return np.nan
        
        if 'to' in val or '-' in val and not val.startswith('-'):
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", val)
            if numbers:
                return float(numbers[0])
        
        clean_val = re.sub(r'[^\d.-]', ' ', val)
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", clean_val)
        if numbers:
            return float(numbers[0])
        
        try:
            return float(val)
        except:
            pass
    
    try:
        return pd.to_numeric(val, errors='coerce')
    except:
        return np.nan

def compute_aqi_from_pollutants(row, pollutant_cols):
    """Calculate AQI from multiple pollutant values"""
    aqi_vals = []
    for col in pollutant_cols:
        if pd.notna(row[col]):
            concentration = enhanced_parse_to_numeric(row[col])
            if concentration > 0 and not pd.isna(concentration):
                pollutant_aqi = calculate_pollutant_aqi_cached(col, concentration)
                if not pd.isna(pollutant_aqi):
                    aqi_vals.append(pollutant_aqi)
    
    return max(aqi_vals) if aqi_vals else np.nan

def show_current_aqi(df, station_name, pollutant_cols):
    """Display current AQI gauge and value"""
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
    
    # Format AQI value to show decimal if present
    if aqi_val % 1 == 0:
        aqi_display = f"{aqi_val:.0f}"
    else:
        aqi_display = f"{aqi_val:.1f}"
        
    st.markdown(
        f"<div style='padding:15px;border-radius:12px;background:{color};color:white;text-align:center;font-size:20px;font-weight:bold;'>"
        f"🌟 Current AQI: {aqi_display}</div>", 
        unsafe_allow_html=True
    )

def plot_pollutant_timeseries(df, pollutant_cols, station_name):
    """Plot pollutant time series for selected pollutants"""
    if not pollutant_cols:
        st.warning("⚠️ No pollutants selected for visualization")
        return
    
    # Prepare data for plotting
    plot_data = []
    
    for col in pollutant_cols:
        # Clean and process data for this pollutant
        temp_df = df[['Date', col]].copy()
        temp_df[col] = temp_df[col].apply(enhanced_parse_to_numeric)
        temp_df = temp_df.dropna(subset=[col])
        temp_df = temp_df[temp_df[col] > 0]
        
        if len(temp_df) > 0:
            # Sample data for better performance with large datasets
            if len(temp_df) > 1000:
                temp_df = temp_df.iloc[::len(temp_df)//1000]
            
            for _, row in temp_df.iterrows():
                plot_data.append({
                    'Date': row['Date'],
                    'Pollutant': col,
                    'Concentration': row[col],
                    'Units': get_pollutant_units(col)
                })
    
    if not plot_data:
        st.warning("⚠️ No valid pollutant data available for visualization")
        return
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create the plot
    if len(pollutant_cols) == 1:
        # Single pollutant - detailed view
        pollutant = pollutant_cols[0]
        units = get_pollutant_units(pollutant)
        
        fig = px.line(
            plot_df, 
            x='Date', 
            y='Concentration',
            title=f"📊 {pollutant} Trend for {station_name}",
            labels={'Concentration': f'Concentration ({units})', 'Date': 'Date'},
            color_discrete_sequence=['#ff4444']
        )
        
    else:
        # Multiple pollutants - comparative view
        fig = px.line(
            plot_df, 
            x='Date', 
            y='Concentration',
            color='Pollutant',
            title=f"📊 Multiple Pollutant Trends for {station_name}",
            labels={'Concentration': 'Concentration', 'Date': 'Date'},
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        
        # Update legend and layout for better readability
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
    
    # Common layout updates
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        ),
        hovermode='x unified',
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# OPTIMIZED ADVANCED MODEL IMPLEMENTATION
# =============================================================================

class OptimizedAirQualityPredictor:
    """Optimized predictor with parallel processing and performance enhancements"""
    
    def __init__(self):
        self.models = {}
        self.model_performance = {}
        self.selected_models = []
        self.scalers = {}
        self.trained_pollutants = set()
        
    def set_selected_models(self, models):
        """Set which models to use for prediction"""
        self.selected_models = models
        
    def prepare_optimized_data(self, data, lookback=7):
        """Optimized data preparation for models"""
        if len(data) < lookback + 1:
            return np.array([]), np.array([])
        
        # Use vectorized operations for better performance
        X = np.lib.stride_tricks.sliding_window_view(data[:-1], lookback)
        y = data[lookback:]
        return X, y
    
    def build_fast_lstm_model(self, lookback=7):
        """Build optimized LSTM model"""
        model = Sequential([
            LSTM(32, return_sequences=True, input_shape=(lookback, 1)),
            Dropout(0.1),
            LSTM(32, return_sequences=False),
            Dropout(0.1),
            Dense(16),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def train_fast_arima(self, data, pollutant_name):
        """Fast ARIMA training with limited parameters"""
        try:
            # Use simple ARIMA parameters for speed
            model = ARIMA(data, order=(1, 1, 1))
            fitted_model = model.fit()
            return fitted_model
        except:
            try:
                # Fallback to even simpler model
                model = ARIMA(data, order=(1, 0, 0))
                fitted_model = model.fit()
                return fitted_model
            except:
                return None
    
    def train_fast_prophet(self, dates, values, pollutant_name):
        """Fast Prophet training with minimal configuration"""
        try:
            df = pd.DataFrame({
                'ds': dates,
                'y': values
            })
            model = Prophet(
                yearly_seasonality=False,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05
            )
            model.fit(df)
            return model
        except Exception as e:
            return None
    
    def train_fast_lstm(self, data, pollutant_name):
        """Fast LSTM training with early stopping"""
        try:
            if len(data) < 20:
                return None, None
            
            # Normalize data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()
            self.scalers[pollutant_name] = scaler
            
            lookback = min(7, len(data_scaled) - 10)  # Reduced lookback for speed
            X, y = self.prepare_optimized_data(data_scaled, lookback)
            
            if len(X) < 5:
                return None, None
            
            # Split data
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Reshape for LSTM
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            
            model = self.build_fast_lstm_model(lookback)
            
            # Train with minimal epochs
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            
            history = model.fit(
                X_train, y_train,
                batch_size=16,
                epochs=50,  # Reduced epochs
                validation_data=(X_test, y_test),
                verbose=0,
                callbacks=[early_stop]
            )
            
            return model, lookback
        except Exception as e:
            return None, None
    
    def train_fast_random_forest(self, dates, values):
        """Fast Random Forest training"""
        try:
            # Create time-based features
            date_features = []
            for date in dates:
                if isinstance(date, (int, np.integer)):
                    dt = datetime.fromordinal(date)
                else:
                    dt = pd.to_datetime(date)
                
                ts = pd.Timestamp(dt)
                date_features.append([
                    ts.month,
                    ts.day,
                    ts.dayofweek,
                    np.sin(2 * np.pi * ts.dayofyear / 365),
                    np.cos(2 * np.pi * ts.dayofyear / 365),
                ])
            
            X = np.array(date_features)
            y = np.array(values)
            
            model = RandomForestRegressor(
                n_estimators=100,  # Reduced estimators
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X, y)
            return model
        except Exception as e:
            return None
    
    def train_fast_linear_regression(self, dates, values):
        """Fast Linear Regression training"""
        try:
            # Convert dates to numeric features
            X = np.array([date.toordinal() if hasattr(date, 'toordinal') else pd.to_datetime(date).toordinal() 
                         for date in dates]).reshape(-1, 1)
            
            # Simple linear features only
            X_poly = np.column_stack([X])
            
            y = np.array(values)
            
            model = LinearRegression()
            model.fit(X_poly, y)
            return model
        except Exception as e:
            return None
    
    def evaluate_model(self, y_true, y_pred, model_name):
        """Fast model evaluation"""
        if len(y_true) == 0 or len(y_pred) == 0:
            return float('inf'), float('inf'), 0
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return rmse, mae, r2
    
    def train_single_model_parallel(self, model_type, train_dates, train_values, test_dates, test_values, pollutant_name):
        """Train a single model in parallel"""
        try:
            if model_type == 'Linear Regression':
                model = self.train_fast_linear_regression(train_dates, train_values)
                if model:
                    test_numeric = np.array([date.toordinal() if hasattr(date, 'toordinal') else pd.to_datetime(date).toordinal() 
                                           for date in test_dates]).reshape(-1, 1)
                    test_numeric_poly = np.column_stack([test_numeric])
                    pred = model.predict(test_numeric_poly)
                    return model_type, model, pred
                    
            elif model_type == 'Random Forest':
                model = self.train_fast_random_forest(train_dates, train_values)
                if model:
                    test_features = []
                    for date in test_dates:
                        dt = pd.to_datetime(date)
                        test_features.append([
                            dt.month, dt.day, dt.dayofweek,
                            np.sin(2 * np.pi * dt.dayofyear / 365),
                            np.cos(2 * np.pi * dt.dayofyear / 365),
                        ])
                    pred = model.predict(np.array(test_features))
                    return model_type, model, pred
                    
            elif model_type == 'ARIMA':
                model = self.train_fast_arima(train_values, pollutant_name)
                if model:
                    try:
                        pred = model.forecast(steps=len(test_values))
                        return model_type, model, pred
                    except:
                        return None
                        
            elif model_type == 'Prophet':
                model = self.train_fast_prophet(train_dates, train_values, pollutant_name)
                if model:
                    try:
                        future = model.make_future_dataframe(periods=len(test_values))
                        forecast = model.predict(future)
                        pred = forecast['yhat'].values[-len(test_values):]
                        return model_type, model, pred
                    except:
                        return None
                        
            elif model_type == 'LSTM':
                model, lookback = self.train_fast_lstm(train_values, pollutant_name)
                if model and lookback:
                    try:
                        scaler = self.scalers.get(pollutant_name, StandardScaler())
                        train_data_scaled = scaler.transform(train_values.reshape(-1, 1)).flatten()
                        test_data_scaled = scaler.transform(test_values.reshape(-1, 1)).flatten()
                        
                        test_sequences = []
                        for i in range(lookback, len(test_data_scaled)):
                            test_sequences.append(test_data_scaled[i-lookback:i])
                        
                        if len(test_sequences) > 0:
                            X_test_lstm = np.array(test_sequences).reshape(-1, lookback, 1)
                            pred_scaled = model.predict(X_test_lstm, verbose=0).flatten()
                            pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
                            actual_test_values = test_values[lookback:]
                            if len(pred) == len(actual_test_values):
                                return model_type, (model, lookback), pred
                    except:
                        return None
        except:
            return None
        
        return None
    
    def train_selected_models_parallel(self, dates, values, pollutant_name):
        """Train models in parallel and select the best one"""
        if len(values) < 10:
            return None, "Insufficient data"
        
        # Prepare data
        numeric_dates = [date.toordinal() if hasattr(date, 'toordinal') else pd.to_datetime(date).toordinal() 
                        for date in dates]
        
        # Split data for evaluation
        split_idx = max(int(0.8 * len(values)), len(values) - min(20, len(values) - 1))
        train_dates, test_dates = dates[:split_idx], dates[split_idx:]
        train_values, test_values = values[:split_idx], values[split_idx:]
        
        if len(test_values) == 0:
            return None, "No test data available"
        
        models = {}
        performances = {}
        
        # Train models in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(self.selected_models), MAX_WORKERS)) as executor:
            future_to_model = {
                executor.submit(
                    self.train_single_model_parallel, 
                    model_type, train_dates, train_values, test_dates, test_values, pollutant_name
                ): model_type for model_type in self.selected_models
            }
            
            for future in concurrent.futures.as_completed(future_to_model):
                model_type = future_to_model[future]
                try:
                    result = future.result(timeout=30)  # Timeout for each model
                    if result:
                        model_name, model, pred = result
                        rmse, mae, r2 = self.evaluate_model(test_values[:len(pred)], pred, model_name)
                        models[model_name] = model
                        performances[model_name] = (rmse, mae, r2)
                except concurrent.futures.TimeoutError:
                    continue
                except Exception:
                    continue
        
        # Select best model based on RMSE
        if performances:
            if len(performances) == 1:
                best_model_name = list(performances.keys())[0]
            else:
                best_model_name = min(performances.items(), key=lambda x: x[1][0])[0]
            
            best_model = models[best_model_name]
            best_performance = performances[best_model_name]
            
            self.models[pollutant_name] = {
                'model': best_model,
                'model_type': best_model_name,
                'performance': best_performance,
                'all_performances': performances
            }
            
            self.trained_pollutants.add(pollutant_name)
            
            return best_model_name, best_performance
        else:
            return None, "No models trained successfully"

    def predict_with_model_fast(self, model_info, target_date, historical_data=None, pollutant_name=None):
        """Fast prediction using trained model"""
        if not model_info:
            return np.nan
        
        model = model_info['model']
        model_type = model_info['model_type']
        
        try:
            if model_type == 'Linear Regression':
                target_ordinal = pd.to_datetime(target_date).toordinal()
                target_features = np.array([[target_ordinal]])
                return model.predict(target_features)[0]
            
            elif model_type == 'Random Forest':
                target_dt = pd.to_datetime(target_date)
                features = [[
                    target_dt.month, target_dt.day, target_dt.dayofweek,
                    np.sin(2 * np.pi * target_dt.dayofyear / 365),
                    np.cos(2 * np.pi * target_dt.dayofyear / 365),
                ]]
                return model.predict(np.array(features))[0]
            
            elif model_type == 'ARIMA':
                return model.forecast(steps=1)[0]
            
            elif model_type == 'Prophet':
                future = pd.DataFrame({'ds': [target_date]})
                forecast = model.predict(future)
                return forecast['yhat'].values[0]
            
            elif model_type == 'LSTM':
                model_obj, lookback = model
                if historical_data is not None and len(historical_data) >= lookback and pollutant_name:
                    scaler = self.scalers.get(pollutant_name)
                    if scaler:
                        historical_scaled = scaler.transform(np.array(historical_data[-lookback:]).reshape(-1, 1)).flatten()
                        x_input = historical_scaled.reshape(1, lookback, 1)
                        pred_scaled = model_obj.predict(x_input, verbose=0)[0][0]
                        return scaler.inverse_transform([[pred_scaled]])[0][0]
                
                return np.mean(historical_data) if historical_data else np.nan
            
        except Exception as e:
            return np.nan
        
        return np.nan

# =============================================================================
# OPTIMIZED PREDICTION FUNCTIONS
# =============================================================================

def predict_pollutants_fast(df_filtered, pollutant_cols, target_date, predictor):
    """Fast prediction using optimized models"""
    predictions = {}
    prediction_notes = {}
    model_info = {}
    
    # Prepare data for all pollutants first
    pollutant_data = {}
    for col in pollutant_cols:
        try:
            temp_df = df_filtered[['Date', col]].copy()
            temp_df = temp_df.dropna(subset=[col, 'Date'])
            temp_df[col] = temp_df[col].apply(enhanced_parse_to_numeric)
            temp_df = temp_df[temp_df[col] > 0]
            
            if len(temp_df) >= 10:
                pollutant_data[col] = {
                    'dates': temp_df['Date'].values,
                    'values': temp_df[col].values,
                    'historical_stats': {
                        'mean': np.mean(temp_df[col].values),
                        'std': np.std(temp_df[col].values),
                        'min': np.min(temp_df[col].values),
                        'max': np.max(temp_df[col].values)
                    }
                }
            else:
                predictions[col] = np.nan
                prediction_notes[col] = f"Insufficient data ({len(temp_df)} points)"
                
        except Exception as e:
            predictions[col] = np.nan
            prediction_notes[col] = f"Data preparation error: {str(e)}"
    
    # Train models and make predictions in batches
    for col, data_info in pollutant_data.items():
        try:
            # Train models and select best one
            best_model, performance = predictor.train_selected_models_parallel(
                data_info['dates'], data_info['values'], col
            )
            
            if best_model:
                # Make prediction
                pred_val = predictor.predict_with_model_fast(
                    predictor.models[col], 
                    target_date,
                    historical_data=data_info['values'].tolist(),
                    pollutant_name=col
                )
                
                if not pd.isna(pred_val):
                    # Apply reasonable bounds
                    stats = data_info['historical_stats']
                    lower_bound = max(0, stats['mean'] - 2 * stats['std'])
                    upper_bound = stats['mean'] + 2 * stats['std']
                    
                    if pred_val < lower_bound:
                        pred_val = max(stats['min'] * 0.9, lower_bound)
                        prediction_notes[col] = f"Adjusted low prediction ({best_model})"
                    elif pred_val > upper_bound:
                        pred_val = min(stats['max'] * 1.1, upper_bound)
                        prediction_notes[col] = f"Adjusted high prediction ({best_model})"
                    else:
                        rmse, mae, r2 = performance
                        prediction_notes[col] = f"{best_model} (R²: {r2:.3f})"
                    
                    predictions[col] = round(float(pred_val), 3)
                    model_info[col] = {
                        'model': best_model,
                        'performance': performance,
                        'all_models': predictor.models[col]['all_performances'] if col in predictor.models else {}
                    }
                else:
                    # Use historical average as fallback
                    predictions[col] = round(float(data_info['historical_stats']['mean']), 3)
                    prediction_notes[col] = f"Used historical average (model failed)"
            else:
                # Use historical average as fallback
                predictions[col] = round(float(data_info['historical_stats']['mean']), 3)
                prediction_notes[col] = f"Used historical average (no model trained)"
                
        except Exception as e:
            # Use historical average as fallback
            predictions[col] = round(float(data_info['historical_stats']['mean']), 3)
            prediction_notes[col] = f"Used historical average (error: {str(e)})"
    
    return predictions, prediction_notes, model_info

def predict_next_7_days_fast(df_filtered, pollutant_cols, predictor):
    """Fast 7-day prediction using optimized models"""
    next_7_days = []
    today = datetime.now().date()
    
    # Prepare data once for all predictions
    pollutant_data_cache = {}
    for col in pollutant_cols:
        temp_df = df_filtered[['Date', col]].copy()
        temp_df = temp_df.dropna(subset=[col, 'Date'])
        temp_df[col] = temp_df[col].apply(enhanced_parse_to_numeric)
        temp_df = temp_df[temp_df[col] > 0]
        
        if len(temp_df) >= 10:
            pollutant_data_cache[col] = {
                'dates': temp_df['Date'].values,
                'values': temp_df[col].values
            }
    
    # Generate predictions for each day
    for i in range(1, 8):
        target_date = today + timedelta(days=i)
        
        predictions = {}
        for col, data_info in pollutant_data_cache.items():
            if col in predictor.trained_pollutants:
                # Use already trained model
                pred_val = predictor.predict_with_model_fast(
                    predictor.models[col], 
                    target_date,
                    historical_data=data_info['values'].tolist(),
                    pollutant_name=col
                )
            else:
                # Use simple forecasting (moving average)
                pred_val = np.mean(data_info['values'][-7:]) if len(data_info['values']) >= 7 else np.mean(data_info['values'])
            
            if not pd.isna(pred_val) and pred_val > 0:
                predictions[col] = pred_val
        
        # Calculate AQI from predicted pollutants
        aqi_vals = []
        detailed_predictions = {}
        for poll, concentration in predictions.items():
            if not pd.isna(concentration) and concentration > 0:
                poll_aqi = calculate_pollutant_aqi_cached(poll, concentration)
                if not pd.isna(poll_aqi):
                    aqi_vals.append(poll_aqi)
                    detailed_predictions[poll] = {
                        'concentration': concentration,
                        'aqi_impact': poll_aqi,
                        'units': get_pollutant_units(poll)
                    }
        
        if aqi_vals:
            predicted_aqi = max(aqi_vals)
            predicted_aqi = min(500, max(0, predicted_aqi))
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
            'Color': color,
            'Detailed_Predictions': detailed_predictions
        })
    
    return pd.DataFrame(next_7_days)

# =============================================================================
# ENHANCED SUMMARY AND TABLE FUNCTIONS
# =============================================================================

def create_prediction_summary_table(predictions, prediction_notes, model_info, target_date):
    """Create detailed prediction summary table"""
    summary_data = []
    
    for poll, value in predictions.items():
        if not pd.isna(value):
            poll_aqi = calculate_pollutant_aqi_cached(poll, value)
            units = get_pollutant_units(poll)
            model_note = prediction_notes.get(poll, 'Unknown')
            
            # Extract model performance if available
            performance_info = ""
            if poll in model_info:
                rmse, mae, r2 = model_info[poll]['performance']
                performance_info = f"R²: {r2:.3f}"
            
            summary_data.append({
                'Pollutant': poll,
                'Predicted Value': f"{value:.3f} {units}",
                'AQI Impact': f"{poll_aqi:.1f}" if not pd.isna(poll_aqi) else "N/A",
                'Model Used': model_note,
                'Performance': performance_info
            })
    
    return pd.DataFrame(summary_data)

def create_7day_detailed_table(forecast_df):
    """Create detailed 7-day forecast table"""
    table_data = []
    
    for _, day_data in forecast_df.iterrows():
        if pd.isna(day_data['AQI']):
            table_data.append({
                'Date': day_data['Full_Date'],
                'Day': day_data['Day'],
                'AQI': 'N/A',
                'Category': 'Unknown',
                'Primary Pollutant': 'N/A',
                'Max Impact': 'N/A',
                'Pollutant Count': 0
            })
        else:
            detailed_preds = day_data['Detailed_Predictions']
            if detailed_preds:
                # Find primary pollutant (max AQI impact)
                primary_pollutant = max(detailed_preds.items(), key=lambda x: x[1]['aqi_impact'])
                primary_name = primary_pollutant[0]
                primary_value = primary_pollutant[1]['concentration']
                primary_units = primary_pollutant[1]['units']
                max_impact = primary_pollutant[1]['aqi_impact']
                
                # Format AQI value
                if day_data['AQI'] % 1 == 0:
                    aqi_display = f"{day_data['AQI']:.0f}"
                else:
                    aqi_display = f"{day_data['AQI']:.1f}"
                    
                table_data.append({
                    'Date': day_data['Full_Date'],
                    'Day': day_data['Day'],
                    'AQI': aqi_display,
                    'Category': day_data['Category'],
                    'Primary Pollutant': f"{primary_name} ({primary_value:.1f} {primary_units})",
                    'Max Impact': f"{max_impact:.1f}",
                    'Pollutant Count': len(detailed_preds)
                })
            else:
                if day_data['AQI'] % 1 == 0:
                    aqi_display = f"{day_data['AQI']:.0f}"
                else:
                    aqi_display = f"{day_data['AQI']:.1f}"
                    
                table_data.append({
                    'Date': day_data['Full_Date'],
                    'Day': day_data['Day'],
                    'AQI': aqi_display,
                    'Category': day_data['Category'],
                    'Primary Pollutant': 'N/A',
                    'Max Impact': 'N/A',
                    'Pollutant Count': 0
                })
    
    return pd.DataFrame(table_data)

def create_model_performance_table(model_info):
    """Create detailed model performance comparison table"""
    performance_data = []
    
    for poll, info in model_info.items():
        if 'all_models' in info and info['all_models']:
            for model_name, (rmse, mae, r2) in info['all_models'].items():
                performance_data.append({
                    'Pollutant': poll,
                    'Model': model_name,
                    'RMSE': f"{rmse:.4f}",
                    'MAE': f"{mae:.4f}",
                    'R² Score': f"{r2:.4f}",
                    'Status': '✅ Selected' if model_name == info['model'] else '❌ Not selected'
                })
        else:
            rmse, mae, r2 = info['performance']
            performance_data.append({
                'Pollutant': poll,
                'Model': info['model'],
                'RMSE': f"{rmse:.4f}",
                'MAE': f"{mae:.4f}",
                'R² Score': f"{r2:.4f}",
                'Status': '✅ Selected'
            })
    
    return pd.DataFrame(performance_data)

# =============================================================================
# OPTIMIZED DATA PROCESSING
# =============================================================================

def optimize_dataframe(df):
    """Optimize DataFrame for memory and performance"""
    # Convert object columns to category where possible
    for col in df.select_dtypes(['object']).columns:
        if df[col].nunique() / len(df) < 0.5:  # If low cardinality
            df[col] = df[col].astype('category')
    
    # Downcast numeric columns
    for col in df.select_dtypes(['int']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    for col in df.select_dtypes(['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    return df

def sample_large_dataset(df, max_samples=MAX_DATA_POINTS):
    """Sample large datasets for training performance"""
    if len(df) > max_samples:
        # Use stratified sampling to maintain temporal distribution
        df_sampled = df.iloc[::len(df)//max_samples]
        st.info(f"📊 Large dataset sampled from {len(df):,} to {len(df_sampled):,} records for faster processing")
        return df_sampled
    return df

# =============================================================================
# STREAMLIT UI WITH PERFORMANCE OPTIMIZATIONS
# =============================================================================

# Configure the Streamlit page
st.set_page_config(page_title="🌬️ Advanced Air Quality Predictor", layout="wide")
st.markdown("## 🌈 Advanced Air Quality Analysis & Prediction", unsafe_allow_html=True)
st.markdown("<hr style='border:2px solid #4da6ff'>", unsafe_allow_html=True)

# Performance settings
st.sidebar.markdown("### ⚡ Performance Settings")
enable_fast_mode = st.sidebar.checkbox("🚀 Enable Fast Mode", value=True, 
                                      help="Optimize for speed over accuracy")
max_data_points = st.sidebar.slider("Max Data Points for Training", 
                                   min_value=1000, max_value=50000, 
                                   value=10000, step=1000,
                                   help="Limit data points for faster processing")

# File uploader with progress
uploaded = st.file_uploader("📂 Upload CSV file (up to 200MB)", type=['csv'])
if not uploaded:
    st.stop()

# Initialize predictor
if 'predictor' not in st.session_state:
    st.session_state.predictor = OptimizedAirQualityPredictor()

# Progress bar for data loading
progress_bar = st.progress(0)
status_text = st.empty()

try:
    status_text.text("📥 Loading data...")
    progress_bar.progress(10)
    
    # Read CSV with chunks for large files
    if uploaded.size > 10 * 1024 * 1024:  # 10MB
        chunks = []
        chunk_size = 10000
        for chunk in pd.read_csv(uploaded, chunksize=chunk_size):
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
    else:
        df = pd.read_csv(uploaded)
    
    progress_bar.progress(30)
    status_text.text("🔄 Optimizing data...")
    
    # Optimize DataFrame
    df = optimize_dataframe(df)
    
    progress_bar.progress(50)
    status_text.text("📅 Processing dates...")

except Exception as e:
    st.error(f"❌ Error reading CSV: {e}")
    st.stop()

# Date processing
date_cols = df.filter(regex='date|ds|time|datetime', axis=1).columns
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
elif len(date_cols) > 0:
    df["Date"] = pd.to_datetime(df[date_cols[0]], errors="coerce")
else:
    df["Date"] = pd.to_datetime(df.index, errors="coerce")

df = df[df["Date"].notna()]

# Sample large datasets
if enable_fast_mode and len(df) > max_data_points:
    df = sample_large_dataset(df, max_data_points)

# Detect location column
city_col = None
for c in df.columns:
    if any(x in c.lower() for x in ["city", "station", "location", "site", "area"]):
        city_col = c
        break

# Pollutant detection
common_pollutants = [
    'pm2.5', 'pm25', 'pm2_5', 'pm10', 'pm_10', 'o3', 'ozone', 'no2', 'nitrogen',
    'so2', 'sulfur', 'co', 'carbon', 'co2', 'nh3', 'ammonia', 'voc', 'volatile',
    'ch4', 'methane', 'c6h6', 'benzene', 'hcho', 'formaldehyde', 'h2s',
    'aqi', 'quality', 'pollution'
]

exclude_cols = ["date", "time", "index", "id", "no", "code", "row", "serial"]

pollutant_cols = []
for c in df.columns:
    c_lower = c.lower()
    if any(exclude in c_lower for exclude in exclude_cols):
        continue
    if c == city_col or c == "Date":
        continue
    
    is_known_pollutant = any(poll in c_lower for poll in common_pollutants)
    has_data = df[c].notna().sum() > 0
    
    if has_data:
        sample_data = df[c].dropna().head(10)
        numeric_count = sum(1 for val in sample_data if not pd.isna(enhanced_parse_to_numeric(val)) and enhanced_parse_to_numeric(val) > 0)
        
        if (numeric_count / len(sample_data) > 0.3) or is_known_pollutant:
            pollutant_cols.append(c)

if not pollutant_cols:
    st.error("❌ No pollutant columns detected!")
    st.stop()

# Data cleaning
df_clean = df.copy()
df_clean = df_clean[df_clean[pollutant_cols].notna().any(axis=1)]

for col in pollutant_cols:
    df_clean[col] = df_clean[col].apply(enhanced_parse_to_numeric)

df = df_clean.sort_values('Date')

progress_bar.progress(80)
status_text.text("🎯 Finalizing setup...")

# UI Controls
cities = ["All"] + (sorted(df[city_col].dropna().unique()) if city_col else [])
pollutants = ["All"] + pollutant_cols

col1, col2 = st.columns([1.5, 1.5])
with col1:
    station = st.selectbox("🏙️ Select City/Station", options=cities)
with col2:
    selected_pollutant = st.selectbox("💨 Select Pollutant", options=pollutants)

# Model Selection
st.markdown("### 🤖 Select Prediction Models")
available_models = ["ARIMA", "Prophet", "LSTM", "Random Forest", "Linear Regression"]
selected_models = st.multiselect(
    "Choose which models to use for prediction:",
    options=available_models,
    default=["Random Forest", "Linear Regression"],  # Default to faster models
    help="Select one or more models. If multiple models are selected, the best performing one will be automatically chosen."
)

if not selected_models:
    st.warning("⚠️ Please select at least one model for prediction")
else:
    st.session_state.predictor.set_selected_models(selected_models)
    st.success(f"✅ Selected models: {', '.join(selected_models)}")

# Filter data
df_filtered = df.copy()
if city_col and station != "All":
    df_filtered = df_filtered[df_filtered[city_col] == station]
    st.success(f"📊 Showing data for: {station}")

st.info(f"📈 Available data: {len(df_filtered)} records with {len(pollutant_cols)} pollutants")

progress_bar.progress(100)
status_text.text("✅ Data loaded successfully!")
progress_bar.empty()
status_text.empty()

# Current AQI
st.markdown("### 🌟 Current Air Quality (AQI Gauge)")
show_current_aqi(df_filtered, station or "Site", pollutant_cols)

# Pollutant Time Series Visualization
st.markdown("### 📊 Pollutant Trends Over Time")

# Determine which pollutants to show in the graph
if selected_pollutant == "All":
    pollutants_to_plot = pollutant_cols[:5]  # Limit to 5 for performance
else:
    pollutants_to_plot = [selected_pollutant]

plot_pollutant_timeseries(df_filtered, pollutants_to_plot, station or "Selected Site")

# Advanced Prediction Section with Timer
st.markdown("### 🔮 Advanced Multi-Model Prediction")
st.info(f"🎯 Using selected models: {', '.join(selected_models) if selected_models else 'No models selected'}")

col1, col2 = st.columns([2, 1])
with col1:
    # Allow selection of any future date (100+ years)
    min_date = datetime.now().date() + timedelta(days=1)
    max_date = datetime.now().date() + timedelta(days=365*100)  # 100 years in future
    
    target_date = st.date_input(
        "🗓️ Select future date for prediction",
        min_value=min_date,
        max_value=max_date,
        value=datetime.now().date() + timedelta(days=7),
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🚀 Advanced Predict", type="primary", disabled=not selected_models)

if target_date and predict_btn and selected_models:
    if len(df_filtered) < 10:
        st.error("❌ Need at least 10 data points for advanced prediction")
    else:
        import time
        start_time = time.time()
        
        with st.spinner(f"🤖 Training {len(selected_models)} model(s) with parallel processing..."):
            predictions, prediction_notes, model_info = predict_pollutants_fast(
                df_filtered, pollutant_cols, target_date, st.session_state.predictor
            )
        
        processing_time = time.time() - start_time
        st.success(f"✅ Prediction completed in {processing_time:.2f} seconds")
        
        valid_predictions = {k: v for k, v in predictions.items() if not pd.isna(v)}
        
        if valid_predictions:
            st.subheader(f"🎯 Advanced Predictions for {target_date}")
            
            # Calculate overall AQI
            aqi_vals = []
            for poll, concentration in valid_predictions.items():
                poll_aqi = calculate_pollutant_aqi_cached(poll, concentration)
                if not pd.isna(poll_aqi):
                    aqi_vals.append(poll_aqi)
            
            predicted_aqi = max(aqi_vals) if aqi_vals else np.nan
            
            # Display results in expandable sections
            with st.expander("📊 Prediction Results Summary", expanded=True):
                results_data = []
                for poll, value in valid_predictions.items():
                    poll_aqi = calculate_pollutant_aqi_cached(poll, value)
                    units = get_pollutant_units(poll)
                    model_name = prediction_notes.get(poll, 'Unknown model')
                    
                    results_data.append({
                        'Pollutant': poll,
                        'Units': units,
                        'Predicted Value': value,
                        'AQI Impact': round(poll_aqi, 1) if not pd.isna(poll_aqi) else "N/A",
                        'Model Used': model_name
                    })
                
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df, width='stretch')
            
            # NEW: Detailed Prediction Summary Table
            with st.expander("📋 Detailed Prediction Summary Table", expanded=False):
                summary_table = create_prediction_summary_table(predictions, prediction_notes, model_info, target_date)
                if not summary_table.empty:
                    st.dataframe(
                        summary_table,
                        use_container_width=True,
                        column_config={
                            "Pollutant": st.column_config.TextColumn("💨 Pollutant", width="medium"),
                            "Predicted Value": st.column_config.TextColumn("📊 Predicted Value", width="medium"),
                            "AQI Impact": st.column_config.TextColumn("🌡️ AQI Impact", width="small"),
                            "Model Used": st.column_config.TextColumn("🤖 Model Used", width="large"),
                            "Performance": st.column_config.TextColumn("📈 Performance", width="small")
                        }
                    )
                else:
                    st.info("No prediction data available for summary table")
            
            # Show overall AQI prediction
            if not pd.isna(predicted_aqi):
                category, color = aqi_category(predicted_aqi)
                
                if predicted_aqi % 1 == 0:
                    aqi_display = f"{predicted_aqi:.0f}"
                else:
                    aqi_display = f"{predicted_aqi:.1f}"
                    
                st.markdown(
                    f"<div style='padding:20px;border-radius:12px;background:{color};color:white;text-align:center;font-size:24px;font-weight:bold;margin:20px 0;'>"
                    f"📊 Predicted Overall AQI: {aqi_display} — {category}</div>", 
                    unsafe_allow_html=True
                )
            
            # NEW: Model Performance Comparison Table
            with st.expander("🔍 Model Performance Analysis", expanded=False):
                if model_info:
                    performance_table = create_model_performance_table(model_info)
                    if not performance_table.empty:
                        st.markdown("### 🎯 Model Performance Comparison")
                        st.dataframe(
                            performance_table,
                            use_container_width=True,
                            column_config={
                                "Pollutant": st.column_config.TextColumn("💨 Pollutant", width="medium"),
                                "Model": st.column_config.TextColumn("🤖 Model", width="medium"),
                                "RMSE": st.column_config.TextColumn("📉 RMSE", width="small"),
                                "MAE": st.column_config.TextColumn("📏 MAE", width="small"),
                                "R² Score": st.column_config.TextColumn("📈 R² Score", width="small"),
                                "Status": st.column_config.TextColumn("✅ Status", width="small")
                            }
                        )
                        
                        # Performance visualization
                        st.markdown("### 📊 Performance Metrics Visualization")
                        fig = px.bar(
                            performance_table,
                            x='Model',
                            y='R² Score',
                            color='Pollutant',
                            title="Model R² Scores by Pollutant",
                            barmode='group'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No detailed model performance data available")
                else:
                    st.info("No model performance information available")
        else:
            st.error("❌ No valid predictions could be generated with the selected models")

# 7-Day Advanced Forecast with Timer
st.markdown("### 🌟 7-Day Advanced Forecast")

if len(df_filtered) >= 10 and selected_models:
    import time
    start_time = time.time()
    
    with st.spinner("🌤️ Generating 7-day forecast with optimized models..."):
        forecast_df = predict_next_7_days_fast(df_filtered, pollutant_cols, st.session_state.predictor)
    
    forecast_time = time.time() - start_time
    st.success(f"✅ 7-day forecast completed in {forecast_time:.2f} seconds")
    
    if not forecast_df.empty:
        # Display forecast cards
        st.markdown("#### 📅 Daily AQI Forecast")
        cols = st.columns(7)
        
        for idx, (_, day_data) in enumerate(forecast_df.iterrows()):
            with cols[idx]:
                if pd.isna(day_data['AQI']):
                    st.markdown(
                        f"<div style='padding:10px;border-radius:12px;background:#f8f9fa;text-align:center;border:2px solid #dee2e6;min-height:160px;'>"
                        f"<div style='font-weight:bold;font-size:14px;color:#6c757d;'>{day_data['Full_Date']}</div>"
                        f"<div style='font-size:12px;color:#6c757d;margin-bottom:8px;'>{day_data['Day']}</div>"
                        f"<div style='margin:12px 0;font-size:24px;'>❓</div>"
                        f"<div style='font-size:12px;color:#6c757d;'>No Data</div>"
                        f"</div>", 
                        unsafe_allow_html=True
                    )
                else:
                    emoji = "😊" if "Good" in day_data['Category'] else "🙂" if "Satisfactory" in day_data['Category'] else "😐" if "Moderate" in day_data['Category'] else "😷" if "Poor" in day_data['Category'] else "🤢" if "Very Poor" in day_data['Category'] else "☠️"
                    text_color = "white" if day_data['AQI'] > 200 else "#111111"
                    
                    if day_data['AQI'] % 1 == 0:
                        aqi_display = f"{day_data['AQI']:.0f}"
                    else:
                        aqi_display = f"{day_data['AQI']:.1f}"
                    
                    st.markdown(
                        f"<div style='padding:12px;border-radius:12px;background:{day_data['Color']};text-align:center;color:{text_color};border:3px solid {day_data['Color']}80;min-height:160px;'>"
                        f"<div style='font-weight:bold;font-size:15px;'>{day_data['Full_Date']}</div>"
                        f"<div style='font-size:13px;opacity:0.9;margin-bottom:10px;'>📅 {day_data['Day']}</div>"
                        f"<div style='margin:12px 0;font-size:28px;'>{emoji}</div>"
                        f"<div style='font-weight:bold;font-size:18px;'>AQI {aqi_display}</div>"
                        f"<div style='font-size:12px;opacity:0.9;margin-top:5px;'>{day_data['Category'].split()[0]}</div>"
                        f"</div>", 
                        unsafe_allow_html=True
                    )
        
        # NEW: Detailed 7-Day Forecast Table
        st.markdown("#### 📋 Detailed 7-Day Forecast Table")
        
        detailed_7day_table = create_7day_detailed_table(forecast_df)
        if not detailed_7day_table.empty:
            st.dataframe(
                detailed_7day_table,
                use_container_width=True,
                column_config={
                    "Date": st.column_config.TextColumn("📅 Date", width="medium"),
                    "Day": st.column_config.TextColumn("📆 Day", width="small"),
                    "AQI": st.column_config.TextColumn("🌡️ AQI", width="small"),
                    "Category": st.column_config.TextColumn("📊 Category", width="medium"),
                    "Primary Pollutant": st.column_config.TextColumn("💨 Primary Pollutant", width="large"),
                    "Max Impact": st.column_config.TextColumn("⚡ Max Impact", width="small"),
                    "Pollutant Count": st.column_config.NumberColumn("🔢 Pollutant Count", width="small")
                }
            )
        
        # NEW: 7-Day Forecast Summary
        with st.expander("📈 7-Day Forecast Summary", expanded=False):
            valid_forecasts = forecast_df.dropna(subset=['AQI'])
            if not valid_forecasts.empty:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    avg_aqi = valid_forecasts['AQI'].mean()
                    if avg_aqi % 1 == 0:
                        avg_display = f"{avg_aqi:.0f}"
                    else:
                        avg_display = f"{avg_aqi:.1f}"
                    st.metric("📊 Average AQI", avg_display)
                with col2:
                    best_day = valid_forecasts.loc[valid_forecasts['AQI'].idxmin()]
                    if best_day['AQI'] % 1 == 0:
                        best_display = f"{best_day['AQI']:.0f}"
                    else:
                        best_display = f"{best_day['AQI']:.1f}"
                    st.metric("🎯 Best Day", f"{best_day['Day']} ({best_display})")
                with col3:
                    worst_day = valid_forecasts.loc[valid_forecasts['AQI'].idxmax()]
                    if worst_day['AQI'] % 1 == 0:
                        worst_display = f"{worst_day['AQI']:.0f}"
                    else:
                        worst_display = f"{worst_day['AQI']:.1f}"
                    st.metric("⚠️ Worst Day", f"{worst_day['Day']} ({worst_display})")
                with col4:
                    trend = "📈 Improving" if valid_forecasts['AQI'].iloc[-1] < valid_forecasts['AQI'].iloc[0] else "📉 Worsening"
                    st.metric("🔍 Trend", trend)
                
                # Forecast trend chart
                fig = px.line(
                    valid_forecasts, 
                    x='Full_Date', 
                    y='AQI',
                    title="7-Day AQI Forecast Trend",
                    markers=True,
                    line_shape='spline'
                )
                fig.update_traces(line=dict(width=4), marker=dict(size=8))
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="AQI",
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No valid forecast data available for summary")
elif not selected_models:
    st.warning("⚠️ Please select at least one model to generate 7-day forecast")

# Data Summary Section
st.markdown("### 📋 Data Summary & Statistics")
with st.expander("📊 View Comprehensive Data Summary", expanded=False):
    def generate_fast_summary(df, pollutant_cols):
        """Generate fast data summary"""
        summary_data = []
        for col in pollutant_cols:
            clean_data = df[col].apply(enhanced_parse_to_numeric).dropna()
            if len(clean_data) > 0:
                summary_data.append({
                    'Pollutant': col,
                    'Units': get_pollutant_units(col),
                    'Data Points': len(clean_data),
                    'Mean': f"{clean_data.mean():.3f}",
                    'Median': f"{clean_data.median():.3f}",
                    'Std Dev': f"{clean_data.std():.3f}",
                    'Min': f"{clean_data.min():.3f}",
                    'Max': f"{clean_data.max():.3f}",
                    'Missing %': f"{(df[col].isna().sum() / len(df) * 100):.1f}%"
                })
        return pd.DataFrame(summary_data)
    
    summary_df = generate_fast_summary(df_filtered, pollutant_cols)
    if not summary_df.empty:
        st.dataframe(summary_df, width='stretch')
        
        # Overall statistics
        st.markdown("#### 📈 Overall Dataset Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Records", len(df_filtered))
        with col2:
            st.metric("⏱️ Time Span", f"{(df_filtered['Date'].max() - df_filtered['Date'].min()).days} days")
        with col3:
            st.metric("💨 Pollutants Tracked", len(pollutant_cols))
        with col4:
            completeness = (df_filtered[pollutant_cols].notna().sum().sum() / (len(df_filtered) * len(pollutant_cols))) * 100
            st.metric("✅ Data Completeness", f"{completeness:.1f}%")
        
        # Data quality visualization
        st.markdown("#### 🎯 Data Quality Overview")
        quality_data = []
        for col in pollutant_cols:
            clean_data = df_filtered[col].apply(enhanced_parse_to_numeric).dropna()
            if len(clean_data) > 0:
                quality_data.append({
                    'Pollutant': col,
                    'Data Points': len(clean_data),
                    'Completeness %': (len(clean_data) / len(df_filtered)) * 100
                })
        
        if quality_data:
            quality_df = pd.DataFrame(quality_data)
            fig = px.bar(
                quality_df, 
                x='Pollutant', 
                y='Completeness %',
                title="Data Completeness by Pollutant",
                color='Completeness %',
                color_continuous_scale='viridis'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No valid data available for summary")

# Performance Statistics
st.sidebar.markdown("### 📊 Performance Stats")
st.sidebar.metric("Total Records", f"{len(df):,}")
st.sidebar.metric("Pollutants Tracked", len(pollutant_cols))
st.sidebar.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 20px;'>"
    "🌍 <b>Advanced Air Quality Predictor</b> - Optimized Multi-Model AI System<br>"
    "<small>⚡ Parallel Processing • Optimized Models • Fast Predictions • Detailed Analytics</small>"
    "</div>",
    unsafe_allow_html=True
)