import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import requests
from datetime import datetime
from sqlalchemy import create_engine

# ==========================================
# 1. PAGE CONFIGURATION & GREY/NAVY CSS
# ==========================================
st.set_page_config(page_title="MFU PM2.5 Analytics", layout="wide")

st.markdown("""
    <style>
    /* Grey App Background */
    .stApp { background-color: #e2e8f0; color: #000000; }
    
    /* Tab Text Colors (Black & Bold, Active is Navy Blue) */
    .stTabs [data-baseweb="tab-list"] button {
        color: #475569 !important; 
        font-size: 16px;
        font-weight: 700;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: #000000 !important; 
        border-bottom-color: #1e3a8a !important; /* Navy Blue */
    }
    
    /* Hero Card (Grey/White with Navy Blue Accent) */
    .hero-card {
        background-color: #ffffff;
        border-radius: 12px; padding: 25px; color: #000000;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
        border-left: 8px solid #1e3a8a; /* Navy Blue Accent */
        margin-bottom: 15px;
    }
    
    /* Info Cards */
    .info-card {
        background-color: #ffffff; border-radius: 12px; padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; color: #000000;
        border-top: 4px solid #94a3b8; /* Light Grey Accent */
    }

    /* Side Panel (Grey Background, Black Text, Navy Accent) */
    .side-panel {
        background-color: #f8fafc; border-radius: 12px; padding: 25px;
        color: #000000; box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
        border-top: 8px solid #1e3a8a; height: 100%;
    }
    
    /* Force Headers to Black and Bold */
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
        font-weight: 900 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. MODELS & DATA FETCHING
# ==========================================
API_KEY = "b5a0c28f2b79a51d156755c60818dcae"
LAT, LON = "20.045", "99.895"

@st.cache_resource
def load_models():
    xgb = joblib.load('pm25_model.pkl')
    try:
        rf = joblib.load('rf_pm25_model.pkl')
    except:
        rf = None
    return xgb, rf

xgb_model, rf_model = load_models()

def fetch_weather_and_forecast():
    try:
        cw = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric").json()
        cp = requests.get(f"https://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}").json()
        fw = requests.get(f"https://api.openweathermap.org/data/2.5/forecast?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric").json()
        
        current = {
            "temp": cw['main']['temp'], "humidity": cw['main']['humidity'], 
            "wind_speed": cw['wind']['speed'], "desc": cw['weather'][0]['description'].title(),
            "pm25_current": cp['list'][0]['components']['pm2_5']
        }
        
        forecast_list = []
        for item in fw['list']:
            forecast_list.append({
                "datetime": datetime.fromtimestamp(item['dt']),
                "pressure_avg": item['main']['pressure'], "temp_avg": item['main']['temp'],
                "humidity_avg": item['main']['humidity'], "precipitation": item.get('rain', {}).get('3h', 0),
                "sunshine": 5.0, "wind_direct": item['wind'].get('deg', 0),
                "wind_speed": item['wind']['speed'], "pm25_lag1": current["pm25_current"]
            })
        return current, pd.DataFrame(forecast_list)
    except Exception as e:
        return None, None

def get_db_data():
    try:
        engine = create_engine('mysql+mysqlconnector://root:@localhost/pm25_project')
        return pd.read_sql("SELECT log_date, pm25, temp_avg, humidity_avg FROM pollution_logs ORDER BY log_date DESC LIMIT 100", engine)
    except:
        return pd.DataFrame()

current_data, forecast_df = fetch_weather_and_forecast()

# ==========================================
# 3. DASHBOARD UI WITH 3 TABS
# ==========================================
st.title("🌬️ MFU Valley Air Quality System")
st.markdown("<p style='color:#000000; font-size:18px;'>Developed by <b>The Outliers</b></p>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🚀 Live Forecast", "📊 Historical Trends", "🔬 Model Analysis"])

# --- TAB 1: LIVE & FUTURE ---
with tab1:
    if current_data and not forecast_df.empty:
        features = ['pressure_avg', 'temp_avg', 'humidity_avg', 'precipitation', 'sunshine', 'wind_direct', 'wind_speed', 'pm25_lag1']
        forecast_df['predicted_pm25'] = xgb_model.predict(forecast_df[features]).clip(min=0)
        forecast_df['predicted_pm25'] = forecast_df['predicted_pm25'].rolling(2, min_periods=1).mean()
        
        current_pred = forecast_df.iloc[0]['predicted_pm25']
        
        # Color Logic for PM2.5
        if current_pred <= 25: 
            status_text = "Good"
            pred_color = "#16a34a" # Green
        elif current_pred <= 50: 
            status_text = "Moderate"
            pred_color = "#d97706" # Yellow/Orange
        elif current_pred <= 100: 
            status_text = "Unhealthy"
            pred_color = "#dc2626" # Red
        else: 
            status_text = "Hazardous"
            pred_color = "#9333ea" # Purple
            
        col_left, col_right = st.columns([7, 3])
        
        with col_left:
            st.markdown(f"""
                <div class="hero-card">
                    <p style="margin:0; font-size:16px; font-weight:bold; color:#000000;">📍 Mae Fah Luang University</p>
                    <h2 style="margin:5px 0; color:#000000;">Current PM2.5: <span style="color:{pred_color}; font-size:42px;">{current_pred:.1f} µg/m³</span></h2>
                    <p style="margin:0; font-size:16px; font-weight:bold; color:#000000;">Model: <span style="color:#1e3a8a;">XGBoost</span> | Status: <span style="color:{pred_color};">{status_text}</span></p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<h5>📈 5-Day PM2.5 Forecast Trend</h5>", unsafe_allow_html=True)
            fig_trend = px.line(forecast_df, x='datetime', y='predicted_pm25',
                                labels={"datetime": "Date & Time", "predicted_pm25": "PM2.5 (µg/m³)"})
            fig_trend.update_traces(line=dict(color='#1e3a8a', width=3)) # Navy Blue Chart Line
            fig_trend.update_layout(
                height=320, margin=dict(l=60, r=20, t=10, b=40),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='black', weight='bold')
            )
            fig_trend.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="Danger Limit (50)")
            fig_trend.update_xaxes(showgrid=False)
            fig_trend.update_yaxes(showgrid=True, gridcolor='#cbd5e1')
            st.plotly_chart(fig_trend, use_container_width=True, theme=None)

            c1, c2, c3 = st.columns(3)
            c1.markdown(f'<div class="info-card"><span style="font-size:15px; font-weight:bold; color:#000000;">Avg Future Temp</span><br><span style="font-size:24px; color:#1e3a8a; font-weight:900;">{forecast_df["temp_avg"].mean():.1f} °C</span></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="info-card"><span style="font-size:15px; font-weight:bold; color:#000000;">Avg Future Wind</span><br><span style="font-size:24px; color:#1e3a8a; font-weight:900;">{forecast_df["wind_speed"].mean():.1f} m/s</span></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="info-card"><span style="font-size:15px; font-weight:bold; color:#000000;">Max Predicted PM2.5</span><br><span style="font-size:24px; color:#dc2626; font-weight:900;">{forecast_df["predicted_pm25"].max():.1f}</span></div>', unsafe_allow_html=True)

            st.markdown("""
                <div class="info-card" style="margin-top: 15px; text-align: left;">
                    <h5 style="margin-top:0; margin-bottom:10px; color:#000000;">📊 PM2.5 Level Guide (µg/m³)</h5>
                    <div style="display: flex; gap: 10px; text-align: center; font-size: 14px;">
                        <div style="flex:1; border: 2px solid #16a34a; background-color:#f0fdf4; color:#000; font-weight:bold; padding:10px; border-radius:8px;">0 - 25<br><span style="color:#16a34a;">Good</span></div>
                        <div style="flex:1; border: 2px solid #d97706; background-color:#fefce8; color:#000; font-weight:bold; padding:10px; border-radius:8px;">26 - 50<br><span style="color:#d97706;">Moderate</span></div>
                        <div style="flex:1; border: 2px solid #dc2626; background-color:#fef2f2; color:#000; font-weight:bold; padding:10px; border-radius:8px;">51 - 100<br><span style="color:#dc2626;">Unhealthy</span></div>
                        <div style="flex:1; border: 2px solid #9333ea; background-color:#faf5ff; color:#000; font-weight:bold; padding:10px; border-radius:8px;">100+<br><span style="color:#9333ea;">Hazardous</span></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with col_right:
            daily_forecast = forecast_df.groupby(forecast_df['datetime'].dt.date).agg({'temp_avg': 'mean', 'predicted_pm25': 'max'}).head(5)
            forecast_html = "".join([f"<div style='display:flex; justify-content:space-between; margin-bottom:12px; font-size:15px; font-weight:bold; color:#000000; border-bottom:1px solid #cbd5e1; padding-bottom:8px;'><span>{row.name.strftime('%A')[:3]}</span><span style='color:{'#dc2626' if row['predicted_pm25']>50 else '#1e3a8a'};'>{row['temp_avg']:.1f}°C | AQI: {row['predicted_pm25']:.0f}</span></div>" for _, row in daily_forecast.iterrows()])
            
            st.markdown(f"""
                <div class="side-panel">
                    <h1 style="text-align:center; margin-bottom:0; font-size:50px; color:#1e3a8a !important;">{current_data['temp']}°C</h1>
                    <p style="text-align:center; font-weight:bold; font-size:16px; margin-top:0; color:#000000;">{current_data['desc']}</p>
                    <div style="display:flex; justify-content:space-around; margin: 20px 0; font-weight:bold; color:#000000;">
                        <span>💨 {current_data['wind_speed']} m/s</span>
                        <span>💧 {current_data['humidity']}%</span>
                    </div>
                    <h4 style="margin-bottom:15px; border-top:2px solid #1e3a8a; padding-top:15px; color:#000000;">Forecast Summary</h4>
                    {forecast_html}
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Loading API Data...")

# --- TAB 2: HISTORICAL TRENDS (FIXED MARGINS & LABELS) ---
with tab2:
    st.markdown("<h4 style='color:#000000;'>Historical Local Data (MySQL)</h4>", unsafe_allow_html=True)
    df_history = get_db_data()
    if not df_history.empty:
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            fig_hist = px.line(df_history, x="log_date", y="pm25", title="PM2.5 Over Time",
                               labels={"log_date": "Date", "pm25": "PM2.5 (µg/m³)"})
            fig_hist.update_traces(line=dict(color='#1e3a8a', width=2))
            fig_hist.update_layout(height=350, margin=dict(l=60, r=20, t=40, b=40), 
                                   plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='black', weight='bold'))
            st.plotly_chart(fig_hist, use_container_width=True, theme=None)
        with col_t2:
            fig_scatter = px.scatter(df_history, x="temp_avg", y="pm25", color="humidity_avg", 
                                     title="Temp vs PM2.5 Correlation",
                                     labels={"temp_avg": "Temperature (°C)", "pm25": "PM2.5 (µg/m³)", "humidity_avg": "Humidity (%)"})
            fig_scatter.update_layout(height=350, margin=dict(l=60, r=20, t=40, b=40), 
                                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='black', weight='bold'))
            st.plotly_chart(fig_scatter, use_container_width=True, theme=None)
    else:
        st.warning("No data found in local MySQL database.")

# --- TAB 3: MODEL ANALYSIS ---
with tab3:
    st.markdown("<br>", unsafe_allow_html=True) 
    
    if current_data and not forecast_df.empty and rf_model is not None:
        features = ['pressure_avg', 'temp_avg', 'humidity_avg', 'precipitation', 'sunshine', 'wind_direct', 'wind_speed', 'pm25_lag1']
        current_input = forecast_df.iloc[[0]][features]
        
        live_xgb = xgb_model.predict(current_input)[0]
        live_rf = rf_model.predict(current_input)[0]
        
        st.markdown(f"""
        <div style="background: #ffffff; border-radius: 12px; padding: 25px; border-top: 6px solid #1e3a8a; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 25px;">
            <h3 style="margin-top: 0; color:#000000;">🥊 Live Model Showdown</h3>
            <p style="font-weight: bold; font-size: 15px; color:#000000;">Real-time prediction comparison based on current weather data.</p>
            <div style="display: flex; gap: 20px; margin-top: 20px;">
                <div style="flex: 1; border: 2px solid #1e3a8a; background-color: #f8fafc; padding: 20px; border-radius: 8px;">
                    <p style="margin: 0; font-weight: bold; font-size: 16px; color:#000000;">🏆 XGBoost (Active Engine)</p>
                    <h1 style="margin: 10px 0 0 0; color: #1e3a8a !important; font-size: 36px;">{live_xgb:.1f} <span style="font-size: 18px; color: #000;">µg/m³</span></h1>
                </div>
                <div style="flex: 1; border: 2px solid #94a3b8; background-color: #f8fafc; padding: 20px; border-radius: 8px;">
                    <p style="margin: 0; font-weight: bold; font-size: 16px; color:#000000;">🔬 Random Forest (Secondary)</p>
                    <h1 style="margin: 10px 0 0 0; color: #475569 !important; font-size: 36px;">{live_rf:.1f} <span style="font-size: 18px; color: #000;">µg/m³</span></h1>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    col_metrics, col_charts = st.columns([1, 2])
    
    with col_metrics:
        st.markdown("""
        <div style="background: #ffffff; border-radius: 12px; padding: 25px; border-top: 4px solid #1e3a8a; box-shadow: 0 4px 6px rgba(0,0,0,0.1); height: 100%;">
            <h4 style="margin-top: 0; color:#000000;">📊 Why XGBoost?</h4>
            <div style="margin-top: 20px; font-weight: bold; color:#000000;">
                <p style="margin-bottom: 5px;">Accuracy (R²)</p>
                <div style="display: flex; justify-content: space-between; margin-bottom: 15px; border-bottom: 2px solid #e2e8f0; padding-bottom: 5px;">
                    <span style="color: #1e3a8a; font-weight:900;">XGBoost: 86.76%</span>
                    <span>RF: 86.24%</span>
                </div>
                <p style="margin-bottom: 5px;">Error Rate (MAE)</p>
                <div style="display: flex; justify-content: space-between; margin-bottom: 20px; border-bottom: 2px solid #e2e8f0; padding-bottom: 5px;">
                    <span style="color: #dc2626; font-weight:900;">XGBoost: 5.37</span>
                    <span>RF: 5.45</span>
                </div>
            </div>
            <p style="font-weight: bold; font-size: 14px; line-height: 1.6; color:#000000;">
                XGBoost was chosen as the primary engine because of its <span style="color:#dc2626;">lower error rate</span>. 
                Its gradient boosting approach allows it to capture the chaotic weather shifts better than Random Forest.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_charts:
        st.markdown('<div style="background: #ffffff; border-radius: 12px; padding: 20px; border-top: 4px solid #1e3a8a; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        
        colors = ['#1e3a8a', '#94a3b8']
        
        df_r2 = pd.DataFrame({"Model": ["XGBoost", "Random Forest"], "Accuracy": [86.76, 86.24]})
        fig_r2 = px.bar(df_r2, x="Model", y="Accuracy", color="Model", text_auto='.2f', color_discrete_sequence=colors)
        fig_r2.update_layout(
            height=300, showlegend=False, margin=dict(l=0, r=0, t=40, b=0), yaxis_range=[80, 90],
            title="Accuracy Comparison", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='black', weight='bold')
        )
        c1.plotly_chart(fig_r2, use_container_width=True, theme=None)
        
        df_mae = pd.DataFrame({"Model": ["XGBoost", "Random Forest"], "MAE": [5.37, 5.45]})
        fig_mae = px.bar(df_mae, x="Model", y="MAE", color="Model", text_auto='.2f', color_discrete_sequence=colors)
        fig_mae.update_layout(
            height=300, showlegend=False, margin=dict(l=0, r=0, t=40, b=0),
            title="Error Rate (Lower is Better)", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='black', weight='bold')
        )
        c2.plotly_chart(fig_mae, use_container_width=True, theme=None)
        st.markdown('</div>', unsafe_allow_html=True)