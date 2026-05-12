import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import requests
import os
import base64
import pickle
import json
import math
from io import StringIO
from datetime import datetime, timedelta

# ==========================================
# PAGE CONFIGURATION & CSS
# ==========================================
st.set_page_config(page_title="MFU PM2.5 Analytics (v2)", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #F4F7F9; color: #1E293B; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: 2px solid #E2E8F0; }
.stTabs [data-baseweb="tab-list"] button {
    background-color: transparent; border-radius: 8px 8px 0 0; padding: 12px 24px; 
    color: #475569 !important; font-size: 16px; font-weight: 600; transition: all 0.2s ease;
}
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
    color: #1e3a8a !important; border-bottom: 3px solid #1e3a8a !important; 
    background-color: #FFFFFF; box-shadow: 0 -4px 15px rgba(0,0,0,0.03);
}
.glass-card {
    background-color: #ffffff; border-radius: 16px; padding: 25px; color: #1E293B;
    box-shadow: 0 10px 25px rgba(149, 157, 165, 0.08); border: 1px solid #F1F5F9;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.glass-card:hover { transform: translateY(-3px); box-shadow: 0 15px 35px rgba(149, 157, 165, 0.12); }
.hero-card { border-left: 6px solid #1e3a8a; }
.side-panel { border-top: 6px solid #1e3a8a; height: 100%; display: flex; flex-direction: column; justify-content: space-between; }
.info-card {
    background-color: #ffffff; border-radius: 12px; padding: 20px; text-align: center;
    box-shadow: 0 4px 12px rgba(149, 157, 165, 0.05); border: 1px solid #F1F5F9;
    transition: transform 0.2s ease;
}
.info-card:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(149, 157, 165, 0.1); }
.custom-header {
    display: flex; align-items: center; gap: 12px; padding-bottom: 20px;
    border-bottom: 2px solid #E2E8F0; margin-bottom: 30px;
}
.logo-img { width: 75px; height: auto; filter: drop-shadow(0px 4px 6px rgba(0,0,0,0.05)); }
.icon-img { width: 45px; height: auto; }
.header-text-container { display: flex; flex-direction: column; justify-content: center; }
.header-title { margin: 0 !important; font-size: 36px !important; color: #1e3a8a !important; font-weight: 900 !important; letter-spacing: -0.5px; line-height: 1.1; }
.header-subtitle { margin: 2px 0 0 0 !important; font-size: 16px !important; color: #475569 !important; font-weight: 600 !important; }
.model-card {
    flex: 1; background-color: #FFFFFF; padding: 25px; border-radius: 12px;
    box-shadow: 0 6px 16px rgba(149, 157, 165, 0.06); border: 1px solid #F1F5F9;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.model-card:hover { transform: translateY(-4px); box-shadow: 0 12px 24px rgba(149, 157, 165, 0.12); }
.mc-xgb-fire { border-top: 5px solid #1e3a8a; }
.mc-xgb-base { border-top: 5px solid #3b82f6; }
.mc-rf-base { border-top: 5px solid #475569; }
.mc-svr { border-top: 5px solid #64748b; }
.mc-mlr { border-top: 5px solid #94a3b8; }
.aqi-badge { flex:1; padding: 12px; border-radius: 10px; font-weight: 700; text-align: center; }
.plotly-tooltip { font-family: 'Inter', sans-serif !important; border-radius: 8px !important; box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important; }
</style>
""", unsafe_allow_html=True)

# 🚨 SECURE API KEY HANDLING
try:
    API_KEY = st.secrets["OPENWEATHER_API_KEY"]
    NASA_KEY = st.secrets["NASA_KEY"]
except KeyError:
    API_KEY = "YOUR_OPENWEATHER_KEY"
    NASA_KEY = "YOUR_NASA_KEY"

LAT, LON = 20.045, 99.895

# ==========================================
# SMART 5-MODEL LOADER (FIXED)
# ==========================================
@st.cache_resource
def load_models():
    models = {}
    def safe_load(filename):
        paths = [f"models/{filename}", filename]
        for path in paths:
            if os.path.exists(path):
                try:
                    # Model အားလုံးကို joblib ဖြင့်သာ ဖတ်မည် (Training script များနှင့် ကိုက်ညီစေရန်)
                    return joblib.load(path)
                except Exception as e:
                    pass
        return None

    models['xgb_fire'] = safe_load('pm25_model_v7.pkl')
    models['xgb_base'] = safe_load('pm25_model.pkl')
    models['rf_base'] = safe_load('rf_pm25_model.pkl')
    models['svr_base'] = safe_load('svr_pm25_model.pkl')
    models['mlr_base'] = safe_load('mlr_pm25_model.pkl')
    
    return models

loaded_models = load_models()
xgb_fire_model = loaded_models['xgb_fire']
xgb_base_model = loaded_models['xgb_base']
rf_base_model = loaded_models['rf_base']
svr_model = loaded_models['svr_base']
mlr_model = loaded_models['mlr_base']

# ==========================================
# DATA FETCHING FUNCTIONS (WEATHER & FIRE)
# ==========================================
def fetch_weather_and_forecast():
    try:
        cw = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric").json()
        fw = requests.get(f"https://api.openweathermap.org/data/2.5/forecast?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric").json()
        
        try:
            a4t_url = "http://air4thai.pcd.go.th/services/getNewAQI_JSON.php?stationID=73t"
            a4t_data = requests.get(a4t_url, timeout=5).json()
            real_pm25 = float(a4t_data['AQILast']['PM25']['value'])
        except Exception as e:
            cp = requests.get(f"https://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}").json()
            real_pm25 = float(cp['list'][0]['components']['pm2_5'])

        current = {
            "temp": cw['main']['temp'], 
            "humidity": cw['main']['humidity'], 
            "wind_speed": cw['wind']['speed'], 
            "desc": cw['weather'][0]['description'].title(),
            "pm25_current": real_pm25, 
            "fetch_time": datetime.now().strftime("%d %B %Y, %I:%M %p")
        }
        
        forecast_list = []
        for item in fw['list']:
            forecast_list.append({
                "datetime": datetime.fromtimestamp(item['dt']),
                "pressure_avg": item['main']['pressure'], 
                "temp_avg": item['main']['temp'],
                "humidity_avg": item['main']['humidity'], 
                "precipitation": item.get('rain', {}).get('3h', 0),
                "sunshine": 5.0, 
                "wind_direct": item['wind'].get('deg', 0),
                "wind_speed": item['wind']['speed'], 
                "pm25_lag1": current["pm25_current"] 
            })
        return current, pd.DataFrame(forecast_list)
    except Exception as e: 
        return None, None

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def fetch_nasa_fire(date_str):
    bbox = "99.4,19.6,100.4,20.5"
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{NASA_KEY}/VIIRS_SNPP_NRT/{bbox}/1/{date_str}"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            df = pd.read_csv(StringIO(res.text))
            if df.empty: return 0, 0.0
            df['dist'] = df.apply(lambda r: haversine(LAT, LON, r['latitude'], r['longitude']), axis=1)
            df['pres'] = df['bright_ti4'] / (df['dist'] + 1)**2
            return len(df), round(df['pres'].sum(), 2)
        return None, None
    except:
        return None, None

@st.cache_data(ttl=3600)
def get_realtime_fire_features():
    today = datetime.now()
    d0 = today.strftime("%Y-%m-%d")
    d1 = (today - timedelta(days=1)).strftime("%Y-%m-%d")
    d2 = (today - timedelta(days=2)).strftime("%Y-%m-%d")

    c0, p0 = fetch_nasa_fire(d0)
    c1, p1 = fetch_nasa_fire(d1)
    c2, p2 = fetch_nasa_fire(d2)

    # 🚨 THE FALLBACK: NASA API ERROR OR TIMEOUT
    if c0 is None or p0 is None:
        try:
            # 📂 UPDATED PATH TO MATCH NEW STRUCTURE
            hist_paths = ['data/processed/final_training_data.csv', 'final_training_data.csv']
            df_hist = None
            for p in hist_paths:
                if os.path.exists(p):
                    df_hist = pd.read_csv(p)
                    break
            
            if df_hist is not None and not df_hist.empty:
                c0 = df_hist.iloc[-1]['Fire_Count']
                p0 = df_hist.iloc[-1]['Fire_Pressure']
                p1 = df_hist.iloc[-2]['Fire_Pressure']
                p2 = df_hist.iloc[-3]['Fire_Pressure']
            else:
                c0, p0, p1, p2 = 0, 0.0, 0.0, 0.0
        except:
            c0, p0, p1, p2 = 0, 0.0, 0.0, 0.0

    c0 = c0 or 0; p0 = p0 or 0.0
    p1 = p1 or 0.0; p2 = p2 or 0.0
    p_3day_avg = round((p0 + p1 + p2) / 3, 2)
    
    return c0, p0, p1, p2, p_3day_avg

def get_historical_data():
    try:
        # 📂 UPDATED PATH TO MATCH NEW STRUCTURE
        df = pd.read_csv("data/processed/final_training_data.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        return df.sort_values(by='Date', ascending=False).head(100)
    except Exception as e: 
        try:
            df = pd.read_csv("final_training_data.csv")
            df['Date'] = pd.to_datetime(df['Date'])
            return df.sort_values(by='Date', ascending=False).head(100)
        except: return pd.DataFrame()

# INITIALIZE DATA
current_data, forecast_df = fetch_weather_and_forecast()
live_fire_count, live_p0, live_p1, live_p2, live_p_avg = get_realtime_fire_features()

def get_base64_img(path):
    if os.path.exists(path):
        with open(path, "rb") as f: data = f.read()
        return f"data:image/png;base64,{base64.b64encode(data).decode()}"
    return ""

mfu_base64 = get_base64_img("mfu_logo.png")
icon_base64 = get_base64_img("image_896c03.png")

img_tags = ""
if mfu_base64:
    img_tags += f'<img src="{mfu_base64}" class="logo-img">'
if icon_base64:
    img_tags += f'<img src="{icon_base64}" class="icon-img">'

header_html = (
    '<div class="custom-header">'
    + img_tags +
    '<div class="header-text-container">'
    '<h1 class="header-title">Mae Fah Luang University PM2.5 Forecasting Engine</h1>'
    '<p class="header-subtitle">Developed by <span style="color:#1e3a8a;">The Outliers</span></p>'
    '</div></div>'
)
st.markdown(header_html, unsafe_allow_html=True)

if isinstance(xgb_fire_model, str):
    st.error(f"🚨 CRITICAL ERROR: Champion Model လမ်းကြောင်း မှားယွင်းနေပါသည်။ Error: {xgb_fire_model}")
    st.stop()

# ==========================================
# TABS SETUP
# ==========================================
tab1, tab2, tab3 = st.tabs(["🚀 Live Forecast", "📊 Historical Trends", "🔬 Methodology Defense"])

with tab1:
    if current_data and not forecast_df.empty:
        app_features = ['pressure_avg', 'temp_avg', 'humidity_avg', 'precipitation', 'sunshine', 'wind_direct', 'wind_speed', 'pm25_lag1']
        model_features = ['Pressure_avg', 'Temp_avg', 'Humidity_avg', 'Precipitation', 'Sunshine', 'Wind_direct', 'Wind_speed', 'pm25_lag1']
        
        model_input = forecast_df[app_features].copy()
        model_input.columns = model_features
        
        # Exact 18 Features Injection for Champion V7 with Real Live Data
        model_input['pm25_lag2'] = model_input['pm25_lag1'] 
        model_input['pm25_lag3'] = model_input['pm25_lag1'] 
        model_input['pm25_3Day_Avg'] = model_input['pm25_lag1']
        
        model_input['Fire_Count'] = live_fire_count
        model_input['Fire_Pressure'] = live_p0
        model_input['Fire_Pressure_Lag1'] = live_p1
        model_input['Fire_Pressure_Lag2'] = live_p2
        model_input['Fire_Pressure_3Day_Avg'] = live_p_avg
        
        current_month = datetime.now().month
        model_input['Month'] = current_month
        model_input['Is_Burning_Season'] = 1 if current_month in [2, 3, 4, 5] else 0

        final_18_cols = [
            'Pressure_avg', 'Temp_avg', 'Humidity_avg', 'Precipitation', 'Sunshine', 
            'Wind_direct', 'Wind_speed', 'pm25_lag1', 'pm25_lag2', 'pm25_lag3', 
            'pm25_3Day_Avg', 'Fire_Count', 'Fire_Pressure', 'Fire_Pressure_Lag1', 
            'Fire_Pressure_Lag2', 'Fire_Pressure_3Day_Avg', 'Month', 'Is_Burning_Season'
        ]
        
        try:
            # Reversing the log transformation exactly as requested
            log_predictions = xgb_fire_model.predict(model_input[final_18_cols])
            forecast_df['predicted_pm25'] = np.expm1(log_predictions).clip(min=0)
        except Exception as e:
            st.error(f"Feature Error in Tab 1: {e}")
            st.stop()
            
        forecast_df['predicted_pm25'] = forecast_df['predicted_pm25'].rolling(2, min_periods=1).mean()
        current_pred = forecast_df.iloc[0]['predicted_pm25']
        
        if current_pred <= 25: status_text = "Good"; pred_color = "#16a34a"
        elif current_pred <= 50: status_text = "Moderate"; pred_color = "#d97706"
        elif current_pred <= 100: status_text = "Unhealthy"; pred_color = "#dc2626"
        else: status_text = "Hazardous"; pred_color = "#9333ea"
            
        col_left, col_right = st.columns([7, 3])
        with col_left:
            st.markdown(f"""
<div class="glass-card hero-card">
<p style="margin:0; font-size:14px; font-weight:1000; color:#475569; text-transform:uppercase; letter-spacing:1px;">📍 Mae Fah Luang University</p>
<h2 style="margin:10px 0; font-size:24px; color:#1E293B; font-weight:800;">Current PM2.5: <span style="color:{pred_color}; font-size:46px;">{current_pred:.1f} <span style="font-size:20px;">µg/m³</span></span></h2>
<p style="margin:0; font-size:16px; font-weight:600; color:#334155;">Model: <span style="color:#1e3a8a; font-weight:800;">XGBoost (89% Fire-Integrated)</span> &nbsp;|&nbsp; Status: <span style="color:{pred_color}; font-weight:800;">{status_text}</span></p>
</div>
""", unsafe_allow_html=True)
            
            st.markdown("<h4 style='margin-top:20px; margin-bottom:10px; color:#1E293B;'>📈 5-Day PM2.5 Forecast Trend</h4>", unsafe_allow_html=True)
            fig_trend = px.line(forecast_df, x='datetime', y='predicted_pm25', labels={"datetime": "", "predicted_pm25": "PM2.5 (µg/m³)"})
            fig_trend.update_traces(
                line=dict(color='#1e3a8a', width=4), 
                mode='lines+markers', 
                marker=dict(size=6, color='#1e3a8a'),
                hovertemplate="<b>Date:</b> %{x|%d %b %Y, %H:%M}<br><b>Predicted PM2.5:</b> %{y:.1f} µg/m³<extra></extra>"
            )
            fig_trend.update_layout(height=340, margin=dict(l=40, r=20, t=10, b=40), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color='#334155', size=13), hoverlabel=dict(bgcolor="white", font_size=13, font_family="Inter"))
            fig_trend.add_hline(y=50, line_dash="dash", line_color="#dc2626", annotation_text="Danger Limit (50)", annotation_font_color="#dc2626")
            fig_trend.update_xaxes(showgrid=False)
            fig_trend.update_yaxes(showgrid=True, gridcolor='#94A3B8', gridwidth=2, zeroline=False)
            st.plotly_chart(fig_trend, use_container_width=True, theme=None)
            
            c1, c2, c3 = st.columns(3)
            c1.markdown(f'<div class="info-card"><span style="font-size:15px; font-weight:800; color:#1E293B; text-transform:uppercase; letter-spacing:0.5px;">Avg Future Temp</span><br><span style="font-size:32px; color:#1e3a8a; font-weight:900;">{forecast_df["temp_avg"].mean():.1f}°C</span></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="info-card"><span style="font-size:15px; font-weight:800; color:#1E293B; text-transform:uppercase; letter-spacing:0.5px;">Avg Future Wind</span><br><span style="font-size:32px; color:#1e3a8a; font-weight:900;">{forecast_df["wind_speed"].mean():.1f} <span style="font-size:18px;">m/s</span></span></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="info-card"><span style="font-size:15px; font-weight:800; color:#1E293B; text-transform:uppercase; letter-spacing:0.5px;">Max Predicted PM2.5</span><br><span style="font-size:32px; color:#dc2626; font-weight:900;">{forecast_df["predicted_pm25"].max():.1f}</span></div>', unsafe_allow_html=True)

            st.markdown("""
<div style="margin-top: 25px;">
<h5 style="margin-bottom:12px; color:#1E293B; font-weight:800;">📊 PM2.5 Level Guide (µg/m³)</h5>
<div style="display: flex; gap: 15px; text-align: center; font-size: 14px;">
<div class="aqi-badge" style="background-color:#F0FDF4; border: 2px solid #86EFAC; color:#14532D;">0 - 25<br><span style="font-size:16px; font-weight:900;">Good</span></div>
<div class="aqi-badge" style="background-color:#FEFCE8; border: 2px solid #FDE047; color:#713F12;">26 - 50<br><span style="font-size:16px; font-weight:900;">Moderate</span></div>
<div class="aqi-badge" style="background-color:#FEF2F2; border: 2px solid #FCA5A5; color:#7F1D1D;">51 - 100<br><span style="font-size:16px; font-weight:900;">Unhealthy</span></div>
<div class="aqi-badge" style="background-color:#FAF5FF; border: 2px solid #D8B4FE; color:#581C87;">100+<br><span style="font-size:16px; font-weight:900;">Hazardous</span></div>
</div>
</div>
""", unsafe_allow_html=True)

        with col_right:
            daily_forecast = forecast_df.groupby(forecast_df['datetime'].dt.date).agg({'predicted_pm25': 'max'}).head(5)
            forecast_html = "".join([f"<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; font-size:15px; font-weight:600; color:#1E293B; border-bottom:1px solid #F1F5F9; padding-bottom:10px;'><span style='color:#475569;'>{row.name.strftime('%d %A')}</span><span style='color:{'#dc2626' if row['predicted_pm25']>50 else '#1e3a8a'}; font-weight:800;'>PM2.5: {row['predicted_pm25']:.0f} µg/m³</span></div>" for _, row in daily_forecast.iterrows()])

            st.markdown(f"""
<div class="glass-card side-panel">
<div>
<p style="text-align:center; color:#64748B; font-size:12px; text-transform:uppercase; letter-spacing:1px; margin-bottom:0; font-weight:700;">🕒 Last Updated: {current_data['fetch_time']}</p>
<h1 style="text-align:center; margin-top:5px; margin-bottom:0; font-size:54px; font-weight:900; color:#1e3a8a !important;">{current_data['temp']}°C</h1>
<p style="text-align:center; font-weight:600; font-size:16px; margin-top:5px; color:#334155;">{current_data['desc']}</p>
<div style="background-color:#F8FAFC; border-radius:12px; padding:15px; display:flex; justify-content:space-around; margin: 25px 0; font-weight:700; color:#1E293B; font-size:14px; border: 1px solid #E2E8F0;">
<span>💨 {current_data['wind_speed']} m/s</span>
<span>💧 {current_data['humidity']}%</span>
</div>
<h4 style="margin-bottom:15px; border-top:2px solid #E2E8F0; padding-top:15px; color:#1E293B; font-weight:800;">Forecast Summary</h4>
{forecast_html}
</div>

<div style="margin-top: 20px; background: linear-gradient(135deg, #FFF5F5 0%, #FED7D7 100%); border: 2px solid #FEB2B2; border-radius: 12px; padding: 15px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
<h5 style="color: #C53030; font-weight: 800; margin-top: 0; margin-bottom: 5px;">🔥 Regional Fire Alert</h5>
<p style="margin: 0; font-size: 14px; color: #9B2C2C; font-weight: 600;">Active Hotspots: <span style="font-size: 22px; font-weight: 900; color: #E53E3E;">{live_fire_count}</span></p>
<p style="margin: 5px 0 0 0; font-size: 11px; color: #C53030;">Integrated into XGBoost Predictions</p>
</div>

<div style="margin-top: 20px;">
<a href="https://website01.mch.mfu.ac.th/mch-index.html" target="_blank" style="display: block; width: 100%; background-color: #FEF2F2; color: #dc2626; padding: 15px 0; border: 2px solid #FECACA; border-radius: 12px; text-align: center; text-decoration: none; font-weight: 800; font-size: 15px; transition: transform 0.2s ease;">
🏥 MFU Medical Center
</a>
</div>
</div>
""", unsafe_allow_html=True)
    else: st.info("Loading API Data...")

with tab2:
    st.markdown("<h4 style='color:#1E293B; font-weight:800; margin-bottom:20px;'>Historical Local Data (CSV)</h4>", unsafe_allow_html=True)
    df_history = get_historical_data()
    if not df_history.empty:
        col_t1, col_t2 = st.columns(2)
        y_col = "PM25" if "PM25" in df_history.columns else ("PM2.5" if "PM2.5" in df_history.columns else ("pm25" if "pm25" in df_history.columns else None))
        
        with col_t1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            if y_col:
                fig_hist = px.line(df_history, x="Date", y=y_col, title="PM2.5 Over Time", labels={"Date": "Date", "PM25": "PM2.5 (µg/m³)", "PM2.5": "PM2.5 (µg/m³)"})
                fig_hist.update_traces(
                    line=dict(color='#1e3a8a', width=3),
                    hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>PM2.5:</b> %{y:.1f} µg/m³<extra></extra>"
                )
                fig_hist.update_layout(height=380, margin=dict(l=40, r=20, t=50, b=40), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color='#334155', size=13), title_font=dict(size=18, color='#1E293B', family="Inter"), hoverlabel=dict(bgcolor="white", font_size=13, font_family="Inter"))
                fig_hist.update_xaxes(showgrid=False)
                fig_hist.update_yaxes(showgrid=True, gridcolor='#94A3B8', gridwidth=2)
                st.plotly_chart(fig_hist, use_container_width=True, theme=None)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_t2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            if y_col and "Temp_avg" in df_history.columns:
                fig_scatter = px.scatter(df_history, x="Temp_avg", y=y_col, color="Humidity_avg" if "Humidity_avg" in df_history.columns else None, title="Temp vs PM2.5 Correlation", color_continuous_scale="Blues", labels={"Temp_avg": "Temperature (°C)", "PM25": "PM2.5 (µg/m³)", "PM2.5": "PM2.5 (µg/m³)", "Humidity_avg": "Humidity (%)"})
                fig_scatter.update_traces(
                    hovertemplate="<b>Temp:</b> %{x:.1f}°C<br><b>PM2.5:</b> %{y:.1f} µg/m³<br><b>Humidity:</b> %{marker.color:.1f}%<extra></extra>"
                )
                fig_scatter.update_layout(height=380, margin=dict(l=40, r=20, t=50, b=40), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color='#334155', size=13), title_font=dict(size=18, color='#1E293B', family="Inter"), hoverlabel=dict(bgcolor="white", font_size=13, font_family="Inter"))
                fig_scatter.update_xaxes(showgrid=True, gridcolor='#94A3B8', gridwidth=2)
                fig_scatter.update_yaxes(showgrid=True, gridcolor='#94A3B8', gridwidth=2)
                st.plotly_chart(fig_scatter, use_container_width=True, theme=None)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("<h4 style='color:#C53030; font-weight:800; margin-bottom:10px;'>🔥 Fire Count vs. Fire Pressure Analysis</h4>", unsafe_allow_html=True)
        
        f_count_col = 'Fire_Count' if 'Fire_Count' in df_history.columns else ('Fire_count' if 'Fire_count' in df_history.columns else None)
        f_press_col = 'Fire_Pressure' if 'Fire_Pressure' in df_history.columns else ('Fire_pressure' if 'Fire_pressure' in df_history.columns else None)
        
        if f_count_col and f_press_col:
            fig_fire = go.Figure()
            fig_fire.add_trace(go.Bar(x=df_history['Date'], y=df_history[f_count_col], name='Fire Count (Hotspots)', marker_color='#F6AD55'))
            fig_fire.add_trace(go.Scatter(x=df_history['Date'], y=df_history[f_press_col], name='Fire Pressure', yaxis='y2', line=dict(color='#E53E3E', width=2)))
            fig_fire.update_layout(
                height=400, margin=dict(l=40, r=40, t=20, b=40), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(title='Total Fire Count', showgrid=False),
                yaxis2=dict(title='Fire Pressure Index', overlaying='y', side='right', showgrid=False),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                font=dict(family="Inter", color='#334155', size=13)
            )
            st.plotly_chart(fig_fire, use_container_width=True, theme=None)
        else:
            st.warning("Fire factors columns not found in the historical data.")
        st.markdown('</div>', unsafe_allow_html=True)
    else: st.warning("No data found in local CSV database.")

with tab3:
    st.markdown("<br>", unsafe_allow_html=True) 
    if current_data and not forecast_df.empty:
        # Construct Input for the 4 Baseline Models (8 and 7 features)
        app_features = ['pressure_avg', 'temp_avg', 'humidity_avg', 'precipitation', 'sunshine', 'wind_direct', 'wind_speed', 'pm25_lag1']
        model_features = ['Pressure_avg', 'Temp_avg', 'Humidity_avg', 'Precipitation', 'Sunshine', 'Wind_direct', 'Wind_speed', 'pm25_lag1']
        
        current_input_base = forecast_df.iloc[[0]][app_features].copy()
        current_input_base.columns = model_features
        weather_features = ['Pressure_avg', 'Temp_avg', 'Humidity_avg', 'Precipitation', 'Sunshine', 'Wind_direct', 'Wind_speed']
        current_input_7 = current_input_base[weather_features]

        # Predict Base Models (isinstance(str) အစား None ဖြင့်စစ်မည်)
        live_xgb_base = xgb_base_model.predict(current_input_base)[0] if xgb_base_model is not None else 0
        live_rf_base = rf_base_model.predict(current_input_base)[0] if rf_base_model is not None else 0
        live_svr = svr_model.predict(current_input_7)[0] if svr_model is not None else 0
        live_mlr = mlr_model.predict(current_input_7)[0] if mlr_model is not None else 0
        
        # Predict Champion XGBoost Fire Model (18 Features) with REAL API Data
        current_input_fire = current_input_base.copy()
        current_input_fire['pm25_lag2'] = current_input_fire['pm25_lag1']
        current_input_fire['pm25_lag3'] = current_input_fire['pm25_lag1']
        current_input_fire['pm25_3Day_Avg'] = current_input_fire['pm25_lag1']
        current_input_fire['Fire_Count'] = live_fire_count
        current_input_fire['Fire_Pressure'] = live_p0
        current_input_fire['Fire_Pressure_Lag1'] = live_p1
        current_input_fire['Fire_Pressure_Lag2'] = live_p2
        current_input_fire['Fire_Pressure_3Day_Avg'] = live_p_avg
        current_input_fire['Month'] = datetime.now().month
        current_input_fire['Is_Burning_Season'] = 1 if datetime.now().month in [2, 3, 4, 5] else 0
        
        final_18_cols = [
            'Pressure_avg', 'Temp_avg', 'Humidity_avg', 'Precipitation', 'Sunshine', 
            'Wind_direct', 'Wind_speed', 'pm25_lag1', 'pm25_lag2', 'pm25_lag3', 
            'pm25_3Day_Avg', 'Fire_Count', 'Fire_Pressure', 'Fire_Pressure_Lag1', 
            'Fire_Pressure_Lag2', 'Fire_Pressure_3Day_Avg', 'Month', 'Is_Burning_Season'
        ]
        
        if not isinstance(xgb_fire_model, str):
            try:
                log_pred_fire = xgb_fire_model.predict(current_input_fire[final_18_cols])[0]
                live_xgb_fire = np.expm1(log_pred_fire)
            except:
                live_xgb_fire = 0
        else:
            live_xgb_fire = 0

        st.markdown(f"""
<div class="glass-card" style="margin-bottom: 30px;">
<h3 style="margin-top: 0; color:#1E293B; font-weight:800;">🥊 5-Model Defense Showdown</h3>
<p style="font-weight: 600; font-size: 15px; color:#475569;">Scientific justification for selecting XGBoost with Fire Factors over standard ML approaches.</p>

<div style="display: flex; gap: 20px; margin-top: 25px;">
<div class="model-card mc-xgb-fire" style="background-color: #F0FDF4;">
<p style="margin: 0; font-weight: 700; font-size: 14px; color:#166534; text-transform:uppercase;">🏆 The Ultimate Champion</p>
<h4 style="margin: 5px 0 10px 0; color:#1E293B; font-weight:800;">XGBoost (Fire Factors)</h4>
<h1 style="margin: 0; color: #15803d !important; font-size: 42px; font-weight:900;">{live_xgb_fire:.1f} <span style="font-size: 16px; color: #64748B;">µg/m³</span></h1>
</div>
<div class="model-card mc-xgb-base">
<p style="margin: 0; font-weight: 700; font-size: 14px; color:#475569; text-transform:uppercase;">🌤️ Weather Only Ensembles</p>
<h4 style="margin: 5px 0 10px 0; color:#1E293B; font-weight:800;">XGBoost (Base)</h4>
<h1 style="margin: 0; color: #3b82f6 !important; font-size: 42px; font-weight:900;">{live_xgb_base:.1f} <span style="font-size: 16px; color: #64748B;">µg/m³</span></h1>
</div>
</div>

<div style="display: flex; gap: 20px; margin-top: 20px;">
<div class="model-card mc-rf-base">
<p style="margin: 0; font-weight: 700; font-size: 14px; color:#475569; text-transform:uppercase;">🌤️ Weather Only Ensembles</p>
<h4 style="margin: 5px 0 10px 0; color:#1E293B; font-weight:800;">Random Forest (Base)</h4>
<h1 style="margin: 0; color: #475569 !important; font-size: 42px; font-weight:900;">{live_rf_base:.1f} <span style="font-size: 16px; color: #64748B;">µg/m³</span></h1>
</div>
<div class="model-card mc-svr">
<p style="margin: 0; font-weight: 700; font-size: 14px; color:#475569; text-transform:uppercase;">📈 SVR Baseline</p>
<h4 style="margin: 5px 0 10px 0; color:#1E293B; font-weight:800;">Support Vector Reg.</h4>
<h1 style="margin: 0; color: #64748B !important; font-size: 42px; font-weight:900;">{live_svr:.1f} <span style="font-size: 16px; color: #64748B;">µg/m³</span></h1>
</div>
</div>

<div style="display: flex; gap: 20px; margin-top: 20px;">
<div class="model-card mc-mlr">
<p style="margin: 0; font-weight: 700; font-size: 14px; color:#475569; text-transform:uppercase;">📏 Linear Baseline</p>
<h4 style="margin: 5px 0 10px 0; color:#1E293B; font-weight:800;">Multiple Linear Reg.</h4>
<h1 style="margin: 0; color: #94A3B8 !important; font-size: 42px; font-weight:900;">{live_mlr:.1f} <span style="font-size: 16px; color: #64748B;">µg/m³</span></h1>
</div>
<div style="flex: 1;"></div>
</div>
</div>
""", unsafe_allow_html=True)

    col_metrics, col_charts = st.columns([1, 2])
    with col_metrics:
        st.markdown("""
<div class="glass-card" style="height: 100%;">
<h4 style="margin-top: 0; color:#1E293B; font-weight:800;">📊 The 5-Model Defense</h4>
<div style="margin-top: 25px; font-weight: 600; color:#334155;">

<p style="margin-bottom: 8px; font-size:14px; text-transform:uppercase; color:#475569; font-weight:700;">Accuracy (R²)</p>
<div style="display: flex; justify-content: space-between; margin-bottom: 8px; align-items:center;">
<span style="color: #15803d; font-weight:800; font-size:16px;">XGBoost (Fire)</span> <span style="font-size:18px; font-weight:900; color:#1E293B;">89.00%</span>
</div>
<div style="display: flex; justify-content: space-between; margin-bottom: 8px; align-items:center;">
<span style="color: #1e3a8a; font-weight:700; font-size:15px;">XGBoost (Base)</span> <span style="font-size:16px; color:#1E293B;">86.52%</span>
</div>
<div style="display: flex; justify-content: space-between; margin-bottom: 8px; align-items:center;">
<span style="color: #475569; font-weight:600; font-size:14px;">Random Forest (Base)</span> <span style="font-size:14px; color:#475569;">86.08%</span>
</div>
<div style="display: flex; justify-content: space-between; margin-bottom: 8px; align-items:center;">
<span style="color: #64748B; font-weight:600; font-size:14px;">SVR</span> <span style="font-size:14px; color:#475569;">68.97%</span>
</div>
<div style="display: flex; justify-content: space-between; margin-bottom: 25px; border-bottom: 1px solid #E2E8F0; padding-bottom: 15px; align-items:center;">
<span style="color: #94A3B8; font-weight:600; font-size:14px;">Linear Reg</span> <span style="font-size:14px; color:#475569;">51.95%</span>
</div>

<p style="margin-bottom: 8px; font-size:14px; text-transform:uppercase; color:#475569; font-weight:700;">Error Rate (MAE)</p>
<div style="display: flex; justify-content: space-between; margin-bottom: 8px; align-items:center;">
<span style="color: #15803d; font-weight:800; font-size:16px;">XGBoost (Fire)</span> <span style="font-size:18px; font-weight:900; color:#1E293B;">5.10</span>
</div>
<div style="display: flex; justify-content: space-between; margin-bottom: 8px; align-items:center;">
<span style="color: #1e3a8a; font-weight:700; font-size:15px;">XGBoost (Base)</span> <span style="font-size:16px; color:#1E293B;">5.38</span>
</div>
<div style="display: flex; justify-content: space-between; margin-bottom: 8px; align-items:center;">
<span style="color: #475569; font-weight:600; font-size:14px;">Random Forest (Base)</span> <span style="font-size:14px; color:#475569;">5.46</span>
</div>
<div style="display: flex; justify-content: space-between; margin-bottom: 8px; align-items:center;">
<span style="color: #64748B; font-weight:600; font-size:14px;">SVR</span> <span style="font-size:14px; color:#475569;">7.94</span>
</div>
<div style="display: flex; justify-content: space-between; margin-bottom: 25px; border-bottom: 1px solid #E2E8F0; padding-bottom: 15px; align-items:center;">
<span style="color: #94A3B8; font-weight:600; font-size:14px;">Linear Reg</span> <span style="font-size:14px; color:#475569;">10.98</span>
</div>
</div>

<div style="background-color:#F8FAFC; padding:15px; border-radius:10px; border-left:4px solid #1e3a8a;">
<p style="font-weight: 500; font-size: 14px; line-height: 1.6; color:#334155; margin:0;">
By tracking from baseline MLR up to our Champion XGBoost, we prove that 1) MFU Valley weather requires complex non-linear architectures (Boosting vs Bagging) and 2) Hotspot data is absolutely critical to cross the 86% accuracy threshold.
</p>
</div>
</div>
""", unsafe_allow_html=True)

    with col_charts:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        
        models_list = ["XGB (Fire)", "XGB (Base)", "RF (Base)", "SVR", "MLR"]
        r2_vals = [89.00, 86.52, 86.08, 68.97, 51.95]
        mae_vals = [5.10, 5.38, 5.46, 7.94, 10.98]
        colors_5 = ['#15803d', '#1e3a8a', '#475569', '#64748b', '#cbd5e1']
        
        df_r2 = pd.DataFrame({"Model": models_list, "Accuracy": r2_vals})
        fig_r2 = px.bar(df_r2, x="Model", y="Accuracy", color="Model", text_auto='.2f', color_discrete_sequence=colors_5)
        fig_r2.update_traces(hovertemplate="<b>Model:</b> %{x}<br><b>Accuracy:</b> %{y:.2f}%<extra></extra>")
        fig_r2.update_layout(height=380, showlegend=False, margin=dict(l=20, r=20, t=50, b=0), yaxis_range=[40, 95], title="Accuracy Comparison (5 Models)", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color='#334155', size=13), title_font=dict(size=16, color='#1E293B', family="Inter", weight="bold"), hoverlabel=dict(bgcolor="white", font_size=13, font_family="Inter"))
        fig_r2.update_yaxes(showgrid=True, gridcolor='#94A3B8', gridwidth=2)
        c1.plotly_chart(fig_r2, use_container_width=True, theme=None)
        
        df_mae = pd.DataFrame({"Model": models_list, "MAE": mae_vals})
        fig_mae = px.bar(df_mae, x="Model", y="MAE", color="Model", text_auto='.2f', color_discrete_sequence=colors_5)
        fig_mae.update_traces(hovertemplate="<b>Model:</b> %{x}<br><b>MAE:</b> %{y:.2f}<extra></extra>")
        fig_mae.update_layout(height=380, showlegend=False, margin=dict(l=20, r=20, t=50, b=0), title="Error Rate Comparison", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color='#334155', size=13), title_font=dict(size=16, color='#1E293B', family="Inter", weight="bold"), hoverlabel=dict(bgcolor="white", font_size=13, font_family="Inter"))
        fig_mae.update_yaxes(showgrid=True, gridcolor='#94A3B8', gridwidth=2)
        c2.plotly_chart(fig_mae, use_container_width=True, theme=None)
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div style="margin-top: 50px; padding: 30px 0 20px 0; border-top: 2px solid #E2E8F0; text-align: center;">
<div style="margin-bottom: 25px;">
<a href="mailto:6631501189@lamduan.mfu.ac.th" style="display: inline-block; background-color: #F8FAFC; border: 2px solid #E2E8F0; color: #1e3a8a; padding: 12px 35px; border-radius: 50px; text-decoration: none; font-weight: 800; font-size: 15px; transition: transform 0.2s ease; box-shadow: 0 4px 6px rgba(0,0,0,0.02);">
✉️ Contact Us
</a>
</div>
<p style="margin: 0; font-size: 16px; color: #475569; font-weight: 600; letter-spacing: 0.5px;">
<span style="color:#1e3a8a; font-weight:900;">The Outliers</span> (CPE Senior Project) 
&nbsp;&nbsp;|&nbsp;&nbsp; 
<span style="font-weight:900; color:#1E293B;">Advisor:</span> Aj. Khwunta Kirimasthong
</p>
</div>
""", unsafe_allow_html=True)