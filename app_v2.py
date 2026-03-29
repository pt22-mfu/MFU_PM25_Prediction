import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import requests
import os
import base64
import math
from datetime import datetime, timedelta
from io import StringIO

st.set_page_config(page_title="MFU PM2.5 Analytics", layout="wide")

# --- ORIGINAL CSS STYLING ---
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
.mc-xgb { border-top: 5px solid #1e3a8a; }
.mc-rf { border-top: 5px solid #475569; }
.mc-svr { border-top: 5px solid #94a3b8; }
.mc-mlr { border-top: 5px solid #cbd5e1; }
.aqi-badge { flex:1; padding: 12px; border-radius: 10px; font-weight: 700; text-align: center; }
</style>
""", unsafe_allow_html=True)

# --- CONFIG & NASA LOGIC ---
API_KEY = "b5a0c28f2b79a51d156755c60818dcae"
NASA_KEY = "5952b9b36e2cd0e96a1523d6db0758dc"
LAT, LON = "20.045", "99.895"

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

@st.cache_data(ttl=1800)
def fetch_nasa_live_fire():
    bbox = "99.4,19.6,100.4,20.5"
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{NASA_KEY}/VIIRS_SNPP_NRT/{bbox}/1"
    try:
        res = requests.get(url, timeout=15)
        df = pd.read_csv(StringIO(res.text))
        if df.empty: return 0, 0.0
        df['dist'] = df.apply(lambda r: haversine(float(LAT), float(LON), r['latitude'], r['longitude']), axis=1)
        df['pres'] = df['bright_ti4'] / (df['dist'] + 1)**2
        return len(df), round(df['pres'].sum(), 2)
    except: return 0, 0.0

@st.cache_resource
def load_models():
    xgb = joblib.load('pm25_model.pkl')
    try: rf = joblib.load('rf_pm25_model.pkl')
    except: rf = None
    try: svr = joblib.load('svr_pm25_model.pkl')
    except: svr = None
    try: mlr = joblib.load('mlr_pm25_model.pkl')
    except: mlr = None
    return xgb, rf, svr, mlr

xgb_model, rf_model, svr_model, mlr_model = load_models()

def fetch_weather_and_forecast():
    try:
        cw = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric").json()
        cp = requests.get(f"https://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}").json()
        fw = requests.get(f"https://api.openweathermap.org/data/2.5/forecast?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric").json()
        current = {
            "temp": cw['main']['temp'], "humidity": cw['main']['humidity'], 
            "wind_speed": cw['wind']['speed'], "desc": cw['weather'][0]['description'].title(),
            "pm25_current": cp['list'][0]['components']['pm2_5'],
            "fetch_time": datetime.now().strftime("%d %B %Y, %I:%M %p")
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
    except Exception as e: return None, None

def get_historical_data():
    try:
        df = pd.read_csv("pm25_data.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        return df.sort_values(by='Date', ascending=False).head(100)
    except Exception as e: return pd.DataFrame()

current_data, forecast_df = fetch_weather_and_forecast()
nasa_count, nasa_pressure = fetch_nasa_live_fire()

def get_base64_img(path):
    if os.path.exists(path):
        with open(path, "rb") as f: data = f.read()
        return f"data:image/png;base64,{base64.b64encode(data).decode()}"
    return ""

mfu_base64 = get_base64_img("mfu_logo.png")
icon_base64 = get_base64_img("image_896c03.png")

img_tags = ""
if mfu_base64: img_tags += f'<img src="{mfu_base64}" class="logo-img">'
if icon_base64: img_tags += f'<img src="{icon_base64}" class="icon-img">'

header_html = (
    '<div class="custom-header">'
    + img_tags +
    '<div class="header-text-container">'
    '<h1 class="header-title">Mae Fah Luang University PM2.5 Forecasting Engine</h1>'
    '<p class="header-subtitle">Developed by <span style="color:#1e3a8a;">The Outliers</span></p>'
    '</div></div>'
)
st.markdown(header_html, unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🚀 Live Forecast", "📊 Historical Trends", "🔬 Model Analysis"])

with tab1:
    if current_data and not forecast_df.empty:
        app_features = ['pressure_avg', 'temp_avg', 'humidity_avg', 'precipitation', 'sunshine', 'wind_direct', 'wind_speed', 'pm25_lag1']
        model_features = ['Pressure_avg', 'Temp_avg', 'Humidity_avg', 'Precipitation', 'Sunshine', 'Wind_direct', 'Wind_speed', 'pm25_lag1']
        model_input = forecast_df[app_features].copy(); model_input.columns = model_features
        forecast_df['predicted_pm25'] = xgb_model.predict(model_input).clip(min=0)
        forecast_df['predicted_pm25'] = forecast_df['predicted_pm25'].rolling(2, min_periods=1).mean()
        current_pred = forecast_df.iloc[0]['predicted_pm25']
        
        if current_pred <= 25: status_text, pred_color = "Good", "#16a34a"
        elif current_pred <= 50: status_text, pred_color = "Moderate", "#d97706"
        elif current_pred <= 100: status_text, pred_color = "Unhealthy", "#dc2626"
        else: status_text, pred_color = "Hazardous", "#9333ea"
            
        col_left, col_right = st.columns([7, 3])
        with col_left:
            st.markdown(f'<div class="glass-card hero-card"><p style="margin:0; font-size:14px; font-weight:1000; color:#475569; text-transform:uppercase; letter-spacing:1px;">📍 Mae Fah Luang University</p><h2 style="margin:10px 0; font-size:24px; color:#1E293B; font-weight:800;">Current PM2.5: <span style="color:{pred_color}; font-size:46px;">{current_pred:.1f} <span style="font-size:20px;">µg/m³</span></span></h2><p style="margin:0; font-size:16px; font-weight:600; color:#334155;">Model: <span style="color:#1e3a8a; font-weight:800;">XGBoost</span> &nbsp;|&nbsp; Status: <span style="color:{pred_color}; font-weight:800;">{status_text}</span></p></div>', unsafe_allow_html=True)
            
            st.markdown("<h4 style='margin-top:20px; margin-bottom:10px; color:#1E293B;'>📈 5-Day PM2.5 Forecast Trend</h4>", unsafe_allow_html=True)
            fig_trend = px.line(forecast_df, x='datetime', y='predicted_pm25')
            fig_trend.update_traces(line=dict(color='#1e3a8a', width=4), mode='lines+markers', marker=dict(size=6))
            fig_trend.update_layout(height=340, margin=dict(l=40, r=20, t=10, b=40), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_trend, use_container_width=True, theme=None)
            
            # --- INFO CARDS ---
            c1, c2, c3 = st.columns(3)
            c1.markdown(f'<div class="info-card"><span style="font-size:15px; font-weight:800;">Avg Future Temp</span><br><span style="font-size:32px; color:#1e3a8a; font-weight:900;">{forecast_df["temp_avg"].mean():.1f}°C</span></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="info-card"><span style="font-size:15px; font-weight:800;">Avg Future Wind</span><br><span style="font-size:32px; color:#1e3a8a; font-weight:900;">{forecast_df["wind_speed"].mean():.1f} m/s</span></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="info-card"><span style="font-size:15px; font-weight:800;">Max Predicted PM2.5</span><br><span style="font-size:32px; color:#dc2626; font-weight:900;">{forecast_df["predicted_pm25"].max():.1f}</span></div>', unsafe_allow_html=True)

        with col_right:
            daily_forecast = forecast_df.groupby(forecast_df['datetime'].dt.date).agg({'predicted_pm25': 'max'}).head(5)
            forecast_html = "".join([f"<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; font-size:15px; font-weight:600; color:#1E293B; border-bottom:1px solid #F1F5F9; padding-bottom:10px;'><span style='color:#475569;'>{row.name.strftime('%d %A')}</span><span style='color:{'#dc2626' if row['predicted_pm25']>50 else '#1e3a8a'}; font-weight:800;'>PM2.5: {row['predicted_pm25']:.0f}</span></div>" for _, row in daily_forecast.iterrows()])

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
                
                <div style="margin-top: 20px; background: #FEF2F2; border: 2px solid #FECACA; border-radius: 12px; padding: 15px; text-align: center;">
                    <p style="margin:0; font-size:12px; font-weight:800; color:#991B1B; text-transform:uppercase; letter-spacing:1px;">🔥 NASA Live Fires (50km)</p>
                    <h2 style="margin:5px 0; font-size:36px; color:#dc2626; font-weight:900;">{nasa_count}</h2>
                    <p style="margin:0; font-size:12px; font-weight:600; color:#7F1D1D;">Pressure Index: {nasa_pressure}</p>
                </div>

                <div style="margin-top: 20px;">
                    <a href="https://website01.mch.mfu.ac.th/mch-index.html" target="_blank" style="display: block; width: 100%; background-color: #FEF2F2; color: #dc2626; padding: 15px 0; border: 2px solid #FECACA; border-radius: 12px; text-align: center; text-decoration: none; font-weight: 800; font-size: 15px;">
                    🏥 MFU Medical Center
                    </a>
                </div>
            </div>
            """, unsafe_allow_html=True)

# --- TABS 2 & 3 (UNCHANGED) ---
with tab2:
    st.markdown("<h4 style='color:#1E293B; font-weight:800; margin-bottom:20px;'>Historical Local Data (CSV)</h4>", unsafe_allow_html=True)
    df_history = get_historical_data()
    if not df_history.empty:
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.plotly_chart(px.line(df_history, x="Date", y="PM25", title="PM2.5 Over Time"), use_container_width=True)
        with col_t2:
            st.plotly_chart(px.scatter(df_history, x="Temp_avg", y="PM25", color="Humidity_avg", title="Temp vs PM2.5 Correlation"), use_container_width=True)

with tab3:
    st.markdown("<br>", unsafe_allow_html=True) 
    if current_data and not forecast_df.empty:
        st.markdown('<div class="glass-card"><h3>🥊 Live Model Showdown</h3><p>Real-time prediction comparison.</p></div>', unsafe_allow_html=True)