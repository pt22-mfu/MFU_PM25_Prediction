import pandas as pd
import numpy as np
import requests
import joblib
import math
from datetime import datetime, timedelta
from io import StringIO

print("🔬 THE OUTLIERS: V7 MODEL LIVE TEST (TODAY'S PREDICTION)")

# --- CONFIG ---
WEATHER_API_KEY = "b5a0c28f2b79a51d156755c60818dcae"
NASA_KEY = "5952b9b36e2cd0e96a1523d6db0758dc"
LAT, LON = 20.045, 99.895

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def fetch_nasa_fire(date_str):
    bbox = "99.4,19.6,100.4,20.5"
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{NASA_KEY}/VIIRS_SNPP_NRT/{bbox}/1/{date_str}"
    try:
        res = requests.get(url, timeout=15)
        if res.status_code == 200:
            df = pd.read_csv(StringIO(res.text))
            if df.empty: return 0, 0.0
            df['dist'] = df.apply(lambda r: haversine(LAT, LON, r['latitude'], r['longitude']), axis=1)
            df['pres'] = df['bright_ti4'] / (df['dist'] + 1)**2
            return len(df), round(df['pres'].sum(), 2)
        return 0, 0.0
    except: return 0, 0.0

try:
    print("⏳ Fetching Live APIs (Air4Thai, NASA, OpenWeather)...")
    
    # 1. Fetch Air4Thai
    res_air = requests.get("http://air4thai.pcd.go.th/services/getNewAQI_JSON.php?stationID=73t", timeout=10)
    
    # 🚨 THE FIX: Convert String to Float, and handle "-" errors
    try:
        pm25_lag1 = float(res_air.json()['AQILast']['PM25']['value'])
    except ValueError:
        # Sensor ခဏပျက်နေရင် သုံးဖို့ ယာယီဂဏန်း (Fallback)
        pm25_lag1 = 148.0 
        
    # ယာယီအားဖြင့် လွန်ခဲ့တဲ့ ၂ ရက်စာကို သတ်မှတ်ထားသည် (စမ်းသပ်ရန်)
    pm25_lag2 = 120.0 
    pm25_lag3 = 95.0
    pm25_3day_avg = (pm25_lag1 + pm25_lag2 + pm25_lag3) / 3

    # 2. Fetch OpenWeather
    w = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={WEATHER_API_KEY}&units=metric").json()
    pressure_avg = w['main']['pressure']
    temp_avg = w['main']['temp']
    humidity_avg = w['main']['humidity']
    precip = w.get('rain', {}).get('1h', 0.0)
    sunshine = 5.0
    wind_direct = w['wind'].get('deg', 0)
    wind_speed = w['wind']['speed']

    # 3. Fetch NASA 3-Day History
    today = datetime.now()
    d0 = today.strftime("%Y-%m-%d")
    d1 = (today - timedelta(days=1)).strftime("%Y-%m-%d")
    d2 = (today - timedelta(days=2)).strftime("%Y-%m-%d")

    c0, p0 = fetch_nasa_fire(d0)
    c1, p1 = fetch_nasa_fire(d1)
    c2, p2 = fetch_nasa_fire(d2)
    p_3day_avg = round((p0 + p1 + p2) / 3, 2)

    month = today.month
    is_burning = 1 if month in [2, 3, 4, 5] else 0

    # 4. Prepare Features for V7
    feature_cols = [
        'Pressure_avg', 'Temp_avg', 'Humidity_avg', 'Precipitation', 
        'Sunshine', 'Wind_direct', 'Wind_speed', 
        'pm25_lag1', 'pm25_lag2', 'pm25_lag3', 'pm25_3Day_Avg',
        'Fire_Count', 'Fire_Pressure', 'Fire_Pressure_Lag1', 
        'Fire_Pressure_Lag2', 'Fire_Pressure_3Day_Avg',
        'Month', 'Is_Burning_Season'
    ]

    features = [
        pressure_avg, temp_avg, humidity_avg, precip, 
        sunshine, wind_direct, wind_speed,
        pm25_lag1, pm25_lag2, pm25_lag3, pm25_3day_avg,
        c0, p0, p1, p2, p_3day_avg,
        month, is_burning
    ]

    # 5. Load Model and Predict
    model = joblib.load("pm25_model_v7.pkl")
    df_input = pd.DataFrame([features], columns=feature_cols)
    
    # 🚨 Log Hack Reversal
    log_pred = model.predict(df_input)[0]
    real_pred = np.expm1(log_pred)

    print("\n" + "="*50)
    print("      📊 V7 LIVE PREDICTION RESULTS 📊")
    print("="*50)
    print(f"🌡️ Weather -> Temp: {temp_avg}°C | Wind: {wind_speed} m/s")
    print(f"🇹🇭 Air4Thai -> Yesterday's PM2.5 (Lag1): {pm25_lag1} µg/m³")
    print(f"🔥 NASA Fires -> Today: {p0} | Lag1: {p1} | Lag2: {p2}")
    print("-" * 50)
    print(f"🎯 PREDICTED PM2.5 FOR TODAY: {real_pred:.1f} µg/m³")
    print("="*50)

    if real_pred > 100:
        print("⚠️ DANGER: Bowl Effect Trapped Smoke Detected!")

except Exception as e:
    print(f"❌ Error: {e}")