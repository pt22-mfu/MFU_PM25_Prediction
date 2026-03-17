import requests
import json

# ==========================================
# CONFIGURATION
# ==========================================
API_KEY = "b5a0c28f2b79a51d156755c60818dcae"
LAT = "20.045"   # Mae Fah Luang University Latitude
LON = "99.895"   # Mae Fah Luang University Longitude

def get_live_data():
    print("🌐 CONNECTING TO OPENWEATHERMAP API...")

    # 1. Fetch Current Weather Data
    weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
    
    # 2. Fetch Air Pollution Data (for PM2.5)
    pollution_url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"

    try:
        # Request Weather
        w_response = requests.get(weather_url)
        w_data = w_response.json()

        # Request Pollution
        p_response = requests.get(pollution_url)
        p_data = p_response.json()

        if w_response.status_code == 200 and p_response.status_code == 200:
            print("   ✅ Live Data Retrieved Successfully!\n")
            
            # Extracting features for our XGBoost/RF Models
            extracted_data = {
                "temp_avg": w_data['main']['temp'],
                "humidity_avg": w_data['main']['humidity'],
                "pressure_avg": w_data['main']['pressure'],
                "wind_speed": w_data['wind']['speed'],
                "wind_direct": w_data['wind'].get('deg', 0),
                "pm25_current": p_data['list'][0]['components']['pm2_5'],
                # Precipitation might not be in the JSON if it's not currently raining
                "precipitation": w_data.get('rain', {}).get('1h', 0)
            }

            print("--- CURRENT WEATHER AT MFU ---")
            for key, value in extracted_data.items():
                print(f"📍 {key}: {value}")
            
            return extracted_data

        else:
            print(f"   ❌ API Error: {w_data.get('message', 'Unknown Error')}")
            return None

    except Exception as e:
        print(f"   ❌ Connection Error: {e}")
        return None

if __name__ == "__main__":
    get_live_data()