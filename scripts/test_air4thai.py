import requests
import json

print("🇹🇭 THE OUTLIERS: Fetching Official Thailand PM2.5 Data (Air4Thai)")

def fetch_real_pm25():
    # Chiang Rai Station ID (e.g., 73t is Mueang Chiang Rai)
    url = "http://air4thai.pcd.go.th/services/getNewAQI_JSON.php?stationID=73t"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            # Extracting the latest PM2.5 value
            pm25_real = data['AQILast']['PM25']['value']
            time_updated = data['AQILast']['date'] + " " + data['AQILast']['time']
            
            print("="*50)
            print(f"✅ SUCCESS! Connected to Air4Thai (PCD Thailand)")
            print(f"📍 Station: Mueang Chiang Rai (Near MFU)")
            print(f"🕒 Last Updated: {time_updated}")
            print(f"🎯 Real PM2.5 Value: {pm25_real} µg/m³")
            print("="*50)
            return pm25_real
        else:
            print("❌ API Connection Failed.")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    pm25_lag1_real = fetch_real_pm25()