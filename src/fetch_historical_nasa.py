import requests
import pandas as pd
from io import StringIO
import math
import time
from datetime import datetime, timedelta

# --- CONFIGURATION ---
NASA_KEY = "5952b9b36e2cd0e96a1523d6db0758dc"
LAT, LON = 20.045, 99.895  # MFU Center Coordinates
BBOX = "99.4,19.6,100.4,20.5" # 50km Bounding Box

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the distance between two points on Earth."""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * \
        math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def fetch_historical_fires(start_date_str, end_date_str):
    print("="*50)
    print(f"🛰️ THE OUTLIERS: FETCHING NASA HISTORICAL DATA (SP)")
    print(f"📅 Range: {start_date_str} to {end_date_str}")
    print("⚠️ This process will take around 45 minutes. Please DO NOT close the terminal.")
    print("="*50 + "\n")
    
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    current_date = start_date
    historical_records = []

    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        
        # 🔥 FIX: Changed 'VIIRS_SNPP_NRT' to 'VIIRS_SNPP_SP'
        url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{NASA_KEY}/VIIRS_SNPP_SP/{BBOX}/1/{date_str}"
        
        success = False
        retries = 3  # Auto-retry 3 times if connection drops
        
        while retries > 0 and not success:
            try:
                # Increased timeout to 20 seconds for historical data
                response = requests.get(url, timeout=20)
                if response.status_code == 200:
                    df = pd.read_csv(StringIO(response.text))
                    
                    if not df.empty:
                        # Calculate distance and pressure
                        df['distance'] = df.apply(lambda row: haversine(LAT, LON, row['latitude'], row['longitude']), axis=1)
                        df['pressure'] = df['bright_ti4'] / (df['distance'] + 1)**2 
                        
                        daily_count = len(df)
                        daily_pressure = df['pressure'].sum()
                        
                        historical_records.append({
                            "Date": date_str,
                            "Fire_Count": daily_count,
                            "Fire_Pressure": round(daily_pressure, 2)
                        })
                        print(f"✅ {date_str} | Fires: {daily_count} | Pressure: {round(daily_pressure, 2)}")
                    else:
                        # No fires on this day
                        historical_records.append({
                            "Date": date_str,
                            "Fire_Count": 0,
                            "Fire_Pressure": 0.0
                        })
                        print(f"➖ {date_str} | Fires: 0 | Pressure: 0.0")
                    
                    success = True # Exit the retry loop if successful
                    
                else:
                    print(f"⚠️ API Error on {date_str}: {response.status_code}. Retrying...")
                    retries -= 1
                    time.sleep(2) # Wait 2 seconds before retrying
                    
            except Exception as e:
                print(f"❌ Connection Error on {date_str}: {e}. Retrying...")
                retries -= 1
                time.sleep(2) # Wait 2 seconds before retrying
        
        # If all 3 retries failed, log 0 to avoid breaking the sequence
        if not success:
            print(f"⏭️ Skipping {date_str} after 3 failed attempts.")
            historical_records.append({"Date": date_str, "Fire_Count": 0, "Fire_Pressure": 0.0})
            
        # 1-second delay to respect NASA API rate limits
        time.sleep(1)
        current_date += timedelta(days=1)

    # Convert to DataFrame and save
    final_df = pd.DataFrame(historical_records)
    final_df.to_csv("historical_fire_features.csv", index=False)
    print("\n🎉 DONE! Historical Fire Data saved to 'historical_fire_features.csv'")
    return final_df

if __name__ == "__main__":
    # Exact dates matching your pm25_data.csv
    START_DATE = "2016-07-11" 
    END_DATE = "2023-06-30"
    
    df_history = fetch_historical_fires(START_DATE, END_DATE)