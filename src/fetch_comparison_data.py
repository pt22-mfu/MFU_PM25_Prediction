import requests
import json
import pandas as pd
from io import StringIO
from datetime import datetime, timedelta

# --- CONFIGURATION ---
GISTDA_KEY = "A198E7FBE66F4ED7BB66F635DEBA5A42"
NASA_KEY = "5952b9b36e2cd0e96a1523d6db0758dc"

# MFU Coordinates
LAT, LON = 20.045, 99.895

# --- 🛰️ 1. NASA FIRMS (50km Bounding Box) ---
def fetch_nasa_50km():
    print("🛰️ Fetching NASA FIRMS Data (50km BBOX)...")
    # Bounding Box: West, South, East, North (approx 50km around MFU)
    bbox = "99.4,19.6,100.4,20.5"
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{NASA_KEY}/VIIRS_SNPP_NRT/{bbox}/1"
    
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            return len(df), df
        else:
            print(f"❌ NASA Error: {response.status_code}")
            return 0, None
    except Exception as e:
        print(f"❌ NASA Connection Error: {e}")
        return 0, None

# --- 🛰️ 2. GISTDA (50km Grid - 5 Points) ---
def fetch_gistda_50km_grid():
    print("🛰️ Fetching GISTDA Data (5 Point Grid - 50km total)...")
    
    # Points: Center, North, South, East, West (+/- 0.2 degrees ~ 20km)
    points = [
        (20.045, 99.895), # Center
        (20.245, 99.895), # North
        (19.845, 99.895), # South
        (20.045, 100.095),# East
        (20.045, 99.695)  # West
    ]
    
    total_unique_hotspots = 0
    today = datetime.now().strftime("%Y-%m-%d")
    last_week = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

    for p_lat, p_lon in points:
        url = f"https://api.sphere.gistda.or.th/services/info/disaster-hotspot?&lon={p_lon}&lat={p_lat}&radius=10000&from={last_week}&to={today}&key={GISTDA_KEY}"
        try:
            res = requests.get(url, timeout=10).json()
            for sat in ['terra', 'aqua', 'suomi-npp']:
                if sat in res and res[sat]:
                    total_unique_hotspots += len(res[sat][0].get('data', []))
        except:
            continue
            
    return total_unique_hotspots

# --- 📊 3. RUN COMPARISON ---
def run_comparison():
    print("\n" + "="*40)
    print("🔥 THE OUTLIERS: FIRE DATA COMPARISON")
    print("="*40)
    
    # Run NASA
    nasa_count, nasa_df = fetch_nasa_50km()
    
    # Run GISTDA
    gistda_count = fetch_gistda_50km_grid()
    
    print("\n--- RESULTS ---")
    print(f"📡 NASA FIRMS (50km BBOX): {nasa_count} Hotspots")
    print(f"📡 GISTDA (5-Point Grid):  {gistda_count} Hotspots")
    print("-" * 40)
    
    if nasa_count > gistda_count:
        print("💡 Insight: NASA detected more hotspots (likely from border areas).")
    elif gistda_count > nasa_count:
        print("💡 Insight: GISTDA detected more local hotspots.")
    else:
        print("💡 Insight: Both sources show identical activity.")
    
    # Save NASA results for review
    if nasa_df is not None:
        nasa_df.to_csv("nasa_raw_test.csv", index=False)
        print("\n💾 NASA raw data saved to nasa_raw_test.csv")

if __name__ == "__main__":
    run_comparison()