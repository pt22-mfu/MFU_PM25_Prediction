import pandas as pd
import mysql.connector
import os

# ==========================================
# CONFIGURATION
# ==========================================
CSV_FILENAME = 'pm25_data.csv' 
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'pm25_project'
}

def run_etl_pipeline():
    print("🚀 STARTING FINAL ETL PIPELINE...")

    # 1. READ CSV (Read as text first to avoid format errors)
    if not os.path.exists(CSV_FILENAME):
        print(f"❌ Error: File '{CSV_FILENAME}' not found.")
        return
    
    df = pd.read_csv(CSV_FILENAME, dtype=str)
    df.columns = df.columns.str.strip() # Remove spaces from headers
    
    print(f"   📂 Raw file loaded: {len(df)} rows found.")

    # 2. FIX DATE FORMAT
    print("   🔧 Normalizing Date Format...")
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=False, errors='coerce')
    
    # Drop rows where date is completely broken (if any)
    df = df.dropna(subset=['Date'])
    
    # Convert to MySQL Standard 'YYYY-MM-DD'
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

    # 3. CLEAN & STANDARDIZE
    print("   🧹 Cleaning numerical data...")
    cols_to_numeric = ['Pressure_avg', 'Temp_avg', 'Humidity_avg', 'Precipitation', 
                       'Sunshine', 'Wind_direct', 'Wind_speed', 'PM25']
    
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # ==========================================
    # ==========================================
    print("   ⏳ Generating Time-Lag Feature (pm25_lag1)...")
    df = df.sort_values(by='Date') 
    df['pm25_lag1'] = df['PM25'].shift(1) 
    df = df.dropna(subset=['pm25_lag1'])

    # 4. CONNECT TO DATABASE
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        print("   ✅ Connected to Database.")
    except Exception as e:
        print(f"❌ Database Connection Failed: {e}")
        return

    # 5. WIPE OLD DATA (Start Fresh)
    cursor.execute("TRUNCATE TABLE pollution_logs")
    print("   🗑️  Old data cleared.")

    # 6. UPLOAD DATA
    print(f"   📥 Uploading {len(df)} rows... (Please wait)")
    
    query = """
    INSERT INTO pollution_logs 
    (log_date, pressure_avg, temp_avg, humidity_avg, precipitation, sunshine, wind_direct, wind_speed, pm25_lag1, pm25)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    for index, row in df.iterrows():
        values = (
            row['Date'],
            row['Pressure_avg'],
            row['Temp_avg'],
            row['Humidity_avg'],
            row['Precipitation'],
            row['Sunshine'],
            row['Wind_direct'],
            row['Wind_speed'],
            row['pm25_lag1'], 
            row['PM25']
        )
        cursor.execute(query, values)

    conn.commit()
    cursor.close()
    conn.close()
    print("   🎉 ETL Pipeline Completed Successfully!")

if __name__ == "__main__":
    run_etl_pipeline()