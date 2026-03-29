import pandas as pd

print("🔄 Loading datasets...")
# 1. Load Weather + PM2.5 Data
df_weather = pd.read_csv("pm25_data.csv")
# Convert 'M/D/YYYY' to proper datetime format
df_weather['Date'] = pd.to_datetime(df_weather['Date'], format='%m/%d/%Y')

# 2. Load NASA Fire Data
df_fire = pd.read_csv("historical_fire_features.csv")
# Convert 'YYYY-MM-DD' to proper datetime format
df_fire['Date'] = pd.to_datetime(df_fire['Date'])

print("🔗 Merging datasets...")
# 3. Merge based on exactly matching dates
df_final = pd.merge(df_weather, df_fire, on='Date', how='inner')

# 4. Save to a new final CSV
df_final.to_csv("final_training_data.csv", index=False)

print("\n✅ Merge Complete! Saved as 'final_training_data.csv'")
print(f"Total Records: {len(df_final)} days")

print("\n📊 Correlation with PM2.5 (The Evidence!):")
corr = df_final[['PM25', 'Fire_Count', 'Fire_Pressure', 'Temp_max', 'Wind_speed']].corr()
print(corr['PM25'].sort_values(ascending=False))