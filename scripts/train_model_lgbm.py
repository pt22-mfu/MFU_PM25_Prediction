import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

print("⚡ THE OUTLIERS: LIGHTGBM PROTOCOL (PUSHING FOR 90%)")

# 1. Load Data (Using the new Clean Structure)
data_path = "data/final_training_data.csv"
if not os.path.exists(data_path):
    # Fallback in case you run it from the scripts folder directly
    data_path = "../data/final_training_data.csv" 

df = pd.read_csv(data_path)
df.columns = df.columns.str.strip()
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# 2. Engineering Features (Same as V7 Triple Lags)
df['pm25_lag1'] = df['PM25'].shift(1)
df['pm25_lag2'] = df['PM25'].shift(2)
df['pm25_lag3'] = df['PM25'].shift(3)
df['pm25_3Day_Avg'] = df[['pm25_lag1', 'pm25_lag2', 'pm25_lag3']].mean(axis=1)

df['Fire_Pressure_Lag1'] = df['Fire_Pressure'].shift(1)
df['Fire_Pressure_Lag2'] = df['Fire_Pressure'].shift(2)
df['Fire_Pressure_3Day_Avg'] = df['Fire_Pressure'].rolling(window=3).mean()
df['Month'] = df['Date'].dt.month
df['Is_Burning_Season'] = df['Month'].isin([2, 3, 4, 5]).astype(int)

df = df.dropna().reset_index(drop=True)

feature_cols = [
    'Pressure_avg', 'Temp_avg', 'Humidity_avg', 'Precipitation', 
    'Sunshine', 'Wind_direct', 'Wind_speed', 
    'pm25_lag1', 'pm25_lag2', 'pm25_lag3', 'pm25_3Day_Avg',
    'Fire_Count', 'Fire_Pressure', 'Fire_Pressure_Lag1', 
    'Fire_Pressure_Lag2', 'Fire_Pressure_3Day_Avg',
    'Month', 'Is_Burning_Season'
]

X = df[feature_cols]
y_log = np.log1p(df['PM25']) 

X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)
_, _, _, y_test_real = train_test_split(X, df['PM25'], test_size=0.2, random_state=42)

# 3. Train LightGBM Engine
print("  🚀 Firing up LightGBM (Leaf-wise growth)...")
lgb_model = lgb.LGBMRegressor(
    n_estimators=700,         # A bit more trees
    learning_rate=0.03,
    max_depth=-1,             # LightGBM prefers no depth limit, relies on num_leaves
    num_leaves=31,            # Control complexity here
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1                # Hide warnings
)
lgb_model.fit(X_train, y_train_log)

# 4. Evaluate
predictions_log = lgb_model.predict(X_test)
predictions_real = np.expm1(predictions_log) 

mae = mean_absolute_error(y_test_real, predictions_real)
r2 = r2_score(y_test_real, predictions_real)

print("\n" + "="*50)
print("      🌟 LIGHTGBM EXPERIMENT REPORT 🌟      ")
print("="*50)
print(f"  🎯 R-Squared (Accuracy): {r2*100:.2f}%")
print(f"  📉 Mean Absolute Error:  {mae:.2f} µg/m³")
print("="*50 + "\n")

# Save to the models folder (Make sure 'models' folder exists!)
save_path = "models/lgbm_pm25_model.pkl"
if not os.path.exists("models"):
    os.makedirs("models")
joblib.dump(lgb_model, save_path)
print(f"  💾 SUCCESS! Saved as '{save_path}'")