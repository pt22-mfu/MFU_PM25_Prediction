import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

print("🔥 THE OUTLIERS: V7 (THE BOWL EFFECT + TRIPLE PM2.5 LAGS)")

# 1. Load Data
df = pd.read_csv("final_training_data.csv")
df.columns = df.columns.str.strip()
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

print("  ⚙️ Engineering the 'Smoke Takes Time' Features...")
# --- THE GENIUS IDEA: TRIPLE LAGS ---
df['pm25_lag1'] = df['PM25'].shift(1)
df['pm25_lag2'] = df['PM25'].shift(2)
df['pm25_lag3'] = df['PM25'].shift(3)
df['pm25_3Day_Avg'] = df[['pm25_lag1', 'pm25_lag2', 'pm25_lag3']].mean(axis=1)

# --- Fire Lags & Seasonality ---
df['Fire_Pressure_Lag1'] = df['Fire_Pressure'].shift(1)
df['Fire_Pressure_Lag2'] = df['Fire_Pressure'].shift(2)
df['Fire_Pressure_3Day_Avg'] = df['Fire_Pressure'].rolling(window=3).mean()
df['Month'] = df['Date'].dt.month
df['Is_Burning_Season'] = df['Month'].isin([2, 3, 4, 5]).astype(int)

# Drop NaN rows created by shifting
df = df.dropna().reset_index(drop=True)

feature_cols = [
    'Pressure_avg', 'Temp_avg', 'Humidity_avg', 'Precipitation', 
    'Sunshine', 'Wind_direct', 'Wind_speed', 
    'pm25_lag1', 'pm25_lag2', 'pm25_lag3', 'pm25_3Day_Avg', # <-- NEW SUPER FEATURES!
    'Fire_Count', 'Fire_Pressure', 'Fire_Pressure_Lag1', 
    'Fire_Pressure_Lag2', 'Fire_Pressure_3Day_Avg',
    'Month', 'Is_Burning_Season'
]

X = df[feature_cols]

# The Math Hack for High Spikes
y_log = np.log1p(df['PM25']) 

X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)
_, _, _, y_test_real = train_test_split(X, df['PM25'], test_size=0.2, random_state=42)

# 3. Train Engine
print("  🚀 Firing up XGBoost with Triple Lags...")
xgb_model = XGBRegressor(
    n_estimators=500,      
    learning_rate=0.04,    
    max_depth=6,          
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train_log)

# 4. Evaluate
predictions_log = xgb_model.predict(X_test)
predictions_real = np.expm1(predictions_log) 

mae = mean_absolute_error(y_test_real, predictions_real)
r2 = r2_score(y_test_real, predictions_real)

print("\n" + "="*50)
print("      🏆 MODEL V7 (TRIPLE LAGS) REPORT 🏆      ")
print("="*50)
print(f"  🎯 R-Squared (Accuracy): {r2*100:.2f}%")
print(f"  📉 Mean Absolute Error:  {mae:.2f} µg/m³")
print("="*50 + "\n")

joblib.dump(xgb_model, "pm25_model_v7.pkl")
print("  💾 SUCCESS! Model V7 saved as 'pm25_model_v7.pkl'")

# Check what the model values most
importances = xgb_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print("\n🔥 TOP 5 DRIVERS FOR V7:")
print(feature_importance_df.head())