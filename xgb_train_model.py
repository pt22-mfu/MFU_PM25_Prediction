import pandas as pd
import joblib
import numpy as np
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# CONFIGURATION
# ==========================================
CSV_FILENAME = 'pm25_data.csv'
MODEL_FILENAME = 'pm25_model.pkl'

def train_advanced_model():
    print("🧠 STARTING ADVANCED AI TRAINING PROTOCOL (XGBOOST)...")

    # 1. FETCH DATA FROM CSV
    if not os.path.exists(CSV_FILENAME):
        print(f"  ❌ Error: '{CSV_FILENAME}' file not found.")
        return

    try:
        df = pd.read_csv(CSV_FILENAME)
        
        # 🚨 Auto-Clean Column Names (Fixes spaces like 'Sunshine ')
        df.columns = df.columns.str.strip()
        print(f"  ✅ Data Loaded: {len(df)} records from {CSV_FILENAME}.")
    except Exception as e:
        print(f"  ❌ File Read Error: {e}")
        return

    # 2. GENERATE TIME-LAG FEATURE (pm25_lag1) DYNAMICALLY
    # PT ရဲ့ CSV ထဲမှာ မပါတဲ့အတွက် Code ကနေ အလိုအလျောက် ဖန်တီးပေးခြင်းဖြစ်ပါတယ်
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
    
    print("  ⚙️ Generating historical time-lag feature (pm25_lag1)...")
    df['pm25_lag1'] = df['PM25'].shift(1)
    
    # Drop the first row which will have NaN because of shift
    df = df.dropna(subset=['pm25_lag1']).reset_index(drop=True)

    # 3. PREPARE FEATURES & TARGET (Matching PT's Exact Columns)
    feature_cols = ['Pressure_avg', 'Temp_avg', 'Humidity_avg', 'Precipitation', 
                    'Sunshine', 'Wind_direct', 'Wind_speed', 'pm25_lag1']
    
    missing_cols = [col for col in feature_cols + ['PM25'] if col not in df.columns]
    if missing_cols:
        print(f"  ❌ Error: Missing columns in CSV -> {missing_cols}")
        return

    X = df[feature_cols]
    y = df['PM25']

    # Split Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. TRAIN XGBOOST MODEL
    print("  🚀 Training XGBoost Regressor...")
    xgb_model = XGBRegressor(
        n_estimators=200,      
        learning_rate=0.1,    
        max_depth=5,          
        random_state=42
    )
    xgb_model.fit(X_train, y_train)

    # 5. EVALUATE PERFORMANCE
    print("  📊 Evaluating Model Accuracy...")
    predictions = xgb_model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    # 6. PRINT REPORT
    print("\n" + "="*40)
    print("      MODEL PERFORMANCE REPORT      ")
    print("="*40)
    print(f"  Model Type:      XGBoost Regressor")
    print(f"  Input Features:  {len(feature_cols)} Factors (Includes Lag-1)")
    print(f"  Training Data:   {len(X_train)} rows")
    print(f"  Test Data:       {len(X_test)} rows")
    print("-" * 40)
    print(f"  🎯 R-Squared (Accuracy): {r2*100:.2f}%")
    print(f"  📉 Mean Absolute Error:  {mae:.2f} µg/m³")
    print(f"  📉 Root Mean Sq Error:   {rmse:.2f} µg/m³")
    print("="*40 + "\n")

    # 7. SAVE MODEL
    joblib.dump(xgb_model, MODEL_FILENAME)
    print(f"  💾 Model successfully saved as '{MODEL_FILENAME}'")
    print("  ✅ Phase 2 Completed!")

if __name__ == "__main__":
    train_advanced_model()