import pandas as pd
import joblib
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# CONFIGURATION
# ==========================================
CSV_FILENAME = 'pm25_data.csv'
MODEL_FILENAME = 'svr_pm25_model.pkl'

def train_svr_model():
    print("🧠 STARTING ADVANCED ML PROTOCOL (SUPPORT VECTOR REGRESSION)...")

    # 1. FETCH DATA FROM CSV
    if not os.path.exists(CSV_FILENAME):
        print(f"  ❌ Error: '{CSV_FILENAME}' file not found.")
        return

    try:
        df = pd.read_csv(CSV_FILENAME)
        df.columns = df.columns.str.strip()
        print(f"  ✅ Data Loaded: {len(df)} records from {CSV_FILENAME}.")
    except Exception as e:
        print(f"  ❌ File Read Error: {e}")
        return

    # Drop rows with missing PM25
    df = df.dropna(subset=['PM25']).reset_index(drop=True)

    # 2. PREPARE FEATURES & TARGET (REMOVED PM25_LAG1)
    feature_cols = ['Pressure_avg', 'Temp_avg', 'Humidity_avg', 'Precipitation', 
                    'Sunshine', 'Wind_direct', 'Wind_speed']
    
    missing_cols = [col for col in feature_cols + ['PM25'] if col not in df.columns]
    if missing_cols:
        print(f"  ❌ Error: Missing columns in CSV -> {missing_cols}")
        return

    X = df[feature_cols]
    y = df['PM25']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. TRAIN SVR MODEL (With built-in Standard Scaler)
    print("  🚀 Training Support Vector Regressor (Weather-Only)...")
    svr_model = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, gamma='scale'))
    svr_model.fit(X_train, y_train)

    # 4. EVALUATE PERFORMANCE
    print("  📊 Evaluating Model Accuracy...")
    predictions = svr_model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    # 5. PRINT REPORT
    print("\n" + "="*40)
    print("      MODEL PERFORMANCE REPORT      ")
    print("========================================")
    print(f"  Model Type:      Support Vector Regression (RBF)")
    print(f"  Input Features:  {len(feature_cols)} Factors (NO LAG)")
    print(f"  Training Data:   {len(X_train)} rows")
    print(f"  Test Data:       {len(X_test)} rows")
    print("-" * 40)
    print(f"  🎯 R-Squared (Accuracy): {r2*100:.2f}%")
    print(f"  📉 Mean Absolute Error:  {mae:.2f} µg/m³")
    print(f"  📉 Root Mean Sq Error:   {rmse:.2f} µg/m³")
    print("========================================\n")

    # 6. SAVE MODEL
    joblib.dump(svr_model, MODEL_FILENAME)
    print(f"  💾 Model successfully saved as '{MODEL_FILENAME}'")
    print("  ✅ Phase 4 (SVR) Completed!")

if __name__ == "__main__":
    train_svr_model()