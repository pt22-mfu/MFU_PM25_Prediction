import mysql.connector
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# CONFIGURATION
# ==========================================
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'pm25_project'
}
MODEL_FILENAME = 'pm25_model.pkl'

def train_advanced_model():
    print("🧠 STARTING ADVANCED AI TRAINING PROTOCOL (XGBOOST)...")

    # 1. FETCH DATA FROM DATABASE
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        query = """
        SELECT 
            pressure_avg, 
            temp_avg, 
            humidity_avg, 
            precipitation, 
            sunshine, 
            wind_direct, 
            wind_speed, 
            pm25_lag1,  
            pm25
        FROM pollution_logs
        ORDER BY log_date ASC
        """
        df = pd.read_sql(query, conn)
        conn.close()
        print(f"   ✅ Data Loaded: {len(df)} records.")
    except Exception as e:
        print(f"   ❌ Database Error: {e}")
        return

    # 2. PREPARE FEATURES & TARGET
    feature_cols = ['pressure_avg', 'temp_avg', 'humidity_avg', 'precipitation', 
                    'sunshine', 'wind_direct', 'wind_speed', 'pm25_lag1']
    
    X = df[feature_cols]
    y = df['pm25']

    # Split Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. TRAIN XGBOOST MODEL
    print("   🚀 Training XGBoost Regressor...")
    xgb_model = XGBRegressor(
        n_estimators=200,      
        learning_rate=0.1,    
        max_depth=5,          
        random_state=42
    )
    xgb_model.fit(X_train, y_train)

    # 4. EVALUATE PERFORMANCE
    print("   📊 Evaluating Model Accuracy...")
    predictions = xgb_model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    # 5. PRINT REPORT
    print("\n" + "="*40)
    print("      MODEL PERFORMANCE REPORT      ")
    print("="*40)
    print(f"   Model Type:      XGBoost Regressor")
    print(f"   Input Features:  {len(feature_cols)} Factors (Includes Lag-1)")
    print(f"   Training Data:   {len(X_train)} rows")
    print(f"   Test Data:       {len(X_test)} rows")
    print("-" * 40)
    print(f"   🎯 R-Squared (Accuracy): {r2*100:.2f}%")
    print(f"   📉 Mean Absolute Error:  {mae:.2f} µg/m³")
    print(f"   📉 Root Mean Sq Error:   {rmse:.2f} µg/m³")
    print("="*40 + "\n")

    # 6. SAVE MODEL
    joblib.dump(xgb_model, MODEL_FILENAME)
    print(f"   💾 Model successfully saved as '{MODEL_FILENAME}'")
    print("   ✅ Phase 2 Completed!")

if __name__ == "__main__":
    train_advanced_model()