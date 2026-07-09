import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os

def prepare_features(historical_data):
    # Sort data by province and date to create lag features
    df = historical_data.copy()
    df = df.sort_values(by=['province', 'date'])
    
    # Create temporal features
    df['month_sin'] = pd.Series(df['month']).apply(lambda x: np.sin(2 * np.pi * x / 12))
    df['month_cos'] = pd.Series(df['month']).apply(lambda x: np.cos(2 * np.pi * x / 12))
    
    # Create lag features (previous month's cases)
    df['cases_lag_1'] = df.groupby('province')['cases'].shift(1)
    df['cases_lag_2'] = df.groupby('province')['cases'].shift(2)
    
    # Fill NA for early periods with 0
    df = df.fillna(0)
    
    return df

def train_model(historical_data, model_path='models/dengue_xgboost.pkl'):
    print("Preparing features for model training...")
    data = prepare_features(historical_data)
    
    features = ['year', 'month', 'population', 'cases_lag_1', 'cases_lag_2', 'month_sin', 'month_cos']
    
    X = data[features]
    y = data['cases']
    
    if len(data) < 10:
        print("Not enough data to train a meaningful model yet. Need at least 10 records.")
        return None
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training XGBoost Regressor...")
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    print(f"Model Evaluation - RMSE: {rmse:.2f}, MAE: {mae:.2f}")
    
    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    return model

def predict_future(model, latest_data):
    """
    Predict next month's cases based on latest available data.
    """
    latest_data = prepare_features(latest_data)
    features = ['year', 'month', 'population', 'cases_lag_1', 'cases_lag_2', 'month_sin', 'month_cos']
    
    # We want to predict for next month
    pred_data = latest_data.copy()
    pred_data['month'] = pred_data['month'] + 1
    pred_data.loc[pred_data['month'] > 12, 'year'] += 1
    pred_data.loc[pred_data['month'] > 12, 'month'] = 1
    
    # recompute sin/cos
    pred_data['month_sin'] = pd.Series(pred_data['month']).apply(lambda x: np.sin(2 * np.pi * x / 12))
    pred_data['month_cos'] = pd.Series(pred_data['month']).apply(lambda x: np.cos(2 * np.pi * x / 12))
    
    # Shift lags
    pred_data['cases_lag_2'] = pred_data['cases_lag_1']
    pred_data['cases_lag_1'] = pred_data['cases']
    
    X_pred = pred_data[features]
    predictions = model.predict(X_pred)
    
    pred_data['predicted_cases'] = predictions
    # Ensure no negative predictions
    pred_data['predicted_cases'] = pred_data['predicted_cases'].clip(lower=0)
    
    return pred_data
