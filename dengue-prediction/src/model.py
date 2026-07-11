import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import logging

try:
    from . import paths
except ImportError:
    import paths

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FEATURES = ['year', 'month', 'population', 'cases_lag_1', 'cases_lag_2',
            'month_sin', 'month_cos', 'province']


def prepare_features(historical_data, province_categories=None):
    """
    Adds temporal + lag features.

    province_categories: if provided, 'province' is cast to a category dtype
    with exactly these categories (so train-time and predict-time encodings
    always match, even if a given batch doesn't contain every province).
    """
    df = historical_data.copy()
    df = df.sort_values(by=['province', 'date'])

    # Create temporal features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Create lag features (previous month's cases) - requires that df
    # actually contains the prior month(s) for each province, otherwise
    # these come out as NaN -> 0, which silently degrades predictions.
    df['cases_lag_1'] = df.groupby('province')['cases'].shift(1)
    df['cases_lag_2'] = df.groupby('province')['cases'].shift(2)

    # Fill NA for early periods (start of the time series) with 0
    df[['cases_lag_1', 'cases_lag_2']] = df[['cases_lag_1', 'cases_lag_2']].fillna(0)

    if province_categories is not None:
        df['province'] = pd.Categorical(df['province'], categories=province_categories)
    else:
        df['province'] = df['province'].astype('category')

    return df


def train_model(historical_data, model_path=None, test_months=3):
    """
    Trains an XGBoost regressor to predict monthly dengue cases per province.

    test_months: number of most-recent months held out as a time-based test
    set. A random split (the previous approach) lets the model "see the
    future" relative to some test rows, which inflates evaluation metrics
    for a forecasting task - so we hold out the tail of the timeline instead.
    """
    if model_path is None:
        model_path = paths.MODEL_FILE

    logger.info("Preparing features for model training...")
    province_categories = sorted(historical_data['province'].unique())
    data = prepare_features(historical_data, province_categories=province_categories)

    if len(data) < 10:
        logger.warning("Not enough data to train a meaningful model yet. Need at least 10 records.")
        return None

    unique_dates = sorted(data['date'].unique())
    if len(unique_dates) <= test_months:
        logger.warning(
            f"Only {len(unique_dates)} distinct months available; falling back to a random split "
            f"since a time-based holdout of {test_months} months isn't possible yet."
        )
        train_data = data.sample(frac=0.8, random_state=42)
        test_data = data.drop(train_data.index)
    else:
        cutoff_date = unique_dates[-test_months]
        train_data = data[data['date'] < cutoff_date]
        test_data = data[data['date'] >= cutoff_date]

    X_train, y_train = train_data[FEATURES], train_data['cases']
    X_test, y_test = test_data[FEATURES], test_data['cases']

    logger.info(f"Training XGBoost Regressor on {len(X_train)} rows, testing on {len(X_test)} rows...")
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        enable_categorical=True
    )

    model.fit(X_train, y_train)

    # Evaluate
    if len(X_test) > 0:
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        logger.info(f"Model Evaluation (time-based holdout) - RMSE: {rmse:.2f}, MAE: {mae:.2f}")

    # Save model bundled with the province categories it was trained on,
    # so predict_future can reproduce the exact same encoding later.
    model_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {'model': model, 'province_categories': province_categories}
    joblib.dump(bundle, model_path)
    logger.info(f"Model saved to {model_path}")

    return bundle


def predict_future(model_bundle, recent_history):
    """
    Predict next month's cases per province.

    IMPORTANT: `recent_history` must contain at least the last 3 months of
    data per province (not just the single latest month). The previous
    version of this function only received one month of data, which meant
    cases_lag_1/cases_lag_2 could never be computed from real history -
    they always came out as 0 for cases_lag_2 regardless of the province's
    actual trend. Passing enough trailing history lets prepare_features
    compute correct lag values for the latest row before we shift them
    forward by one month.
    """
    model = model_bundle['model']
    province_categories = model_bundle['province_categories']

    prepared = prepare_features(recent_history, province_categories=province_categories)

    # Keep only the latest month per province - by this point it already
    # carries correctly-computed lag_1/lag_2 from real trailing history.
    latest_idx = prepared.groupby('province', observed=True)['date'].idxmax()
    latest_rows = prepared.loc[latest_idx].copy()

    # Roll forward one month
    pred_data = latest_rows.copy()
    pred_data['month'] = pred_data['month'] + 1
    pred_data.loc[pred_data['month'] > 12, 'year'] += 1
    pred_data.loc[pred_data['month'] > 12, 'month'] = 1

    # recompute sin/cos for the forecast month
    pred_data['month_sin'] = np.sin(2 * np.pi * pred_data['month'] / 12)
    pred_data['month_cos'] = np.cos(2 * np.pi * pred_data['month'] / 12)

    # Shift lags forward by one month using the ACTUAL latest values,
    # not zeros: new lag_2 = old lag_1 (real M-1 cases), new lag_1 = actual
    # cases at the latest observed month.
    pred_data['cases_lag_2'] = latest_rows['cases_lag_1']
    pred_data['cases_lag_1'] = latest_rows['cases']

    X_pred = pred_data[FEATURES]
    predictions = model.predict(X_pred)

    pred_data['predicted_cases'] = predictions
    # Ensure no negative predictions
    pred_data['predicted_cases'] = pred_data['predicted_cases'].clip(lower=0)

    return pred_data