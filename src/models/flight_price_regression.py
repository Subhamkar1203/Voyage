"""
Flight Price Regression Model
Trains multiple regression models, selects the best, and provides prediction interface.
"""
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')


MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'outputs', 'models')


def train_all_regressors(X, y, feature_cols):
    """Train multiple regression models and return results + best model."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'regression_scaler.pkl'))

    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=1.0),
        'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5),
        'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=6, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=200, max_depth=6, random_state=42, verbosity=0),
        'LightGBM': LGBMRegressor(n_estimators=200, max_depth=6, random_state=42, verbose=-1),
    }

    results = {}
    best_r2 = -np.inf
    best_name = None
    best_model = None

    linear_models = {'Linear Regression', 'Ridge', 'Lasso', 'ElasticNet'}

    print("=" * 80)
    print("✈️  FLIGHT PRICE REGRESSION — MODEL TRAINING")
    print("=" * 80)

    for name, model in models.items():
        use_scaled = name in linear_models

        if use_scaled:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            cv = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            cv = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100

        results[name] = {
            'model': model,
            'MAE': round(mae, 4),
            'MSE': round(mse, 4),
            'RMSE': round(rmse, 4),
            'R2': round(r2, 4),
            'MAPE': round(mape, 4),
            'CV_R2_mean': round(cv.mean(), 4),
            'CV_R2_std': round(cv.std(), 4),
        }

        print(f"\n  📌 {name}")
        print(f"     R²={r2:.4f} | RMSE={rmse:.2f} | MAE={mae:.2f} | MAPE={mape:.1f}%")
        print(f"     CV R²={cv.mean():.4f} ± {cv.std():.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_name = name
            best_model = model

    print(f"\n{'=' * 80}")
    print(f"🏆 BEST: {best_name} — R²={best_r2:.4f}")
    print(f"{'=' * 80}")

    # Save best model
    joblib.dump(best_model, os.path.join(MODEL_DIR, 'flight_price_model.pkl'))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, 'regression_feature_cols.pkl'))
    print(f"[OK] Saved: flight_price_model.pkl")

    return results, best_name, best_model, X_test, y_test


def hyperparameter_tune(X, y):
    """Grid-search XGBoost for best hyperparams."""
    from sklearn.model_selection import GridSearchCV

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
    }

    print("\n[TUNE] Hyperparameter Tuning (XGBoost)...")
    gs = GridSearchCV(
        XGBRegressor(random_state=42, verbosity=0),
        param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=0
    )
    gs.fit(X_train, y_train)

    y_pred = gs.best_estimator_.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"   Best params: {gs.best_params_}")
    print(f"   Test R²={r2:.4f} | RMSE={rmse:.2f}")

    joblib.dump(gs.best_estimator_, os.path.join(MODEL_DIR, 'flight_price_model_tuned.pkl'))
    print("[OK] Saved: flight_price_model_tuned.pkl")

    return gs.best_estimator_, gs.best_params_


def predict_price(features_dict):
    """
    Predict flight price from a dict of features.
    Returns predicted price (float).
    """
    model = joblib.load(os.path.join(MODEL_DIR, 'flight_price_model_tuned.pkl'))
    encoders = joblib.load(os.path.join(MODEL_DIR, 'label_encoders.pkl'))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, 'regression_feature_cols.pkl'))

    from_enc = encoders['from'].transform([features_dict['from']])[0]
    to_enc = encoders['to'].transform([features_dict['to']])[0]
    class_enc = encoders['flightType'].transform([features_dict['flightType']])[0]
    agency_enc = encoders['agency'].transform([features_dict['agency']])[0]

    row = {
        'from_encoded': from_enc, 'to_encoded': to_enc,
        'flightType_encoded': class_enc, 'agency_encoded': agency_enc,
        'distance': features_dict['distance'], 'time': features_dict['time'],
        'month': features_dict['month'], 'day_of_week': features_dict['day_of_week'],
        'quarter': features_dict['quarter'], 'is_weekend': features_dict.get('is_weekend', 0),
    }

    X = np.array([[row[c] for c in feature_cols]])
    return round(float(model.predict(X)[0]), 2)
