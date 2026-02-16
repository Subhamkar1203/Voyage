"""
Flight Class Classification Model
Predicts flightType (economic / premium / firstClass) from flight features.
Replaces gender classification — uses only flights.csv.
"""
import numpy as np
import joblib
import os
import json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')


MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'outputs', 'models')


def train_all_classifiers(X, y, feature_cols, class_names):
    """Train multiple classifiers for flight class prediction."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'classifier_scaler.pkl'))

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=7),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=150, max_depth=5, random_state=42, verbosity=0, eval_metric='mlogloss'),
        'LightGBM': LGBMClassifier(n_estimators=150, max_depth=5, random_state=42, verbose=-1),
    }

    results = {}
    best_f1 = -1
    best_name = None
    best_model = None

    linear_models = {'Logistic Regression', 'KNN'}

    print("=" * 80)
    print("🎫 FLIGHT CLASS CLASSIFICATION — MODEL TRAINING")
    print("=" * 80)
    print(f"   Classes: {class_names}")

    for name, model in models.items():
        use_scaled = name in linear_models

        if use_scaled:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            cv = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1_weighted')
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            cv = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        results[name] = {
            'model': model,
            'accuracy': round(acc, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1': round(f1, 4),
            'cv_f1_mean': round(cv.mean(), 4),
            'cv_f1_std': round(cv.std(), 4),
        }

        print(f"\n  📌 {name}")
        print(f"     Acc={acc:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | F1={f1:.4f}")
        print(f"     CV F1={cv.mean():.4f} ± {cv.std():.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_model = model

    print(f"\n{'=' * 80}")
    print(f"🏆 BEST: {best_name} — F1={best_f1:.4f}")
    print(f"{'=' * 80}")

    # Detailed report
    if best_name in linear_models:
        y_pred_best = best_model.predict(X_test_scaled)
    else:
        y_pred_best = best_model.predict(X_test)
    print(f"\n📋 Classification Report ({best_name}):")
    print(classification_report(y_test, y_pred_best, target_names=class_names))

    # Save
    joblib.dump(best_model, os.path.join(MODEL_DIR, 'flight_class_model.pkl'))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, 'classifier_feature_cols.pkl'))
    joblib.dump(class_names, os.path.join(MODEL_DIR, 'class_names.pkl'))
    print("[OK] Saved: flight_class_model.pkl")

    # Save results (without model objects)
    results_clean = {k: {kk: vv for kk, vv in v.items() if kk != 'model'} for k, v in results.items()}
    with open(os.path.join(MODEL_DIR, 'classification_results.json'), 'w') as f:
        json.dump(results_clean, f, indent=2)

    return results, best_name, best_model


def predict_flight_class(features_dict):
    """Predict flight class from feature dict."""
    model = joblib.load(os.path.join(MODEL_DIR, 'flight_class_model.pkl'))
    encoders = joblib.load(os.path.join(MODEL_DIR, 'label_encoders.pkl'))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, 'classifier_feature_cols.pkl'))
    class_names = joblib.load(os.path.join(MODEL_DIR, 'class_names.pkl'))

    from_enc = encoders['from'].transform([features_dict['from']])[0]
    to_enc = encoders['to'].transform([features_dict['to']])[0]
    agency_enc = encoders['agency'].transform([features_dict['agency']])[0]

    row = {
        'from_encoded': from_enc, 'to_encoded': to_enc, 'agency_encoded': agency_enc,
        'distance': features_dict['distance'], 'time': features_dict['time'],
        'price': features_dict['price'],
        'month': features_dict['month'], 'day_of_week': features_dict['day_of_week'],
        'quarter': features_dict['quarter'], 'is_weekend': features_dict.get('is_weekend', 0),
    }

    X = np.array([[row[c] for c in feature_cols]])
    pred_idx = model.predict(X)[0]
    pred_class = class_names[pred_idx]

    confidence = None
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)[0]
        confidence = {class_names[i]: round(float(p), 4) for i, p in enumerate(proba)}

    return pred_class, confidence
