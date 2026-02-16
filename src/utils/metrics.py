"""Custom metrics utilities."""
import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, precision_score, recall_score
)


def regression_metrics(y_true, y_pred):
    """Compute regression metrics."""
    return {
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)),
        'mape': float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
    }


def classification_metrics(y_true, y_pred, average='weighted'):
    """Compute classification metrics."""
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1_weighted': float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        'precision': float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average=average, zero_division=0))
    }
