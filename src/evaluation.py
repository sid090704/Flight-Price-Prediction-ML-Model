"""
evaluation.py
Evaluation helpers for regression models.
"""

from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd

def evaluate_regression(y_true, y_pred):
    """
    Compute RMSE, MAE, and R^2 for regression predictions.
    Returns a dict with keys: rmse, mae, r2
    """
    rmse = root_mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}

def actual_vs_pred_df(y_true, y_pred, X=None, n=100):
    """
    Return a DataFrame with actual, predicted and absolute error; optionally include features from X.
    """
    df = pd.DataFrame({"actual": list(y_true), "predicted": list(y_pred)})
    df["abs_error"] = (df["actual"] - df["predicted"]).abs()
    if X is not None:
        X_reset = X.reset_index(drop=True)
        df = pd.concat([df, X_reset.iloc[:, :n].reset_index(drop=True)], axis=1)
    return df
