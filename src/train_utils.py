"""
train_utils.py
Helpers for training models: splitting, cross-validation utilities, hyperparameter tuning and model saving.
"""

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from typing import Tuple, Dict, Any
import numpy as np
import joblib

def split_data(X, y, test_size=0.2, random_state=42, shuffle=True) -> Tuple:
    """
    Split X and y into train/test sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=shuffle)

def cv_rmse(estimator: BaseEstimator, X, y, cv=3, n_jobs=-1) -> np.ndarray:
    """
    Compute cross-validated RMSE (returns array of RMSE per fold).
    """
    scores = cross_val_score(estimator, X, y, cv=cv, scoring="neg_mean_squared_error", n_jobs=n_jobs)
    return np.sqrt(-scores)

def randomized_search(estimator: BaseEstimator, param_distributions: Dict[str, Any], X, y,
                      n_iter=20, cv=3, scoring="neg_mean_squared_error", random_state=42, n_jobs=-1, verbose=1):
    """
    Run RandomizedSearchCV and return fitted RandomizedSearchCV object.
    """
    rs = RandomizedSearchCV(estimator=estimator, param_distributions=param_distributions,
                            n_iter=n_iter, cv=cv, scoring=scoring, random_state=random_state,
                            n_jobs=n_jobs, verbose=verbose)
    rs.fit(X, y)
    return rs

def save_model(model: BaseEstimator, path: str):
    """
    Save a fitted model to disk using joblib.
    """
    joblib.dump(model, path, compress=3)
