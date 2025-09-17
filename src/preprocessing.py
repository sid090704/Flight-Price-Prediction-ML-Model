"""
preprocessing.py
Contains reusable preprocessing pipeline builder for the flight price project.
"""

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from typing import List, Tuple
import numpy as np

def build_preprocessor(numeric_cols: List[str], cat_cols: List[str]):
    """
    Build a ColumnTransformer that applies:
      - median imputation + StandardScaler to numeric columns
      - most_frequent imputation + OneHotEncoder to categorical columns (handle_unknown='ignore')
    Returns:
        preprocessor: ColumnTransformer
    """
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # OneHotEncoder sparse_output introduced in newer sklearn; prefer a compatibility approach
    try:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", onehot)
    ])

    transformers = [("num", num_pipe, numeric_cols)]
    if len(cat_cols) > 0:
        transformers.append(("cat", cat_pipe, cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor
