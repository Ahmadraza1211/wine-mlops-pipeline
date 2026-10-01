"""
Module: src/data.py
Description: Loads, validates, and splits the Wine dataset for training and evaluation.
"""

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


def load_raw_data():
    """ first loads the Wine dataset as pandas DataFrame and Series """
    wine = load_wine(as_frame=True)
    X = wine.data
    y = wine.target
    feature_names = list(wine.feature_names)
    target_names = list(wine.target_names)
    return X, y, feature_names, target_names


def validate_data(X, y):
    """
    Validates data requirements:
    1. Feature count must be 13.
    2. No missing (null) values in features or targets.
    3. Target must only have classes 0, 1, 2.
    """
    # Check 1: Feature count
    if X.shape[1] != 13:
        raise ValueError(f"Expected 13 features, but found {X.shape[1]}")

    # Check 2: Missing values
    if X.isnull().values.any():
        raise ValueError("Features matrix contains null/missing values.")
    if y.isnull().values.any():
        raise ValueError("Target contains null/missing values.")

    # Check 3: Class labels
    unique_classes = set(y.unique())
    if not unique_classes.issubset({0, 1, 2}):
        raise ValueError(f"Unexpected classes found: {unique_classes}")


def get_train_test_data(test_size=0.2, random_state=42):
    """
    Loads, validates, and creates a stratified 80/20 train-test split.
    """
    X, y, feature_names, target_names = load_raw_data()
    validate_data(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test, feature_names, target_names


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, features, targets = get_train_test_data()
    print("Data loaded and validated successfully!")
    print(f"Train samples: {len(X_tr)}, Test samples: {len(X_te)}")
    print(f"Features ({len(features)}): {features}")
