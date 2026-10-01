"""
Unit tests for data loading, validation, and splitting module (src/data.py).
"""

import pytest
import numpy as np
from src.data import load_raw_data, validate_data, get_train_test_data


def test_load_raw_data_structure():
    """Verify raw data feature count, sample count, and class balance."""
    X, y, feature_names, target_names = load_raw_data()

    assert X.shape == (178, 13), f"Expected shape (178, 13), got {X.shape}"
    assert len(y) == 178, f"Expected 178 target labels, got {len(y)}"
    assert len(feature_names) == 13, f"Expected 13 feature names, got {len(feature_names)}"
    assert len(target_names) == 3, f"Expected 3 target class names, got {len(target_names)}"


def test_validate_data_success():
    """Verify validate_data passes on valid wine data without exceptions."""
    X, y, _, _ = load_raw_data()
    try:
        validate_data(X, y)
    except Exception as exc:
        pytest.fail(f"validate_data raised an unexpected exception: {exc}")


def test_validate_data_null_detection():
    """Verify validate_data raises ValueError when missing values are injected."""
    X, y, _, _ = load_raw_data()
    X_corrupted = X.copy()
    X_corrupted.iloc[0, 0] = np.nan

    with pytest.raises(ValueError, match="null/missing values"):
        validate_data(X_corrupted, y)


def test_validate_data_feature_count_detection():
    """Verify validate_data raises ValueError when feature count is not 13."""
    X, y, _, _ = load_raw_data()
    X_corrupted = X.iloc[:, :10]  # only 10 features

    with pytest.raises(ValueError, match="Expected 13 features"):
        validate_data(X_corrupted, y)


def test_train_test_split_ratios_and_stratification():
    """Verify 80/20 train-test split sizes and class stratification ratios."""
    X_train, X_test, y_train, y_test, _, _ = get_train_test_data(
        test_size=0.2, random_state=42
    )

    total_samples = len(X_train) + len(X_test)
    assert total_samples == 178
    assert len(X_train) == 142  # ~80% of 178
    assert len(X_test) == 36    # ~20% of 178

    # Verify stratification (class proportions should be nearly identical)
    train_dist = y_train.value_counts(normalize=True).sort_index().to_numpy()
    test_dist = y_test.value_counts(normalize=True).sort_index().to_numpy()

    np.testing.assert_allclose(
        train_dist, test_dist, atol=0.05,
        err_msg="Train and test splits should have balanced class distribution."
    )
