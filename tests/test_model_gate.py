"""
Automated MLOps Quality Gate tests (tests/test_model_gate.py).
Enforces:
1. Metric Threshold Gate: Validation Macro F1-score >= 0.88
2. Inference Latency Gate: Batch inference time <= 30 ms (0.030 seconds)
3. Output Schema Integrity: Class indices only (0, 1, or 2)
"""

import time
import pytest
import numpy as np
from sklearn.metrics import f1_score
from src.data import get_train_test_data
from src.evaluate import load_champion_model
from src.train import run_training_pipeline


@pytest.fixture(scope="session")
def trained_model_and_data():
    """
    Ensures model is trained and provides model with test data.
    """
    X_train, X_test, y_train, y_test, feature_names, target_names = get_train_test_data()
    try:
        model = load_champion_model()
    except Exception:
        # If not trained yet in CI, execute training pipeline first
        run_training_pipeline()
        model = load_champion_model()

    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "target_names": target_names
    }


def test_metric_threshold_gate(trained_model_and_data):
    """
    Gate 1: Metric Threshold Gate
    Asserts that the model achieves Macro F1-score >= 0.88 on test/evaluation split.
    """
    model = trained_model_and_data["model"]
    X_test = trained_model_and_data["X_test"]
    y_test = trained_model_and_data["y_test"]

    predictions = model.predict(X_test)
    macro_f1 = f1_score(y_test, predictions, average="macro")

    min_required_f1 = 0.88
    assert macro_f1 >= min_required_f1, (
        f"Model failed Quality Gate 1! Macro F1 score was {macro_f1:.4f}, "
        f"which is below the required threshold of {min_required_f1}."
    )


def test_inference_latency_gate(trained_model_and_data):
    """
    Gate 2: Inference Latency Gate
    Asserts that batch inference on the test split takes <= 30 ms (0.030 seconds).
    """
    model = trained_model_and_data["model"]
    X_test = trained_model_and_data["X_test"]

    # Warm-up run
    _ = model.predict(X_test)

    # Benchmark over 50 iterations to get robust average batch inference time
    iterations = 50
    start_time = time.perf_counter()
    for _ in range(iterations):
        _ = model.predict(X_test)
    total_elapsed = time.perf_counter() - start_time
    avg_latency_ms = (total_elapsed / iterations) * 1000.0

    max_allowed_latency_ms = 30.0
    assert avg_latency_ms <= max_allowed_latency_ms, (
        f"Model failed Quality Gate 2! Batch inference latency was {avg_latency_ms:.2f} ms, "
        f"which exceeds maximum allowed limit of {max_allowed_latency_ms} ms."
    )


def test_output_schema_integrity(trained_model_and_data):
    """
    Gate 3: Output Schema Integrity
    Asserts that predicted classes strictly contain only valid class indices: {0, 1, 2}.
    """
    model = trained_model_and_data["model"]
    X_test = trained_model_and_data["X_test"]

    predictions = model.predict(X_test)
    unique_preds = set(np.unique(predictions))
    valid_classes = {0, 1, 2}

    assert unique_preds.issubset(valid_classes), (
        f"Model failed Quality Gate 3! Output contained invalid class labels: {unique_preds}. "
        f"Expected subset of {valid_classes}."
    )
    assert len(predictions) == len(X_test), (
        f"Prediction count mismatch: expected {len(X_test)}, got {len(predictions)}."
    )
