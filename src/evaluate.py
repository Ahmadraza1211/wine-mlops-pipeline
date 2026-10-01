"""
Module: src/evaluate.py
Description: Loads the champion model from MLflow and evaluates it on unseen test data.
"""

import os
from sklearn.metrics import accuracy_score, f1_score, log_loss, classification_report
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

try:
    from src.data import get_train_test_data
except ImportError:
    from data import get_train_test_data

EXPERIMENT_NAME = "Wine-Cultivar-Classification"
REGISTERED_MODEL_NAME = "WineClassifier"


def load_champion_model():
    """
    Loads the registered champion model from MLflow Model Registry.
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)

    try:
        model_uri = f"models:/{REGISTERED_MODEL_NAME}@champion"
        return mlflow.sklearn.load_model(model_uri)
    except Exception:
        # Fallback: load best run directly if registry alias is not resolved locally
        client = MlflowClient()
        exp = client.get_experiment_by_name(EXPERIMENT_NAME)
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["metrics.val_macro_f1 DESC"],
            max_results=1
        )
        best_run_id = runs[0].info.run_id
        return mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")


def evaluate_test_set():
    """
    Runs model inference on held-out test split and prints final metrics.
    """
    _, X_test, _, y_test, _, target_names = get_train_test_data()
    model = load_champion_model()

    # Model predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    loss = log_loss(y_test, y_proba, labels=[0, 1, 2])

    print("\n" + "=" * 45)
    print("CHAMPION MODEL TEST PERFORMANCE")
    print("=" * 45)
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro F1 : {f1:.4f}")
    print(f"Log Loss : {loss:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    return {"accuracy": acc, "macro_f1": f1, "log_loss": loss}


if __name__ == "__main__":
    evaluate_test_set()
