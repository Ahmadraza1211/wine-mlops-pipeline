"""
Module: src/train.py
Description: Trains RandomForest & GradientBoosting models with 5-fold Cross-Validation,
logs runs and artifacts to MLflow, and registers the best model as 'champion'.
"""

import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

try:
    from src.data import get_train_test_data
except ImportError:
    from data import get_train_test_data

EXPERIMENT_NAME = "Wine-Cultivar-Classification"
REGISTERED_MODEL_NAME = "WineClassifier"
RANDOM_STATE = 99


def evaluate_cross_validation(model_class, params, X_train, y_train, n_splits=5):
    """
    Performs 5-fold Stratified Cross-Validation and returns average train & val metrics.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    val_f1_scores = []
    val_acc_scores = []
    val_losses = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Train model on this fold
        model = model_class(**params)
        model.fit(X_tr, y_tr)

        # Validation predictions
        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)

        val_f1_scores.append(f1_score(y_val, preds, average="macro"))
        val_acc_scores.append(accuracy_score(y_val, preds))
        val_losses.append(log_loss(y_val, probs, labels=[0, 1, 2]))

    return {
        "val_macro_f1": float(np.mean(val_f1_scores)),
        "val_accuracy": float(np.mean(val_acc_scores)),
        "val_log_loss": float(np.mean(val_losses))
    }


def get_model_configurations():
    """
    Returns 6 model configurations (3 RandomForest + 3 GradientBoosting).
    """
    return [
        # Model Family A: RandomForest
        {
            "name": "RF_Config_1_shallow",
            "family": "RandomForestClassifier",
            "class": RandomForestClassifier,
            "params": {"n_estimators": 50, "max_depth": 3, "random_state": RANDOM_STATE}
        },
        {
            "name": "RF_Config_2_default",
            "family": "RandomForestClassifier",
            "class": RandomForestClassifier,
            "params": {"n_estimators": 100, "max_depth": 5, "random_state": RANDOM_STATE}
        },
        {
            "name": "RF_Config_3_deep",
            "family": "RandomForestClassifier",
            "class": RandomForestClassifier,
            "params": {"n_estimators": 200, "max_depth": 8, "random_state": RANDOM_STATE}
        },
        # Model Family B: GradientBoosting
        {
            "name": "GBM_Config_1_slow",
            "family": "GradientBoostingClassifier",
            "class": GradientBoostingClassifier,
            "params": {"n_estimators": 50, "learning_rate": 0.05, "max_depth": 3, "random_state": RANDOM_STATE}  # noqa: E501
        },
        {
            "name": "GBM_Config_2_standard",
            "family": "GradientBoostingClassifier",
            "class": GradientBoostingClassifier,
            "params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "random_state": RANDOM_STATE}  # noqa: E501
        },
        {
            "name": "GBM_Config_3_fast",
            "family": "GradientBoostingClassifier",
            "class": GradientBoostingClassifier,
            "params": {"n_estimators": 150, "learning_rate": 0.2, "max_depth": 4, "random_state": RANDOM_STATE}  # noqa: E501
        },
    ]


def run_training_pipeline():
    """
    Main training routine:
    1. Loads dataset
    2. Trains and tracks all models in MLflow
    3. Promotes best model to MLflow Model Registry as 'champion'
    """
    # Setup MLflow
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test, _, _ = get_train_test_data()
    configs = get_model_configurations()
    all_runs = []

    print(f"Starting MLflow experiment: '{EXPERIMENT_NAME}'")

    for cfg in configs:
        with mlflow.start_run(run_name=cfg["name"]) as run:
            run_id = run.info.run_id

            # 1. Log parameters and tags
            mlflow.log_params(cfg["params"])
            mlflow.set_tag("model_family", cfg["family"])
            mlflow.set_tag("dataset", "load_wine")

            # 2. 5-Fold Cross Validation
            metrics = evaluate_cross_validation(
                cfg["class"], cfg["params"], X_train, y_train, n_splits=5
            )
            mlflow.log_metrics(metrics)

            # 3. Train final model on full training set
            final_model = cfg["class"](**cfg["params"])
            final_model.fit(X_train, y_train)

            # 4. Infer model signature & save input example
            signature = infer_signature(X_train, final_model.predict(X_train))
            input_example = X_train.iloc[:5]

            # 5. Log trained model artifact
            mlflow.sklearn.log_model(
                sk_model=final_model,
                artifact_path="model",
                signature=signature,
                input_example=input_example
            )

            print(
                f"[{cfg['family']}] {cfg['name']} -> "
                f"Val F1: {metrics['val_macro_f1']:.4f}, "
                f"Val Acc: {metrics['val_accuracy']:.4f}, "
                f"Val Loss: {metrics['val_log_loss']:.4f}"
            )

            all_runs.append({
                "run_id": run_id,
                "name": cfg["name"],
                "val_macro_f1": metrics["val_macro_f1"]
            })

    # Select Champion Model (highest validation Macro F1)
    best_run = max(all_runs, key=lambda r: r["val_macro_f1"])
    print("\n" + "=" * 50)
    print(f"CHAMPION MODEL SELECTED: {best_run['name']}")
    print(f"Validation Macro F1: {best_run['val_macro_f1']:.4f}")
    print("=" * 50)

    # Register in MLflow Model Registry
    client = MlflowClient()
    model_uri = f"runs:/{best_run['run_id']}/model"
    try:
        registered = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
        client.set_registered_model_alias(
            name=REGISTERED_MODEL_NAME,
            alias="champion",
            version=registered.version
        )
        print(f"Registered '{REGISTERED_MODEL_NAME}' version {registered.version} as '@champion'.")
    except Exception as exc:
        print(f"Registry note: {exc}")

    return best_run


if __name__ == "__main__":
    run_training_pipeline()
