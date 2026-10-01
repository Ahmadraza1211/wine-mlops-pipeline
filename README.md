# End-to-End MLOps Pipeline: Wine Cultivar Classification

[![CI/CD Pipeline](https://github.com/Ahmadraza1211/Shaadi_FYP/actions/workflows/ci.yml/badge.svg)](https://github.com/Ahmadraza1211/Shaadi_FYP/actions/workflows/ci.yml)
[![MLflow Tracking](https://img.shields.io/badge/MLflow-Tracking%20%26%20Registry-blue)](https://mlflow.org/)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-green.svg)](https://www.python.org/)
[![Code Style: Flake8](https://img.shields.io/badge/Code%20Style-Flake8-black)](https://flake8.pycqa.org/)

A production-grade, reproducible MLOps continuous integration pipeline for a multi-class chemical cultivar classification system using `sklearn.datasets.load_wine` (178 samples, 13 features, 3 target classes).

---

## 📁 Repository Structure

```
Assignment 1/
├── .github/
│   └── workflows/
│       └── ci.yml               # Automated GitHub Actions workflow (Lint + Test + Train)
├── data/
│   └── .gitkeep                 # Data directory placeholder
├── src/
│   ├── __init__.py
│   ├── data.py                  # Stratified 80/20 split & data validation checks
│   ├── train.py                 # Dual classifier training, 5-fold CV, MLflow tracking & champion promotion
│   └── evaluate.py              # Champion model inference verification from MLflow registry
├── tests/
│   ├── __init__.py
│   ├── test_data.py             # Data pipeline unit tests
│   └── test_model_gate.py       # Automated MLOps Quality Gate tests (F1, latency, schema)
├── .gitignore                   # Ignores mlruns, pycache, virtualenvs
├── Makefile                     # GNU Makefile for local-to-CI command parity
├── requirements.txt             # Exact pinned dependency versions
├── GIT_CONFLICT_GUIDE.md        # Comprehensive Git collaboration & merge conflict tutorial
└── README.md                    # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup
Clone the repository and install dependencies using the provided Makefile:
```bash
make install
```
*Or manually:*
```bash
pip install -r requirements.txt
```

### 2. Linting Code
Enforce PEP8 compliance with flake8 (max line length 100):
```bash
make lint
```

### 3. Training & MLflow Tracking
Execute the dual-classifier training routine with 5-fold cross-validation and MLflow tracking:
```bash
make train
```

### 4. Running Unit Tests & Automated Quality Gate
Execute pytest test suite verifying data validation and MLOps Quality Gates:
```bash
make test
```

### 5. Clean Workspace
Clean all bytecode and test caches:
```bash
make clean
```

---

## 📊 Milestone Breakdown & Technical Architecture

### Milestone 1: Local Automation & Environment Management
- **Pinned Dependencies**: `scikit-learn`, `mlflow`, `pandas`, `numpy`, `pytest`, `flake8`.
- **Standard GNU Makefile**: Uniform interface across developer machines and GitHub Actions CI.
- **Git Security**: `.gitignore` strictly blocks `mlruns/`, `mlflow.db`, `__pycache__/`, `.venv/` from polluting Git history.

---

### Milestone 2: Modular Pipeline & Dual Classifier Training
- **Data Ingestion & Validation (`src/data.py`)**:
  - Loads 178 chemical samples with 13 features and 3 cultivar classes.
  - Validates feature dimension (`== 13`), missing values (`== 0`), and class label integrity.
  - Stratified 80/20 train-test split with fixed seed `random_state=42`.
- **Model Families (`src/train.py`)**:
  - **Family A: RandomForestClassifier** (3 configurations: shallow, default, deep).
  - **Family B: GradientBoostingClassifier** (3 configurations: slow, standard, fast).
  - Evaluated with 5-fold stratified cross-validation on training data.
  - Logs **Macro F1-score**, **Accuracy**, and **Log Loss**.

---

### Milestone 3: MLflow Tracking & Model Registry
- **Experiment Tracking**: All 6+ runs recorded under experiment `Wine-Cultivar-Classification`.
- **Artifacts & Metadata**:
  - Hyperparameters (`n_estimators`, `max_depth`, `learning_rate`, etc.).
  - Validation metrics across all 5 folds.
  - Model input example and schema signature (`mlflow.models.infer_signature`).
  - Serialized model artifact via `mlflow.sklearn.log_model()`.
- **Model Registry & Promotion**:
  - Compares all logged runs and selects winning model by validation Macro F1 score.
  - Programmatically registers model under `WineClassifier` and assigns alias `champion`.
- **Inference Verification (`src/evaluate.py`)**:
  - Loads champion model from registry (`models:/WineClassifier@champion`).
  - Runs evaluation against held-out test split (20% unseen data).

---

### Milestone 4: CI/CD Pipeline & Automated MLOps Quality Gate
- **GitHub Actions Workflow (`.github/workflows/ci.yml`)**:
  - Triggers automatically on `pull_request` to `main` and `push` to `main`.
  - Executes in clean Ubuntu container on Python 3.10.
- **MLOps Quality Gate (`tests/test_model_gate.py`)**:
  1. **Metric Threshold Gate**: Asserts Validation Macro F1 >= **0.88**.
  2. **Inference Latency Gate**: Asserts batch inference time <= **30 ms** (0.030s).
  3. **Output Schema Integrity Gate**: Asserts class predictions strictly belong to `{0, 1, 2}`.

---

### Milestone 5: Git Collaboration & Merge Conflict Resolution
- Full feature branching workflow (`feature/mlflow-tracking` -> PR -> `main`).
- Engineered merge conflict reproduction steps, conflict markers walkthrough (`<<<<<<< HEAD`), manual resolution, and tree visualization.
- Full details documented in [GIT_CONFLICT_GUIDE.md](file:///c:/semester_4/Data_Science/my%20notes/shaadi-sahulat/MlFlow/Assignment%201/GIT_CONFLICT_GUIDE.md).

---

## 📈 MLflow UI Visualization

To start the local MLflow dashboard and explore runs, parameters, metrics plots, and the Model Registry:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Then open your browser at `http://127.0.0.1:5000`.
