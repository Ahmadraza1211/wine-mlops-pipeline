# Complete Guide: Understanding `Makefile` in MLOps

This document explains **line-by-line** what a `Makefile` is, why we use it in Machine Learning & MLOps, and the exact purpose of every single command written inside our project's `Makefile`.

---

## 1. What is a Makefile & Why Do We Use It?

In MLOps and software engineering, you have many recurring terminal commands to run:
- Installing packages
- Checking code quality / formatting (linting)
- Running unit tests
- Training models
- Cleaning cache files

Instead of typing long, error-prone commands like `flake8 src/ tests/ --max-line-length=100` or remembering complex Python cleanup scripts every time, a **`Makefile`** acts as a **central command runner / automation script**.

### Key Benefits:
1. **Developer-to-CI Parity**: The exact same command you run on your laptop (`make test`) is executed by GitHub Actions CI in the cloud.
2. **Single Entry Point**: Anyone joining the project just types `make install` or `make train` without asking how to run the project.
3. **Reproducibility**: Eliminates manual mistakes.

---

## 2. Basic Makefile Syntax Rules

A Makefile is composed of **rules**:

```makefile
target: prerequisites
	command_to_run
```

- **Target**: The name of the action or file you want to execute (e.g., `install`, `train`).
- **Prerequisites / Dependencies** *(optional)*: Other targets that must run *before* this target runs.
- **Command**: The shell command to run. **IMPORTANT**: Every command line MUST be indented with a **`Tab` character**, not spaces!

---

## 3. Line-by-Line Detailed Explanation of Our Makefile

Here is the exact code from our project's `Makefile` broken down line-by-line:

---

### Line 1: `.PHONY` Declaration
```makefile
.PHONY: all install lint test train clean
```

#### What it means:
- By default, `make` checks if a file with the target name exists in the folder. If you had a folder or file named `clean` or `test`, `make` might say *"clean is up to date"* and refuse to run the command!
- **`.PHONY`** tells Make: *"These targets (`all`, `install`, `lint`, `test`, `train`, `clean`) are **actions/commands**, NOT filenames. Always execute them whenever called, regardless of files in the folder."*

---

### Lines 3–4: The Default `all` Target
```makefile
all: install lint test train
```

#### What it means:
- If you simply type `make` in your terminal without specifying a target, `make` runs the first target it finds (which is `all`).
- `all` specifies **prerequisites**: it will run `install`, then `lint`, then `test`, and finally `train` sequentially.
- It provides a complete 1-step pipeline execution.

---

### Lines 5–7: The `install` Target
```makefile
install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
```

#### What it means:
1. `python -m pip install --upgrade pip`: Ensures the Python package installer (`pip`) is updated to the latest stable version so all modern package wheels install smoothly.
2. `pip install -r requirements.txt`: Reads the `requirements.txt` file and installs the exact pinned dependencies (`scikit-learn`, `mlflow`, `pandas`, `pytest`, `flake8`, etc.).

#### How to run it:
```bash
make install
```

---

### Lines 9–10: The `lint` Target
```makefile
lint:
	flake8 src/ tests/ --max-line-length=100
```

#### What it means:
- **Linting** is static code analysis that checks code quality, styling rules (PEP 8), unused imports, and syntax errors without running the code.
- `flake8`: The linting tool.
- `src/ tests/`: Tells flake8 to inspect all Python files inside the `src/` and `tests/` directories.
- `--max-line-length=100`: Extends standard PEP 8 79-character limit to **100 characters** per line (as required by the Assignment 1 specification).

#### How to run it:
```bash
make lint
```

---

### Lines 12–13: The `test` Target
```makefile
test:
	pytest tests/ -v
```

#### What it means:
- **Pytest** is the testing framework used in Python.
- `tests/`: Tells pytest to discover and run all test files inside the `tests/` folder (such as `test_data.py` and `test_model_gate.py`).
- `-v`: Flag for **verbose output**, printing each individual test function name along with `PASSED` or `FAILED`.

#### How to run it:
```bash
make test
```

---

### Lines 15–16: The `train` Target
```makefile
train:
	python src/train.py
```

#### What it means:
- Executes our standalone Python training script.
- This script loads the Wine dataset, runs 5-fold cross-validation on `RandomForest` and `GradientBoosting` classifiers, tracks metrics/parameters/signatures in **MLflow**, and promotes the winning model to the MLflow Model Registry as `champion`.

#### How to run it:
```bash
make train
```

---

### Lines 18–20: The `clean` Target
```makefile
clean:
	rm -rf __pycache__ src/__pycache__ tests/__pycache__ .pytest_cache
	rm -rf *.pyc src/*.pyc tests/*.pyc
```

#### What it means (Simple Explanation):
- When Python scripts and test runners execute, they automatically generate temporary cache folders and compiled bytecode files to speed up subsequent executions. Over time, these files clutter the repository.
- The `clean` target removes all these temporary files in one single command:
  1. `rm -rf __pycache__ src/__pycache__ tests/__pycache__`: Removes Python's cached bytecode directories.
  2. `rm -rf .pytest_cache`: Removes Pytest's temporary test history and cache folder.
  3. `rm -rf *.pyc src/*.pyc tests/*.pyc`: Removes any leftover compiled `.pyc` files.
- **Flags explanation**:
  - `rm`: The "remove" command.
  - `-r`: **Recursive** (deletes folders and everything inside them).
  - `-f`: **Force** (does not throw an error if the folder or file doesn't exist).

#### How to run it:
```bash
make clean
```

---

## 4. Summary Table of Makefile Commands

| Command | What It Does Behind the Scenes | When to Use It |
| :--- | :--- | :--- |
| `make install` | Upgrades `pip` and installs all dependencies from `requirements.txt` | When first setting up the project |
| `make lint` | Runs `flake8` across `src/` and `tests/` (max line length 100) | Before committing code to Git |
| `make test` | Runs `pytest` on all tests in `tests/` with verbose output | Before pushing changes or merging PRs |
| `make train` | Runs `python src/train.py` (5-fold CV + MLflow tracking + Model Registry) | When training/tuning models |
| `make clean` | Removes `__pycache__`, `.pytest_cache`, and `*.pyc` files | To clean up temporary clutter |
| `make all` *(or `make`)* | Runs `install` $\rightarrow$ `lint` $\rightarrow$ `test` $\rightarrow$ `train` sequentially | For a full automated workflow run |
