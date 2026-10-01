.PHONY: all lint test train clean

all: lint train test

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

lint:
	flake8 src/ tests/ --max-line-length=100

test:
	pytest tests/ -v

train:
	python src/train.py

clean:
	rm -rf __pycache__ src/__pycache__ tests/__pycache__ .pytest_cache
	rm -rf *.pyc src/*.pyc tests/*.pyc
