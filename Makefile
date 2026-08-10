# Language Identification - convenience targets
# Usage: make train | make test | make api | make ui | make docker

PYTHON ?= python3

.PHONY: help setup data eda train quick train-test test api ui docker compose clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Install dependencies
	pip install -r requirements.txt

data: ## Download / build the dataset (cached afterwards)
	$(PYTHON) -m src.data_loader

eda: ## Run exploratory data analysis (charts + summary)
	$(PYTHON) -m src.eda

train: ## Run the full training pipeline
	$(PYTHON) -m src.train

quick: ## Fast training run (CI mode)
	$(PYTHON) -m src.train --quick

train-test: ## Quick training run that also runs the test suite (CI entry)
	$(PYTHON) -m src.train --quick
	$(PYTHON) -m pytest -q

test: ## Run the unit test suite
	$(PYTHON) -m pytest -q

api: ## Start the FastAPI backend on :8000
	uvicorn api:app --host 0.0.0.0 --port 8000 --reload

ui: ## Start the Streamlit app on :8501
	streamlit run app.py

docker: ## Build the Docker image
	docker build -t language-identification .

compose: ## Run API + UI via docker compose
	docker compose up --build

clean: ## Remove generated artifacts (model, cache, logs)
	@echo "Remove generated files manually when needed: models/*.joblib, dataset/*.csv, reports/, logs/, __pycache__/."
	@echo "These are regenerated automatically by 'make train' / 'make eda'."
