#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = energy_prices
PYTHON_VERSION = 3.13.5
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	pip install -e .




## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



# ## Run tests (local only, not published)
# .PHONY: test
# test:
# 	python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:

	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y

	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"




#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

# Hourly data (default)
#################################################################################

## Run full pipeline: download + combine + merge (hourly)
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) -m src.data.smard main --dir-name smard_hourly --resolution hour
	$(PYTHON_INTERPRETER) -m src.data.smard main --dir-name smard_hourly_de_at_lu --resolution hour --region DE-AT-LU
	$(PYTHON_INTERPRETER) -m src.data.energy_charts
	$(PYTHON_INTERPRETER) -m src.data.processing combine --resolution hour
	$(PYTHON_INTERPRETER) -m src.data.processing merge --resolution hour
	$(PYTHON_INTERPRETER) -m src.data.commodities main
	$(PYTHON_INTERPRETER) -m src.data.commodities process --smard-path data/processed/merged_dataset_hourly.parquet

## Update existing data incrementally (hourly)
.PHONY: update
update: requirements
	$(PYTHON_INTERPRETER) -m src.data.smard update --dir-name smard_hourly --resolution hour
	$(PYTHON_INTERPRETER) -m src.data.energy_charts
	$(PYTHON_INTERPRETER) -m src.data.processing combine --resolution hour --incremental
	$(PYTHON_INTERPRETER) -m src.data.commodities update
	$(PYTHON_INTERPRETER) -m src.data.commodities process \
		--smard-path data/processed/merged_dataset_hourly.parquet
	$(PYTHON_INTERPRETER) -m src.data.processing merge --resolution hour --incremental

# Quarter-hourly data pipeline (currently unused; uncomment if 15-min resolution needed)
# .PHONY: data-qh
# data-qh: requirements
# 	$(PYTHON_INTERPRETER) -m src.data.dataset main --dir-name smard_api --resolution quarterhour
# 	$(PYTHON_INTERPRETER) -m src.data.dataset main --dir-name DE_AT_LU --resolution quarterhour --region DE-AT-LU
# 	$(PYTHON_INTERPRETER) -m src.data.processing combine --resolution quarterhour
# 	$(PYTHON_INTERPRETER) -m src.data.processing merge --resolution quarterhour
# 	$(PYTHON_INTERPRETER) -m src.data.commodities main
# 	$(PYTHON_INTERPRETER) -m src.data.commodities process --smard-path data/processed/merged_dataset.parquet

# .PHONY: update-data-qh
# update-data-qh: requirements
# 	$(PYTHON_INTERPRETER) -m src.data.dataset update --dir-name smard_api --resolution quarterhour
# 	$(PYTHON_INTERPRETER) -m src.data.processing combine --resolution quarterhour --incremental
# 	$(PYTHON_INTERPRETER) -m src.data.processing merge --resolution quarterhour --incremental
# 	$(PYTHON_INTERPRETER) -m src.data.commodities update
# 	$(PYTHON_INTERPRETER) -m src.data.commodities process --smard-path data/processed/merged_dataset.parquet

# Commodity price data
#################################################################################

## Download all commodity price data (carbon, TTF, Brent)
# .PHONY: commodity-data
# commodity-data: requirements
# 	$(PYTHON_INTERPRETER) -m src.data.commodities main

# ## Update existing commodity data with recent prices
# .PHONY: update-commodity-data
# update-commodity-data: requirements
# 	$(PYTHON_INTERPRETER) -m src.data.commodities update --redundancy-days 14

# ## Process commodity data: combine daily + forward-fill to hourly
# .PHONY: process-commodity-data
# process-commodity-data: requirements
# 	$(PYTHON_INTERPRETER) -m src.data.commodities process

## Add recent data for inference without polluting training dataset
# Commented out as no longer used
# .PHONY: add-data
# add-data:
# 	$(PYTHON_INTERPRETER) -m src.cli data add

# Model training
#################################################################################

## Train baseline models (naive, ARIMA, ETS, Prophet)
.PHONY: baselines
baselines: requirements
	$(PYTHON_INTERPRETER) -m src.modeling.baselines

## Retrain production blend from committed hyperparameters
.PHONY: train-final
train-final:
	$(PYTHON_INTERPRETER) -m src.modeling.train_final

## full-project: Run entire pipeline from scratch using committed hyperparameters.
##   Requires: conda env activated, make requirements run.
##   Note: Uses blend_hyperparams.json for fresh model selection run 'make blend' instead.
.PHONY: full-project
full-project: data train-baselines train-final forecast
	@echo "Full project pipeline complete."

## blend: Initial setup select MLflow candidates, CV-validate, train, compute weights
.PHONY: blend
blend:
	$(PYTHON_INTERPRETER) -m src.cli blend select

## blend-update: Daily incremental tree warm-start + weight refresh (fast)
.PHONY: blend-update
blend-update:
	$(PYTHON_INTERPRETER) -m src.cli blend update

## blend-retrain: Manual biweekly full retrain (same as 'retrain' but without deployment wrapper)
.PHONY: blend-retrain
blend-retrain:
	$(PYTHON_INTERPRETER) -m src.cli blend retrain

## Print blend ensemble info
.PHONY: blend-info
blend-info:
	$(PYTHON_INTERPRETER) -m src.cli blend info

## Start MLflow UI for browsing experiments
.PHONY: mlflow
mlflow:
	mlflow ui --backend-store-uri sqlite:///$(CURDIR)/models/mlflow.db --host 127.0.0.1 --port 5000

# Deployment
#################################################################################

## Generate 24h forecast from blend ensemble
.PHONY: forecast
forecast:
	$(PYTHON_INTERPRETER) -m src.deploy.inference --skip-update

## Generate forecast with data update
.PHONY: forecast-update
forecast-update:
	$(PYTHON_INTERPRETER) -m src.deploy.inference

## retrain: Deployment wrapper, used by GitHub Actions biweekly workflow
.PHONY: retrain
retrain:
	$(PYTHON_INTERPRETER) -m src.deploy.retrain


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
