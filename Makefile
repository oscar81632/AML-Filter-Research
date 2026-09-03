PYTHON ?= python3
RUN := PYTHONPATH=src $(PYTHON) -m aml_fhe

# Spark targets need PYSPARK_PYTHON set to the same interpreter as the driver.
# If .env exists (written by setup.sh), load it automatically.
ifneq (,$(wildcard .env))
  include .env
  export
endif

PYSPARK_PYTHON   ?= $(PYTHON)
PYSPARK_DRIVER_PYTHON ?= $(PYTHON)
SPARK_ENV := PYSPARK_PYTHON=$(PYSPARK_PYTHON) PYSPARK_DRIVER_PYTHON=$(PYSPARK_DRIVER_PYTHON)
SPARK_RUN := PYTHONPATH=src $(SPARK_ENV) $(PYTHON) -m aml_fhe

.PHONY: help setup install data-download data-validate stage features benchmark sweep full schema

help:
	@echo "First time setup:"
	@echo "  bash setup.sh             Check prerequisites and create .venv"
	@echo "  source .env               Load Python/Spark env vars"
	@echo ""
	@echo "Common targets:"
	@echo "  make setup                Run setup.sh"
	@echo "  make install              Install package in editable mode"
	@echo "  make data-download        Download Kaggle IBM AML data"
	@echo "  make data-validate        Check required HI-Small raw files"
	@echo "  make stage                Build staged transaction parquet (~20-40 min)"
	@echo "  make features             Build ExStraQT feature parquets (~2.5 hr, needs ~62 GB RAM)"
	@echo "  make benchmark            Run benchmark on local feature parquets"
	@echo "  make sweep                Quick sweep on local feature parquets"
	@echo "  make full                 Run data validate -> stage -> features -> benchmark"
	@echo "  make schema PATH=...      Print parquet schema"

setup:
	bash setup.sh

install:
	$(PYTHON) -m pip install -e .

data-download:
	$(RUN) data download

data-validate:
	$(RUN) data validate

stage:
	$(SPARK_RUN) data stage

features:
	$(SPARK_RUN) features build

benchmark:
	$(RUN) benchmark run

sweep:
	$(RUN) sweep run --config configs/sweep-quick.toml --strategy quick --save-report

full:
	$(RUN) run all

schema:
	@test -n "$(PATH)" || (echo "usage: make schema PATH=features/hi-small/train_features.parquet" && exit 2)
	$(RUN) features schema "$(PATH)"
