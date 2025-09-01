.PHONY: help install install-dev clean format lint test run-query run-pinecone run-pre-embedded

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	pip install -r requirements.txt

install-dev: ## Install development dependencies
	pip install -r requirements.txt
	pip install -e ".[dev]"

clean: ## Clean up cache and temporary files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete

format: ## Format code with black
	python -m black .

lint: ## Lint code with flake8
	python -m flake8 .

test: ## Run tests
	python -m pytest

run-query: ## Run the query script
	python query.py

run-pinecone: ## Run the Pinecone embedding generation script
	python pinecone_generate_embeddings.py

run-pre-embedded: ## Run the pre-embedded dataset script
	python pinecone_pre_embedded.py

venv: ## Create a new virtual environment
	python -m venv venv

sync: ## Sync dependencies (not applicable with pip)
	@echo "Use 'make install' to install dependencies with pip"

add: ## Add a new dependency (usage: make add PKG=package_name)
	pip install $(PKG)

add-dev: ## Add a new development dependency (usage: make add-dev PKG=package_name)
	pip install $(PKG)
