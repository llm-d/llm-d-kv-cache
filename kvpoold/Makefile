PYTHON ?= python3

.PHONY: help test lint fmt

help: ## List the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-6s %s\n", $$1, $$2}'

test: ## Run the unit tests
	$(PYTHON) -m pytest tests/ -q

lint: ## Check lint and formatting
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

fmt: ## Fix lint and format the code
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .
