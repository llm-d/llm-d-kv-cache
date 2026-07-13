IMAGE_REPOSITORY ?= quay.io/pvc-evictor/pvc-evictor
IMAGE_TAG ?= latest

PYTHON_EXE := $(shell command -v python3.12 || command -v python3)
VENV_DIR := .venv
VENV_BIN := $(VENV_DIR)/bin

.PHONY: venv test docker-build

venv: ## Create the dev venv and install the project with dev dependencies
	@if [ ! -f "$(VENV_BIN)/python" ]; then \
		echo "Creating virtual environment in $(VENV_DIR)..."; \
		$(PYTHON_EXE) -m venv $(VENV_DIR); \
		$(VENV_BIN)/pip install --upgrade pip > /dev/null; \
	fi
	@$(VENV_BIN)/pip install -q -e ".[dev]"

test: venv
	$(VENV_BIN)/python -m pytest tests/ -v

docker-build:
	cd .. && podman build -f pvc_evictor/Dockerfile -t $(IMAGE_REPOSITORY):$(IMAGE_TAG) .
