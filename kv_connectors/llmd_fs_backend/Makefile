PYTHON ?= python3
WHEEL_DIR = wheels
DIST_DIR = dist

.PHONY: wheel build test clean

# Install build dependencies
deps:
	$(PYTHON) -m pip install --upgrade pip setuptools wheel build ninja numpy

# Build wheel into dist/ and copy .whl into wheels/
wheel: deps
	$(PYTHON) -m pip wheel . -w $(DIST_DIR)
	cp $(DIST_DIR)/llmd_fs_connector-*.whl $(WHEEL_DIR)/
	@echo ""
	@echo "Wheel created in $(WHEEL_DIR):"
	@ls -lh $(WHEEL_DIR)

# Editable install for development
build:
	$(PYTHON) -m pip install -e .

# Run tests
test:
	pytest ./tests/test_fs_backend.py -v -sq

# Clean local build artifacts
clean:
	rm -rf build dist *.egg-info $(WHEEL_DIR)
	@echo "Cleaned build artifacts."
