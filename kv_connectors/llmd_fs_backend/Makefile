PYTHON ?= python
WHEEL_DIR = wheels
DIST_DIR = dist

.PHONY: wheel build test clean

# Build wheel into dist/ and copy .whl into wheels/
wheel:
	$(PYTHON) -m pip install --upgrade pip setuptools wheel
	$(PYTHON) -m pip wheel . -w $(DIST_DIR)
	cp $(DIST_DIR)/llmd_fs_connector-*.whl $(WHEEL_DIR)/
	@echo ""
	@echo "Wheel copied to $(WHEEL_DIR):"
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
