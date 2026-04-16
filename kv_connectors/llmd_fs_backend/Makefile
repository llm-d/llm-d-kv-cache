PYTHON ?= python3
WHEEL_DIR = wheels
DIST_DIR = dist
VERSION := $(shell $(PYTHON) -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])")

.PHONY: wheel build test test-performance clean check-gds

# Install build dependencies
deps:
	$(PYTHON) -m pip install --upgrade pip setuptools wheel build ninja numpy

# Check if GDS (cuFile) is available at runtime
check-gds:
	@echo "Checking for GPUDirect Storage (cuFile) runtime availability..."
	@if ldconfig -p 2>/dev/null | grep -q libcufile.so; then \
		echo "✓ cuFile library found - GDS will be available at runtime"; \
	else \
		echo "✗ cuFile library not found - GDS will not be available at runtime (CPU buffer staging only)"; \
	fi

# Build wheel into dist/ and copy .whl into wheels/
wheel: deps
	$(PYTHON) -m pip wheel . -w $(DIST_DIR)
	cp $(DIST_DIR)/llmd_fs_connector-$(VERSION)-*.whl $(WHEEL_DIR)/
	@echo ""
	@echo "Wheel created in $(WHEEL_DIR):"
	@ls -lh $(WHEEL_DIR)

# Editable install for development
build:
	$(PYTHON) -m pip install -e .

# Run unit tests (excludes the performance/ directory)
test:
	pytest ./tests/ -v -sq --ignore=./tests/performance

# Run performance tests (throughput + stress sweep)
test-performance:
	pytest ./tests/performance/ -v -sq

# Clean local build artifacts
clean:
	rm -rf build dist *.egg-info $(WHEEL_DIR)
	@echo "Cleaned build artifacts."
