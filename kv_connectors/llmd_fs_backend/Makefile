PYTHON ?= python3
WHEEL_DIR = wheels
DIST_DIR = dist

.PHONY: wheel build test clean check-gds

# Install build dependencies
deps:
	$(PYTHON) -m pip install --upgrade pip setuptools wheel build ninja numpy

# Check if GDS (cuFile) is available and print whether the wheel will include GDS support
check-gds:
	@echo "Checking for GPUDirect Storage (cuFile) support..."
	@if ldconfig -p 2>/dev/null | grep -q libcufile.so; then \
		echo "✓ cuFile library found - wheel will be built WITH GDS support"; \
	else \
		echo "✗ cuFile library not found - wheel will be built WITHOUT GDS support (CPU buffer staging only)"; \
	fi

# Build wheel into dist/ and copy .whl into wheels/ (GDS enabled automatically if cuFile is available)
wheel: deps check-gds
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
