.PHONY: install clean test lint format dev-setup help check-deps sync-deps

# Variables
PYTHON := python3
PIP := pip3
VENV := venv
VENV_BIN := $(VENV)/bin

help:
	@echo "Available commands:"
	@echo "make dev-setup    - Set up development environment"
	@echo "make install      - Install the package and dependencies"
	@echo "make clean        - Clean up build artifacts and cache"
	@echo "make test         - Run tests"
	@echo "make lint         - Run linting checks"
	@echo "make format       - Format code using black"
	@echo "make check-deps   - Check for dependency updates"
	@echo "make sync-deps    - Sync virtual environment with requirements.txt"
	@echo "make uninstall    - Uninstall the package"

dev-setup: clean
	$(PYTHON) -m venv $(VENV)
	$(VENV_BIN)/pip install --upgrade pip
	$(VENV_BIN)/pip install -e ".[dev]"
	$(VENV_BIN)/pip install -r requirements.txt
	$(VENV_BIN)/pip install black flake8 pytest pip-tools safety
	@echo "\033[1;33mDevelopment environment setup complete. Activate it with: source venv/bin/activate\033[0m"

install:
	$(PIP) install -e .
	@echo "Installation complete. To get started:"
	@echo "\033[1;32m1. Run 'stego --guide' for steganography guide\033[0m"
	@echo "\033[1;32m2. Run 'stego-detect --guide' for detection guide\033[0m"

# New target to check dependencies for updates and security issues
check-deps:
	@echo "Checking for dependency updates..."
	$(VENV_BIN)/pip list --outdated
	@echo "\nChecking for security vulnerabilities..."
	$(VENV_BIN)/safety check

# New target to sync dependencies
sync-deps:
	$(VENV_BIN)/pip install -r requirements.txt --upgrade
	@echo "Dependencies synced successfully"

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf $(VENV)/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".eggs" -exec rm -rf {} +

test:
	$(VENV_BIN)/pytest tests/

lint:
	$(VENV_BIN)/flake8 stego_detector/
	$(VENV_BIN)/flake8 steganography/

format:
	$(VENV_BIN)/black stego_detector/
	$(VENV_BIN)/black steganography/

uninstall:
	$(PIP) uninstall -y steganography