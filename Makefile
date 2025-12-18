.PHONY: dev-install setup-dev setup-user test clean help

# Create dev environment and install kbkit in editable mode
dev-install:
	conda env create --name kbkit-dev -f environment.yaml
	conda run -n kbkit-dev pip install -e .

# One-liner for full dev setup: env + editable install + test
setup-dev: dev-install test

# Create user environment and install kbkit from local build
setup-user:
	conda create -n kbkit python=3.12 kbkit --use-local
	conda run -n kbkit pip install .

# Run tests using pytest
test:
	conda run -n kbkit-dev pytest tests

# Clean build artifacts
clean:
	rm -rf dist/ build/ .pytest_cache/ __pycache__/

# Show help
help:
	@echo "Available targets:"
	@echo "  make dev-install  - Create dev env and install kbkit in editable mode"
	@echo "  make setup-dev    - Full dev setup: env + install + test"
	@echo "  make setup-user   - Build and install into clean user env"
	@echo "  make test         - Run test suite"
	@echo "  make clean        - Remove build artifacts"
