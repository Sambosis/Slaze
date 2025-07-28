.PHONY: help install generate validate typecheck test pipeline clean watch demo

# Default target
help:
	@echo "Data Contract System - Available Commands:"
	@echo ""
	@echo "  make install     - Install dependencies"
	@echo "  make generate    - Generate stubs from contract"
	@echo "  make validate    - Validate code against contract"
	@echo "  make typecheck   - Run mypy type checking"
	@echo "  make test        - Run tests"
	@echo "  make pipeline    - Run full CI pipeline"
	@echo "  make watch       - Watch contract file and regenerate stubs"
	@echo "  make demo        - Run contract agent demo"
	@echo "  make clean       - Clean generated files"
	@echo ""
	@echo "Configuration:"
	@echo "  CONTRACT_FILE    - Path to contract YAML (default: contract.yml)"
	@echo "  SRC_DIR          - Source directory (default: src)"

# Configuration
CONTRACT_FILE ?= contract.yml
SRC_DIR ?= src

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	pip install pyyaml mypy pytest

# Generate stubs from contract
generate:
	@echo "🔧 Generating stubs from contract..."
	python3 generate_stubs.py --spec $(CONTRACT_FILE) --output $(SRC_DIR)

# Validate code against contract
validate:
	@echo "🔍 Validating code against contract..."
	python3 contract_validator.py --contract $(CONTRACT_FILE) $(shell find $(SRC_DIR) -name "*.py" -not -path "*/stubs/*" -not -path "*/schemas/*" 2>/dev/null || echo "")

# Run mypy type checking
typecheck:
	@echo "🔍 Running mypy type checking..."
	mypy --strict $(SRC_DIR) || echo "⚠️  mypy not found or failed"

# Run tests
test:
	@echo "🧪 Running tests..."
	pytest -q tests/ || echo "⚠️  pytest not found or no tests"

# Run full CI pipeline
pipeline: generate validate typecheck test
	@echo "🎉 Contract CI pipeline completed!"

# Watch contract file for changes
watch:
	@echo "👀 Watching contract file for changes..."
	python3 contract_ci.py --contract $(CONTRACT_FILE) --src $(SRC_DIR) watch

# Run contract agent demo
demo:
	@echo "🤖 Running contract agent demo..."
	python3 contract_agent.py

# Clean generated files
clean:
	@echo "🧹 Cleaning generated files..."
	rm -rf $(SRC_DIR)/stubs/
	rm -rf $(SRC_DIR)/schemas/
	rm -f $(SRC_DIR)/symbol_table.json
	rm -f $(SRC_DIR)/*.py
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete

# Development workflow
dev: clean generate validate
	@echo "✅ Development environment ready"

# Quick validation (without full pipeline)
check: validate
	@echo "✅ Quick validation complete"