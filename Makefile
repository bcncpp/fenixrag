# Variables
PYTHON := python
PYTEST := pytest
COVERAGE := coverage
UV := uv
# Project settings
PROJECT_NAME := fenix
SRC_DIR := src
TEST_DIR := tests
COVERAGE_DIR := htmlcov
REPORTS_DIR := reports

# Coverage settings
COVERAGE_MIN := 80
COVERAGE_FAIL_UNDER := 85

# Test file patterns
TEST_PATTERN := test_*.py
COVERAGE_OMIT := */tests/*,*/venv/*,*/__pycache__/*,*/migrations/*,*/node_modules/*
# Colors for output
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
BLUE := \033[0;34m
NC := \033[0m # No Color
.PHONY: setup-vectordb help install test coverage coverage-report  \
        coverage-badge test-unit test-integration test-e2e \
        test-watch test-parallel test-verbose test-failed quality lint format type-check \
        security pre-commit ci clean-all

setup-vectordb: uv run fenix --setup

help: ## Show this help message
	@echo "$(BLUE)Available targets:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(BLUE)Coverage targets:$(NC)"
	@echo "  $(GREEN)coverage$(NC)           Run tests with coverage"
	@echo "  $(GREEN)coverage-report$(NC)    Generate coverage report"
	@echo "  $(GREEN)coverage-badge$(NC)     Generate coverage badge"

install: ## Install dependencies
	@echo "$(BLUE)Installing dependencies...$(NC)"
	$(UV) sync

install-dev: install ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	$(UV) sync --group dev
	$(UV) run pre-commit install

## RUN TESTS
test: ## Run all tests
	@echo "$(BLUE)Running all tests...$(NC)"
	$(UV) run $(PYTEST) $(TEST_DIR) -v

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	$(UV) run $(PYTEST) $(TEST_DIR)/unit -v

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(NC)"
	$(UV) run $(PYTEST) $(TEST_DIR)/integration -v

test-watch: ## Run tests in watch mode
	@echo "$(BLUE)Running tests in watch mode...$(NC)"
	$(UV) run $(PYTEST) $(TEST_DIR) -f

test-parallel: ## Run tests in parallel
	@echo "$(BLUE)Running tests in parallel...$(NC)"
	$(UV) run $(PYTEST) $(TEST_DIR) -n auto

test-verbose: ## Run tests with verbose output
	@echo "$(BLUE)Running tests with verbose output...$(NC)"
	$(UV) run $(PYTEST) $(TEST_DIR) -vvv -s

test-failed: ## Re-run only failed tests
	@echo "$(BLUE)Re-running failed tests...$(NC)"
	$(UV) run $(PYTEST) $(TEST_DIR) --lf

coverage: ## Run tests with coverage
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	@mkdir -p $(REPORTS_DIR)
	$(UV) run $(COVERAGE) run --source=$(SRC_DIR) -m pytest $(TEST_DIR)
	@echo "$(GREEN)Coverage data collected$(NC)"

coverage-erase: ## Erase coverage data
	@echo "$(BLUE)Erasing coverage data...$(NC)"
	$(UV) run $(COVERAGE) erase

coverage-report: coverage ## Generate coverage report to terminal
	@echo "$(BLUE)Generating coverage report...$(NC)"
	$(UV) run $(COVERAGE) report --show-missing --skip-covered
	@echo ""
	@echo "$(BLUE)Coverage Summary:$(NC)"
	@$(UV) run $(COVERAGE) report --format=total | \
		awk '{if($$1>=$(COVERAGE_MIN)) print "$(GREEN)✓ Coverage: " $$1 "%$(NC)"; else print "$(RED)✗ Coverage: " $$1 "% (minimum: $(COVERAGE_MIN)%)$(NC)"}'
coverage-json: coverage ## Generate JSON coverage report
	@echo "$(BLUE)Generating JSON coverage report...$(NC)"
	@mkdir -p $(REPORTS_DIR)
	$(UV) run $(COVERAGE) json -o $(REPORTS_DIR)/coverage.json
	@echo "$(GREEN)JSON coverage report generated: $(REPORTS_DIR)/coverage.json$(NC)"

coverage-badge: coverage-json ## Generate coverage badge
	@echo "$(BLUE)Generating coverage badge...$(NC)"
	@mkdir -p $(REPORTS_DIR)
	$(UV) run coverage-badge -o $(REPORTS_DIR)/coverage.svg
	@echo "$(GREEN)Coverage badge generated: $(REPORTS_DIR)/coverage.svg$(NC)"

coverage-fail-under: coverage ## Check if coverage meets minimum threshold
	@echo "$(BLUE)Checking coverage threshold...$(NC)"
	$(UV) run $(COVERAGE) report --fail-under=$(COVERAGE_FAIL_UNDER)
	@echo "$(GREEN)✓ Coverage meets minimum threshold of $(COVERAGE_FAIL_UNDER)%$(NC)"

coverage-clean: ## Clean coverage files
	@echo "$(BLUE)Cleaning coverage files...$(NC)"
	rm -rf $(COVERAGE_DIR)
	rm -rf $(REPORTS_DIR)/coverage.*
	$(UV) run $(COVERAGE) erase
	@echo "$(GREEN)Coverage files cleaned$(NC)"
	