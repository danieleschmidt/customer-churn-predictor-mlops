# Customer Churn Predictor - Makefile
# Production-ready ML system with comprehensive automation

# =============================================================================
# CONFIGURATION
# =============================================================================

# Project settings
PROJECT_NAME := customer-churn-predictor
VERSION := $(shell grep version pyproject.toml | sed 's/.*"\(.*\)".*/\1/')
PYTHON := python3.12
PIP := pip

# Docker settings
DOCKER_REGISTRY := ghcr.io
DOCKER_REPO := yourorg/customer-churn-predictor
DOCKER_TAG := $(VERSION)
DOCKER_LATEST := latest

# Directories
SRC_DIR := src
TEST_DIR := tests
DOCS_DIR := docs
DATA_DIR := data
MODELS_DIR := models

# Colors for output
CYAN := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m

# =============================================================================
# HELP
# =============================================================================

.PHONY: help
help: ## Show this help message
	@echo "$(CYAN)Customer Churn Predictor - Available Commands$(RESET)"
	@echo ""
	@echo "$(GREEN)Development:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {if ($$1 ~ /^(install|setup|clean|format|lint|test)/) printf "  $(YELLOW)%-20s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)ML Pipeline:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {if ($$1 ~ /^(preprocess|train|evaluate|predict|pipeline)/) printf "  $(YELLOW)%-20s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Docker & Deployment:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {if ($$1 ~ /^(build|run|push|deploy)/) printf "  $(YELLOW)%-20s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Documentation & Quality:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {if ($$1 ~ /^(docs|security|performance|coverage)/) printf "  $(YELLOW)%-20s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# =============================================================================
# DEVELOPMENT SETUP
# =============================================================================

.PHONY: install
install: ## Install all dependencies
	@echo "$(CYAN)Installing dependencies...$(RESET)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .

.PHONY: install-prod
install-prod: ## Install production dependencies only
	@echo "$(CYAN)Installing production dependencies...$(RESET)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

.PHONY: setup
setup: install setup-git ## Complete development setup
	@echo "$(CYAN)Setting up development environment...$(RESET)"
	pre-commit install
	mkdir -p $(DATA_DIR)/raw $(DATA_DIR)/processed $(MODELS_DIR) logs
	@echo "$(GREEN)Development environment ready!$(RESET)"

.PHONY: setup-git
setup-git: ## Setup git hooks and configuration
	@echo "$(CYAN)Setting up git hooks...$(RESET)"
	pre-commit install
	pre-commit install --hook-type commit-msg

# =============================================================================
# CLEANING
# =============================================================================

.PHONY: clean
clean: clean-pyc clean-test clean-build ## Clean all artifacts

.PHONY: clean-pyc
clean-pyc: ## Remove Python bytecode files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +

.PHONY: clean-test
clean-test: ## Remove test and coverage artifacts
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mutmut-cache
	rm -rf test-results/

.PHONY: clean-build
clean-build: ## Remove build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/

.PHONY: clean-docker
clean-docker: ## Remove Docker artifacts
	docker system prune -f
	docker image prune -f

# =============================================================================
# CODE QUALITY
# =============================================================================

.PHONY: format
format: ## Format code with black and isort
	@echo "$(CYAN)Formatting code...$(RESET)"
	black $(SRC_DIR) $(TEST_DIR) scripts/
	isort $(SRC_DIR) $(TEST_DIR) scripts/

.PHONY: lint
lint: ## Run all linting checks
	@echo "$(CYAN)Running linting checks...$(RESET)"
	flake8 $(SRC_DIR) $(TEST_DIR) scripts/
	mypy $(SRC_DIR)
	black --check $(SRC_DIR) $(TEST_DIR) scripts/
	isort --check-only $(SRC_DIR) $(TEST_DIR) scripts/

.PHONY: security
security: ## Run security scans
	@echo "$(CYAN)Running security scans...$(RESET)"
	bandit -r $(SRC_DIR) -f json -o bandit-report.json || true
	safety check --json --output safety-report.json || true
	@echo "$(GREEN)Security scan complete. Check reports for details.$(RESET)"

.PHONY: pre-commit
pre-commit: ## Run pre-commit hooks on all files
	@echo "$(CYAN)Running pre-commit hooks...$(RESET)"
	pre-commit run --all-files

# =============================================================================
# TESTING
# =============================================================================

# =============================================================================
# COMPREHENSIVE TESTING FRAMEWORK
# =============================================================================

.PHONY: test
test: ## Run all tests with comprehensive framework
	@echo "$(CYAN)Running comprehensive test suite...$(RESET)"
	$(PYTHON) run_comprehensive_tests.py

.PHONY: test-unit
test-unit: ## Run unit tests only
	@echo "$(CYAN)Running unit tests...$(RESET)"
	$(PYTHON) run_comprehensive_tests.py --suites unit

.PHONY: test-integration
test-integration: ## Run integration tests only
	@echo "$(CYAN)Running integration tests...$(RESET)"
	$(PYTHON) run_comprehensive_tests.py --suites integration

.PHONY: test-performance
test-performance: ## Run performance tests
	@echo "$(CYAN)Running performance tests...$(RESET)"
	$(PYTHON) run_comprehensive_tests.py --suites performance

.PHONY: test-security
test-security: ## Run security tests
	@echo "$(CYAN)Running security tests...$(RESET)"
	$(PYTHON) run_comprehensive_tests.py --suites security

.PHONY: test-property
test-property: ## Run property-based tests
	@echo "$(CYAN)Running property-based tests...$(RESET)"
	$(PYTHON) run_comprehensive_tests.py --suites property

.PHONY: test-chaos
test-chaos: ## Run chaos engineering tests
	@echo "$(CYAN)Running chaos engineering tests...$(RESET)"
	$(PYTHON) run_comprehensive_tests.py --suites chaos

.PHONY: test-quality-gates
test-quality-gates: ## Run quality gates
	@echo "$(CYAN)Running quality gates...$(RESET)"
	$(PYTHON) run_comprehensive_tests.py --suites quality_gates

.PHONY: test-sequential
test-sequential: ## Run all tests sequentially
	@echo "$(CYAN)Running tests sequentially...$(RESET)"
	$(PYTHON) run_comprehensive_tests.py --no-parallel

.PHONY: test-fast
test-fast: ## Run fast test subset
	@echo "$(CYAN)Running fast test subset...$(RESET)"
	$(PYTHON) run_comprehensive_tests.py --suites unit security

.PHONY: coverage
coverage: ## Run tests with coverage analysis
	@echo "$(CYAN)Running comprehensive coverage analysis...$(RESET)"
	$(PYTHON) -c "from tests.quality_gates.coverage_analyzer import AdvancedCoverageAnalyzer; analyzer = AdvancedCoverageAnalyzer(); result = analyzer.run_comprehensive_analysis(); print(f'Coverage: {result.overall_coverage.line_coverage:.1f}%')"
	@echo "$(GREEN)Coverage reports generated in coverage_reports/$(RESET)"

.PHONY: test-components
test-components: ## Test new system components
	@echo "$(CYAN)Testing new system components...$(RESET)"
	pytest tests/test_comprehensive_new_components.py -v --tb=short

.PHONY: mutation-test
mutation-test: ## Run mutation testing
	@echo "$(CYAN)Running mutation tests...$(RESET)"
	mutmut run --paths-to-mutate $(SRC_DIR)
	mutmut html

# =============================================================================
# ML PIPELINE
# =============================================================================

.PHONY: preprocess
preprocess: ## Preprocess raw data
	@echo "$(CYAN)Preprocessing data...$(RESET)"
	$(PYTHON) scripts/run_preprocessing.py

.PHONY: train
train: ## Train the model
	@echo "$(CYAN)Training model...$(RESET)"
	$(PYTHON) scripts/run_training.py

.PHONY: evaluate
evaluate: ## Evaluate the model
	@echo "$(CYAN)Evaluating model...$(RESET)"
	$(PYTHON) scripts/run_evaluation.py --detailed

.PHONY: predict
predict: ## Run batch prediction
	@echo "$(CYAN)Running prediction...$(RESET)"
	$(PYTHON) scripts/run_prediction.py $(DATA_DIR)/processed/processed_features.csv --output_csv predictions.csv

.PHONY: pipeline
pipeline: preprocess train evaluate ## Run complete ML pipeline
	@echo "$(GREEN)ML pipeline completed successfully!$(RESET)"

.PHONY: monitor
monitor: ## Monitor model performance
	@echo "$(CYAN)Monitoring model performance...$(RESET)"
	$(PYTHON) scripts/run_monitor.py

# =============================================================================
# DOCKER
# =============================================================================

.PHONY: build
build: ## Build Docker image
	@echo "$(CYAN)Building Docker image...$(RESET)"
	docker build \
		--build-arg BUILD_DATE=$(shell date -u +'%Y-%m-%dT%H:%M:%SZ') \
		--build-arg VERSION=$(VERSION) \
		--build-arg VCS_REF=$(shell git rev-parse --short HEAD) \
		--target production \
		-t $(PROJECT_NAME):$(VERSION) \
		-t $(PROJECT_NAME):$(DOCKER_LATEST) \
		.

.PHONY: build-dev
build-dev: ## Build development Docker image
	@echo "$(CYAN)Building development Docker image...$(RESET)"
	docker build \
		--build-arg BUILD_DATE=$(shell date -u +'%Y-%m-%dT%H:%M:%SZ') \
		--build-arg VERSION=$(VERSION) \
		--build-arg VCS_REF=$(shell git rev-parse --short HEAD) \
		--target development \
		-t $(PROJECT_NAME):dev \
		.

.PHONY: run
run: ## Run Docker container
	@echo "$(CYAN)Running Docker container...$(RESET)"
	docker run --rm -p 8000:8000 \
		-e API_KEY=dev-api-key-for-testing \
		$(PROJECT_NAME):$(DOCKER_LATEST)

.PHONY: run-dev
run-dev: ## Run development Docker container
	@echo "$(CYAN)Running development Docker container...$(RESET)"
	docker run --rm -it -p 8000:8000 \
		-v $(PWD):/app \
		-e API_KEY=dev-api-key-for-testing \
		$(PROJECT_NAME):dev

.PHONY: push
push: build ## Push Docker image to registry
	@echo "$(CYAN)Pushing Docker image...$(RESET)"
	docker tag $(PROJECT_NAME):$(VERSION) $(DOCKER_REGISTRY)/$(DOCKER_REPO):$(VERSION)
	docker tag $(PROJECT_NAME):$(DOCKER_LATEST) $(DOCKER_REGISTRY)/$(DOCKER_REPO):$(DOCKER_LATEST)
	docker push $(DOCKER_REGISTRY)/$(DOCKER_REPO):$(VERSION)
	docker push $(DOCKER_REGISTRY)/$(DOCKER_REPO):$(DOCKER_LATEST)

# =============================================================================
# DOCKER COMPOSE
# =============================================================================

.PHONY: up
up: ## Start all services with docker-compose
	@echo "$(CYAN)Starting services...$(RESET)"
	docker-compose up -d

.PHONY: down
down: ## Stop all services
	@echo "$(CYAN)Stopping services...$(RESET)"
	docker-compose down

.PHONY: logs
logs: ## Show logs from all services
	docker-compose logs -f

.PHONY: restart
restart: down up ## Restart all services

# =============================================================================
# DOCUMENTATION
# =============================================================================

.PHONY: docs
docs: ## Build documentation
	@echo "$(CYAN)Building documentation...$(RESET)"
	cd $(DOCS_DIR) && mkdocs build

.PHONY: docs-serve
docs-serve: ## Serve documentation locally
	@echo "$(CYAN)Serving documentation...$(RESET)"
	cd $(DOCS_DIR) && mkdocs serve

.PHONY: docs-deploy
docs-deploy: ## Deploy documentation to GitHub Pages
	@echo "$(CYAN)Deploying documentation...$(RESET)"
	cd $(DOCS_DIR) && mkdocs gh-deploy

# =============================================================================
# API OPERATIONS
# =============================================================================

.PHONY: serve
serve: ## Start the API server locally
	@echo "$(CYAN)Starting API server...$(RESET)"
	uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

.PHONY: serve-prod
serve-prod: ## Start the API server in production mode
	@echo "$(CYAN)Starting API server (production)...$(RESET)"
	gunicorn src.api:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000

# =============================================================================
# LOAD TESTING
# =============================================================================

.PHONY: load-test
load-test: ## Run load tests with Locust
	@echo "$(CYAN)Running load tests...$(RESET)"
	locust -f tests/performance/locustfile.py --headless -u 50 -r 5 -t 60s --host http://localhost:8000

# =============================================================================
# CI/CD HELPERS
# =============================================================================

.PHONY: ci-test
ci-test: clean lint test coverage security ## Run CI test suite
	@echo "$(GREEN)CI tests completed successfully!$(RESET)"

.PHONY: ci-build
ci-build: clean build ## Build for CI/CD
	@echo "$(GREEN)CI build completed successfully!$(RESET)"

# =============================================================================
# DATABASE OPERATIONS (Future)
# =============================================================================

.PHONY: db-init
db-init: ## Initialize database
	@echo "$(YELLOW)Database operations not yet implemented$(RESET)"

.PHONY: db-migrate
db-migrate: ## Run database migrations
	@echo "$(YELLOW)Database operations not yet implemented$(RESET)"

# =============================================================================
# MAINTENANCE
# =============================================================================

.PHONY: update-deps
update-deps: ## Update dependencies
	@echo "$(CYAN)Updating dependencies...$(RESET)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1 | xargs -n1 $(PIP) install -U

.PHONY: check-deps
check-deps: ## Check for dependency vulnerabilities
	@echo "$(CYAN)Checking dependencies...$(RESET)"
	safety check
	$(PIP) list --outdated

.PHONY: version
version: ## Show current version
	@echo "$(GREEN)Current version: $(VERSION)$(RESET)"

# =============================================================================
# DEFAULT TARGET
# =============================================================================

.DEFAULT_GOAL := help