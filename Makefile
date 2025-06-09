.PHONY: help build run stop test test-unit test-integration clean logs lint format security pre-commit ci-local

# Default target
help:
	@echo "Medical Document Classifier - Development Commands"
	@echo "=================================================="
	@echo ""
	@echo "🐳 Docker Commands:"
	@echo "  build          - Build the Docker containers"
	@echo "  run            - Start the API service"
	@echo "  stop           - Stop all services"
	@echo "  dev            - Start in development mode (with auto-reload)"
	@echo "  logs           - View API logs"
	@echo "  clean          - Clean up containers and volumes"
	@echo ""
	@echo "🧪 Testing Commands:"
	@echo "  test           - Run all tests"
	@echo "  test-unit      - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  test-client    - Run the interactive test client"
	@echo "  test-coverage  - Run tests with detailed coverage"
	@echo ""
	@echo "🔧 Code Quality Commands:"
	@echo "  lint           - Run all linting checks"
	@echo "  format         - Auto-format code (black + isort)"
	@echo "  format-check   - Check code formatting without fixing"
	@echo "  security       - Run security scans"
	@echo "  type-check     - Run type checking with mypy"
	@echo ""
	@echo "🚀 CI/CD Commands:"
	@echo "  pre-commit     - Setup and run pre-commit hooks"
	@echo "  ci-local       - Run full CI pipeline locally"
	@echo "  setup-dev      - Setup development environment"
	@echo ""
	@echo "🏥 Health & Setup:"
	@echo "  health         - Check API health"
	@echo "  setup          - Quick setup for first time"

# Docker Commands
# ===============

# Build the Docker containers
build:
	docker-compose build

# Start the API service
run:
	docker-compose up -d
	@echo "API starting... Check logs with 'make logs'"
	@echo "API will be available at http://localhost:8000"
	@echo "API docs at http://localhost:8000/docs"

# Start in development mode
dev:
	docker-compose up
	
# Stop all services
stop:
	docker-compose down

# View API logs
logs:
	docker-compose logs -f api

# Clean up
clean:
	docker-compose down -v
	docker system prune -f

# Testing Commands
# ================

# Run all tests
test: test-unit test-integration

# Run unit tests
test-unit:
	python -m pytest tests/unit/ -v

# Run integration tests (requires API to be running)
test-integration:
	python -m pytest tests/integration/ -v

# Run tests with detailed coverage
test-coverage:
	python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing --cov-report=xml

# Run the test client
test-client:
	python src/utils/test_client.py

# Code Quality Commands
# =====================

# Run all linting checks
lint: format-check type-check
	@echo "Running flake8..."
	flake8 src/ tests/ --max-line-length=88 --extend-ignore=E203,W503
	@echo "✅ All linting checks passed!"

# Auto-format code
format:
	@echo "Formatting code with black..."
	black src/ tests/
	@echo "Sorting imports with isort..."
	isort src/ tests/
	@echo "✅ Code formatting complete!"

# Check formatting without fixing
format-check:
	@echo "Checking code formatting..."
	black --check --diff src/ tests/
	@echo "Checking import sorting..."
	isort --check-only --diff src/ tests/
	@echo "✅ Formatting checks passed!"

# Run security scans
security:
	@echo "Running security scan with bandit..."
	bandit -r src/ -f json -o security-report.json || true
	@echo "Security report saved to security-report.json"
	@echo "Running Docker security scan..."
	docker run --rm -v $(PWD):/tmp/app aquasec/trivy fs /tmp/app --exit-code 0

# Run type checking
type-check:
	@echo "Running type checking with mypy..."
	mypy src/ --ignore-missing-imports
	@echo "✅ Type checking passed!"

# CI/CD Commands
# ==============

# Setup and run pre-commit hooks
pre-commit:
	@echo "Installing pre-commit hooks..."
	pre-commit install
	@echo "Running pre-commit on all files..."
	pre-commit run --all-files

# Run full CI pipeline locally
ci-local: format-check lint security test-coverage
	@echo "Building Docker image..."
	docker build -f infrastructure/docker/Dockerfile.api -t medical-classifier:ci-test .
	@echo "Testing Docker image..."
	docker run --rm -d --name ci-test -p 8001:8000 medical-classifier:ci-test
	@sleep 30
	@curl -f http://localhost:8001/health || (docker stop ci-test && exit 1)
	@curl -X POST http://localhost:8001/predict \
		-H "Content-Type: application/json" \
		-d '{"text": "Patient presents with chest pain."}' \
		| grep -q "prediction" || (docker stop ci-test && exit 1)
	@docker stop ci-test
	@echo "✅ Full CI pipeline completed successfully!"

# Setup development environment
setup-dev:
	@echo "Setting up development environment..."
	pip install -e ".[dev]"
	pre-commit install
	@echo "✅ Development environment ready!"

# Health & Setup Commands
# =======================

# Health check
health:
	@curl -s http://localhost:8000/health | python -m json.tool

# Quick setup for first time
setup: build run
	@echo "Waiting for API to be ready..."
	@sleep 10
	@make health

# Terraform Commands
# ==================

# Validate Terraform configuration
tf-validate:
	@echo "Validating Terraform configuration..."
	cd infrastructure/terraform && terraform init && terraform validate
	@echo "✅ Terraform validation passed!"

# Format Terraform files
tf-format:
	@echo "Formatting Terraform files..."
	cd infrastructure/terraform && terraform fmt -recursive
	@echo "✅ Terraform formatting complete!"

# Plan Terraform deployment
tf-plan:
	@echo "Planning Terraform deployment..."
	cd infrastructure/terraform && terraform plan

# Deployment Commands
# ===================

# Deploy to production (requires proper authentication)
deploy-prod:
	@echo "🚀 Deploying to production..."
	@echo "⚠️  Make sure you're authenticated with GCP!"
	./scripts/deploy.sh

# Test deployed service
test-prod:
	@echo "Testing production deployment..."
	./scripts/deploy.sh test

# View production logs
logs-prod:
	@echo "Fetching production logs..."
	./scripts/deploy.sh logs

# Clean Commands
# ==============

# Clean all generated files
clean-all: clean
	@echo "Cleaning generated files..."
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf security-report.json
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Cleanup complete!" 