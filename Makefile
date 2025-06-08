.PHONY: help build run stop test test-unit test-integration clean logs

# Default target
help:
	@echo "Medical Document Classifier - Development Commands"
	@echo "=================================================="
	@echo "build          - Build the Docker containers"
	@echo "run            - Start the API service"
	@echo "stop           - Stop all services"
	@echo "test           - Run all tests"
	@echo "test-unit      - Run unit tests only"
	@echo "test-integration - Run integration tests only"
	@echo "test-client    - Run the interactive test client"
	@echo "logs           - View API logs"
	@echo "clean          - Clean up containers and volumes"
	@echo "dev            - Start in development mode (with auto-reload)"

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

# Run all tests
test: test-unit test-integration

# Run unit tests
test-unit:
	python -m pytest tests/unit/ -v

# Run integration tests (requires API to be running)
test-integration:
	python -m pytest tests/integration/ -v

# Run the test client
test-client:
	python src/utils/test_client.py

# View API logs
logs:
	docker-compose logs -f api

# Clean up
clean:
	docker-compose down -v
	docker system prune -f

# Health check
health:
	@curl -s http://localhost:8000/health | python -m json.tool

# Quick setup for first time
setup: build run
	@echo "Waiting for API to be ready..."
	@sleep 10
	@make health 