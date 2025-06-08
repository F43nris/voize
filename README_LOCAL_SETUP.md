# Medical Document Classifier - Local Setup

This document describes how to run the Medical Document Classifier API locally using Docker.

## Prerequisites

- Docker and Docker Compose installed
- Python 3.9+ (for running tests locally)
- Make (optional, for using convenience commands)

## Quick Start

1. **Build and start the API:**
   ```bash
   make setup
   ```
   
   Or manually:
   ```bash
   docker-compose build
   docker-compose up -d
   ```

2. **Check that the API is running:**
   ```bash
   make health
   ```
   
   Or manually:
   ```bash
   curl http://localhost:8000/health
   ```

3. **Test the API:**
   ```bash
   make test-client
   ```

## Available Commands

Run `make help` to see all available commands:

- `make build` - Build Docker containers
- `make run` - Start API service in background
- `make dev` - Start API service with logs visible
- `make stop` - Stop all services
- `make test` - Run all tests
- `make test-client` - Interactive test client
- `make logs` - View API logs
- `make clean` - Clean up containers

## API Endpoints

Once running, the API will be available at `http://localhost:8000`:

- **Health Check:** `GET /health`
- **Prediction:** `POST /predict`
- **API Documentation:** `GET /docs` (Swagger UI)
- **OpenAPI Spec:** `GET /openapi.json`

## Example Usage

### Using curl:
```bash
# Health check
curl http://localhost:8000/health

# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient presents with chest pain and shortness of breath."}'
```

### Using the test client:
```bash
python src/utils/test_client.py
```

### Using Python requests:
```python
import requests

# Make a prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "Patient has diabetes and needs insulin adjustment."}
)
print(response.json())
```

## Testing

### Unit Tests
```bash
make test-unit
# or
python -m pytest tests/unit/ -v
```

### Integration Tests
First ensure the API is running, then:
```bash
make test-integration
# or  
python -m pytest tests/integration/ -v
```

## Troubleshooting

### API not responding
1. Check if containers are running:
   ```bash
   docker-compose ps
   ```

2. Check logs:
   ```bash
   make logs
   ```

3. Restart services:
   ```bash
   make stop
   make run
   ```

### Model loading issues
The API loads the trained model files from the `data/` directory. Ensure these files exist:
- `data/optimized_multinomial_nb.pkl`
- `data/feature_engine.pkl`

### Port conflicts
If port 8000 is already in use, modify the port mapping in `docker-compose.yml`:
```yaml
ports:
  - "8001:8000"  # Change 8000 to another port
```

## Development

For development with auto-reload:
```bash
make dev
```

The API code is mounted as a volume, so changes to the source code will automatically reload the API.

## What's Next?

After successfully running the local setup:

1. **Run comprehensive tests** to ensure everything works
2. **Try the interactive test client** with your own medical texts
3. **Review the API documentation** at `http://localhost:8000/docs`
4. **Move to cloud deployment** (AWS infrastructure with Terraform)

## Architecture

The local setup includes:
- **FastAPI Application:** Serves the ML model via REST API
- **Docker Container:** Isolated environment with all dependencies
- **Health Checks:** Automatic monitoring of service health
- **Shared Volumes:** Data persistence and development convenience 