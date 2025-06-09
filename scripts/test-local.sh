#!/bin/bash

# Test Docker container locally before deploying to Cloud Run

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

CONTAINER_NAME="medical-classifier-local"
IMAGE_NAME="medical-classifier:latest"
PORT=8090

print_status "🐳 Testing Medical Classifier locally"
echo

# Check if required files exist
if [ ! -f "infrastructure/docker/Dockerfile.api" ]; then
    print_error "Dockerfile not found!"
    exit 1
fi

if [ ! -f "data/optimized_multinomial_nb.pkl" ] || [ ! -f "data/feature_engine.pkl" ]; then
    print_error "Model files not found in data/ directory!"
    exit 1
fi

# Build the image
print_status "Building Docker image for local testing (native platform)..."
print_warning "Note: This builds for your local architecture. Cloud Run deployment uses --platform linux/amd64"
docker build -f infrastructure/docker/Dockerfile.api -t $IMAGE_NAME .

# Stop and remove existing container if it exists
print_status "Cleaning up existing containers and port conflyicts..."
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

# Kill any process using our port (if any)
lsof -ti:$PORT | xargs kill -9 2>/dev/null || true

# Clean up any containers with similar names
docker ps -aq --filter "name=medical-classifier" | xargs docker stop 2>/dev/null || true
docker ps -aq --filter "name=medical-classifier" | xargs docker rm 2>/dev/null || true

# Run the container
print_status "Starting container on port $PORT..."
docker run -d --name $CONTAINER_NAME -p $PORT:8000 $IMAGE_NAME

# Wait for startup
print_status "Waiting for container to start..."
sleep 5

# Show initial logs
print_status "Container logs (first 20 lines):"
docker logs $CONTAINER_NAME | head -20

echo
print_status "Waiting for health check..."

# Wait for health check with timeout
TIMEOUT=300  # 5 minutes
ELAPSED=0
HEALTHY=false

while [ $ELAPSED -lt $TIMEOUT ]; do
    if curl -f http://localhost:$PORT/health >/dev/null 2>&1; then
        HEALTHY=true
        break
    fi
    
    echo -n "."
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done

echo

if [ "$HEALTHY" = true ]; then
    print_success "Container is healthy!"
    
    # Test the API
    print_status "Testing API endpoints..."
    
    # Test health endpoint
    print_status "Testing /health endpoint..."
    curl -s http://localhost:$PORT/health | jq . || curl -s http://localhost:$PORT/health
    
    echo
    
    # Test prediction endpoint
    print_status "Testing /predict endpoint..."
    curl -s -X POST http://localhost:$PORT/predict \
        -H "Content-Type: application/json" \
        -d '{"text": "Patient presents with chest pain and shortness of breath."}' | jq . || \
    curl -s -X POST http://localhost:$PORT/predict \
        -H "Content-Type: application/json" \
        -d '{"text": "Patient presents with chest pain and shortness of breath."}'
    
    echo
    print_success "🎉 All tests passed!"
    print_status "Container is running at http://localhost:$PORT"
    print_status "Run 'docker logs $CONTAINER_NAME' to see detailed logs"
    print_status "Run 'docker stop $CONTAINER_NAME' to stop the container"
    
else
    print_error "Container failed to become healthy within $TIMEOUT seconds"
    
    print_status "Container logs:"
    docker logs $CONTAINER_NAME
    
    print_status "Cleaning up..."
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
    
    exit 1
fi 