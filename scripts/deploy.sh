#!/bin/bash

# Medical Document Classifier - GCP Deployment Script
# ==================================================
# This script automates the deployment of the ML API to Google Cloud Platform

set -e  # Exit on any error

# Configuration
PROJECT_ID=${PROJECT_ID:-voize-462411}
REGION=${REGION:-us-central1}
IMAGE_NAME="medical-classifier"
REPOSITORY="ml-models"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required commands exist
check_dependencies() {
    print_status "Checking dependencies..."
    
    command -v gcloud >/dev/null 2>&1 || { print_error "gcloud CLI is required but not installed. Aborting."; exit 1; }
    command -v terraform >/dev/null 2>&1 || { print_error "terraform is required but not installed. Aborting."; exit 1; }
    command -v docker >/dev/null 2>&1 || { print_error "docker is required but not installed. Aborting."; exit 1; }
    
    print_success "All dependencies found"
}

# Get project configuration
get_project_config() {
    # Check if terraform.tfvars exists
    if [ ! -f "infrastructure/terraform/terraform.tfvars" ]; then
        print_error "terraform.tfvars not found. Please create it from terraform.tfvars.example"
        exit 1
    fi
    
    # Extract project ID from terraform.tfvars (only match the actual assignment)
    PROJECT_ID=$(grep '^project_id' infrastructure/terraform/terraform.tfvars | cut -d'"' -f2)
    REGION=$(grep '^region' infrastructure/terraform/terraform.tfvars | cut -d'"' -f2)
    
    # Fallback to default region if not found
    if [ -z "$REGION" ]; then
        REGION="us-central1"
    fi
    
    if [ -z "$PROJECT_ID" ]; then
        print_error "PROJECT_ID not found in terraform.tfvars"
        exit 1
    fi
    
    print_status "Using project: $PROJECT_ID"
    print_status "Using region: $REGION"
}

# Setup GCP authentication
setup_gcp_auth() {
    print_status "Setting up GCP authentication..."
    
    # Set the project
    gcloud config set project $PROJECT_ID
    
    # Enable Application Default Credentials
    if ! gcloud auth application-default print-access-token >/dev/null 2>&1; then
        print_status "Setting up Application Default Credentials..."
        gcloud auth application-default login
    fi
    
    print_success "GCP authentication configured"
}

# Deploy infrastructure with Terraform (first pass - without Cloud Run)
deploy_infrastructure_base() {
    print_status "Deploying base infrastructure with Terraform..."
    
    cd infrastructure/terraform
    
    # Initialize Terraform
    terraform init
    
    # Plan the deployment
    print_status "Planning Terraform deployment..."
    terraform plan -target=google_project_service.required_apis -target=google_artifact_registry_repository.ml_models
    
    # Apply base infrastructure first
    print_status "Creating Artifact Registry..."
    terraform apply -target=google_project_service.required_apis -target=google_artifact_registry_repository.ml_models -auto-approve
    
    cd ../..
    
    print_success "Base infrastructure deployed successfully"
}

# Deploy full infrastructure with Terraform
deploy_infrastructure_full() {
    print_status "Deploying full infrastructure with Terraform..."
    
    cd infrastructure/terraform
    
    # Plan the full deployment
    print_status "Planning full Terraform deployment..."
    terraform plan
    
    # Ask for confirmation
    echo
    read -p "Do you want to proceed with the full deployment? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "Deployment cancelled"
        exit 0
    fi
    
    # Apply the full configuration
    print_status "Applying full Terraform configuration..."
    if terraform apply -auto-approve; then
        # Get outputs
        CLOUD_RUN_URL=$(terraform output -raw cloud_run_url)
        
        cd ../..
        
        print_success "Full infrastructure deployed successfully"
        print_success "Service URL: $CLOUD_RUN_URL"
    else
        print_error "Terraform deployment failed!"
        
        # Get the service name and show logs
        SERVICE_NAME="medical-classifier-api"
        print_status "Fetching Cloud Run logs for debugging..."
        
        # Show recent logs
        echo "=== RECENT CLOUD RUN LOGS ==="
        gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=$SERVICE_NAME" \
            --limit=50 \
            --format="table(timestamp,severity,textPayload)" \
            --project=$PROJECT_ID || true
        
        echo
        print_error "Check the logs above for detailed error information"
        print_status "You can also view logs in the GCP Console:"
        print_status "https://console.cloud.google.com/logs/query?project=$PROJECT_ID"
        
        cd ../..
        exit 1
    fi
}

# Build and push Docker image
build_and_push_image() {
    print_status "Building and pushing Docker image..."
    
    # Configure Docker for Artifact Registry
    gcloud auth configure-docker $REGION-docker.pkg.dev
    
    # Build the image for x86_64 platform (Cloud Run architecture)
    IMAGE_URL="$REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_NAME:latest"
    print_status "Building image for x86_64 platform: $IMAGE_URL"
    
    docker build --platform linux/amd64 -f infrastructure/docker/Dockerfile.api -t $IMAGE_URL .
    
    # Push the image
    print_status "Pushing image to Artifact Registry..."
    docker push $IMAGE_URL
    
    print_success "Docker image built and pushed successfully"
}

# Test the deployed service
test_deployment() {
    print_status "Testing the deployed service..."
    
    cd infrastructure/terraform
    CLOUD_RUN_URL=$(terraform output -raw cloud_run_url)
    cd ../..
    
    # Wait for service to be ready
    print_status "Waiting for service to be ready..."
    sleep 30
    
    # Test health endpoint
    print_status "Testing health endpoint..."
    if curl -f "$CLOUD_RUN_URL/health" > /dev/null 2>&1; then
        print_success "Health check passed"
    else
        print_error "Health check failed"
        return 1
    fi
    
    # Test prediction endpoint
    print_status "Testing prediction endpoint..."
    RESPONSE=$(curl -s -X POST "$CLOUD_RUN_URL/predict" \
        -H "Content-Type: application/json" \
        -d '{"text": "Patient presents with chest pain and shortness of breath."}')
    
    if echo $RESPONSE | grep -q "prediction"; then
        print_success "Prediction test passed"
        echo "Response: $RESPONSE"
    else
        print_error "Prediction test failed"
        echo "Response: $RESPONSE"
        return 1
    fi
    
    print_success "All tests passed!"
    print_success "Your ML API is deployed and running at: $CLOUD_RUN_URL"
}

# Main execution
main() {
    echo "🚀 Medical Document Classifier - GCP Deployment"
    echo "=============================================="
    echo
    
    check_dependencies
    get_project_config
    setup_gcp_auth
    deploy_infrastructure_base
    build_and_push_image
    deploy_infrastructure_full
    test_deployment
    
    echo
    print_success "🎉 Deployment completed successfully!"
    echo
    echo "Next steps:"
    echo "1. Visit the GCP Console to view your resources"
    echo "2. Check the monitoring dashboard"
    echo "3. Try the API at the URL shown above"
    echo
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "destroy")
        print_status "Destroying infrastructure..."
        cd infrastructure/terraform
        terraform destroy
        cd ../..
        print_success "Infrastructure destroyed"
        ;;
    "test")
        test_deployment
        ;;
    "logs")
        print_status "Fetching Cloud Run logs..."
        get_project_config
        
        SERVICE_NAME="medical-classifier-api"
        LIMIT=${2:-100}
        
        echo "=== RECENT CLOUD RUN LOGS (last $LIMIT entries) ==="
        gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=$SERVICE_NAME" \
            --limit=$LIMIT \
            --format="table(timestamp,severity,textPayload)" \
            --project=$PROJECT_ID
        ;;
    *)
        echo "Usage: $0 [deploy|destroy|test|logs [limit]]"
        echo "  deploy: Deploy the full application"
        echo "  destroy: Destroy all infrastructure"
        echo "  test: Test the deployed application"
        echo "  logs [limit]: Show Cloud Run logs (default limit: 100)"
        exit 1
        ;;
esac 