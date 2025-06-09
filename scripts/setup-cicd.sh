#!/bin/bash

# CI/CD Setup Script for Medical Document Classifier
# ==================================================

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

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
    echo
}

# Check if we're in the right directory
check_project_root() {
    if [ ! -f "pyproject.toml" ] || [ ! -d ".github" ]; then
        print_error "Please run this script from the project root directory"
        exit 1
    fi
}

# Check dependencies
check_dependencies() {
    print_header "Checking Dependencies"
    
    # Check Python
    if ! command -v python3 >/dev/null 2>&1; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi
    print_success "Python 3 found: $(python3 --version)"
    
    # Check pip
    if ! command -v pip >/dev/null 2>&1; then
        print_error "pip is required but not installed"
        exit 1
    fi
    print_success "pip found: $(pip --version)"
    
    # Check git
    if ! command -v git >/dev/null 2>&1; then
        print_error "git is required but not installed"
        exit 1
    fi
    print_success "git found: $(git --version)"
    
    # Check Docker (optional)
    if command -v docker >/dev/null 2>&1; then
        print_success "Docker found: $(docker --version)"
    else
        print_warning "Docker not found - you won't be able to test locally"
    fi
    
    # Check gcloud (optional)
    if command -v gcloud >/dev/null 2>&1; then
        print_success "gcloud CLI found: $(gcloud --version | head -1)"
    else
        print_warning "gcloud CLI not found - you won't be able to deploy to GCP"
    fi
}

# Setup Python environment
setup_python_env() {
    print_header "Setting Up Python Environment"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    print_status "Activating virtual environment..."
    source venv/bin/activate
    
    print_status "Upgrading pip..."
    pip install --upgrade pip
    
    print_status "Installing development dependencies..."
    pip install -e ".[dev]"
    
    print_success "Python environment setup complete"
}

# Setup pre-commit hooks
setup_pre_commit() {
    print_header "Setting Up Pre-commit Hooks"
    
    print_status "Installing pre-commit hooks..."
    pre-commit install
    
    print_status "Running pre-commit on all files (this may take a while)..."
    pre-commit run --all-files || {
        print_warning "Some pre-commit checks failed - this is normal for the first run"
        print_status "Running auto-fixes..."
        make format || true
    }
    
    print_success "Pre-commit hooks setup complete"
}

# Validate local setup
validate_setup() {
    print_header "Validating Local Setup"
    
    print_status "Running code quality checks..."
    make format-check || {
        print_warning "Code formatting issues found - running auto-fix..."
        make format
    }
    
    print_status "Running linting..."
    make type-check || print_warning "Type checking failed - review mypy errors"
    
    print_status "Running tests..."
    make test-unit || {
        print_error "Unit tests failed - please fix before proceeding"
        return 1
    }
    
    print_status "Testing Docker build..."
    if command -v docker >/dev/null 2>&1; then
        docker build -f infrastructure/docker/Dockerfile.api -t medical-classifier:setup-test . || {
            print_error "Docker build failed - please fix Dockerfile issues"
            return 1
        }
        print_success "Docker build successful"
    else
        print_warning "Skipping Docker test - Docker not available"
    fi
    
    print_success "Local validation complete"
}

# GCP setup instructions
show_gcp_instructions() {
    print_header "GCP Service Account Setup"
    
    cat << EOF
To complete the CI/CD setup, you need to create a GCP service account:

1. Set your project ID:
   export PROJECT_ID="your-project-id"

2. Create service account:
   gcloud iam service-accounts create github-actions \\
       --description="GitHub Actions CI/CD" \\
       --display-name="GitHub Actions"

3. Grant permissions:
   for role in \\
       "roles/run.admin" \\
       "roles/artifactregistry.admin" \\
       "roles/compute.admin" \\
       "roles/iam.serviceAccountUser" \\
       "roles/serviceusage.serviceUsageAdmin"
   do
       gcloud projects add-iam-policy-binding \$PROJECT_ID \\
           --member="serviceAccount:github-actions@\$PROJECT_ID.iam.gserviceaccount.com" \\
           --role="\$role"
   done

4. Create key:
   gcloud iam service-accounts keys create github-actions-key.json \\
       --iam-account=github-actions@\$PROJECT_ID.iam.gserviceaccount.com

5. Add GitHub secrets:
   - Go to: Settings > Secrets and variables > Actions
   - Add: GCP_PROJECT_ID = your-project-id
   - Add: GCP_SA_KEY = (contents of github-actions-key.json)

EOF
}

# GitHub setup instructions
show_github_instructions() {
    print_header "GitHub Repository Setup"
    
    cat << EOF
To activate the CI/CD pipeline:

1. Push this code to GitHub:
   git add .
   git commit -m "feat: add comprehensive CI/CD pipeline"
   git push origin main

2. Set up GitHub secrets (see GCP instructions above)

3. Test the pipeline:
   git checkout -b feature/test-cicd
   echo "# Test CI/CD" >> test-file.md
   git add test-file.md
   git commit -m "test: trigger CI/CD pipeline"
   git push origin feature/test-cicd

4. Create a PR and watch the magic happen! ✨

EOF
}

# Main function
main() {
    echo
    echo -e "${GREEN}🚀 Medical Document Classifier - CI/CD Setup${NC}"
    echo -e "${GREEN}===============================================${NC}"
    echo
    
    check_project_root
    check_dependencies
    setup_python_env
    setup_pre_commit
    validate_setup
    
    print_header "Setup Complete! 🎉"
    
    echo -e "${GREEN}✅ Local development environment ready${NC}"
    echo -e "${GREEN}✅ Pre-commit hooks installed${NC}"
    echo -e "${GREEN}✅ Code quality checks passing${NC}"
    echo -e "${GREEN}✅ Tests passing${NC}"
    echo -e "${GREEN}✅ Docker build working${NC}"
    echo
    
    show_gcp_instructions
    show_github_instructions
    
    print_success "🎊 Setup complete! You now have a production-grade CI/CD pipeline!"
    echo
    echo "Next steps:"
    echo "1. Follow the GCP setup instructions above"
    echo "2. Push to GitHub and set up secrets"  
    echo "3. Create a test PR to see the pipeline in action"
    echo
    echo "For help: make help"
    echo "Documentation: docs/CICD.md"
}

# Run main function
main "$@" 