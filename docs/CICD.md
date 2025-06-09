# CI/CD Pipeline Documentation

This document describes the automated CI/CD pipeline for the Medical Document Classifier project.

## 🏗️ Pipeline Overview

Our CI/CD pipeline ensures code quality, security, and reliable deployments through automated testing and deployment workflows.

### Pipeline Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Pull Request  │───▶│   PR Validation  │───▶│   Code Review   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Main Branch   │◀───│      Merge       │◀───│    Approval     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │
        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CI Pipeline   │───▶│   Build & Test   │───▶│   Security Scan │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │
        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CD Pipeline   │───▶│   Deploy to GCP  │───▶│ Integration Test│
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │
        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Performance Test│───▶│  Monitoring      │───▶│  Notification   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🔧 Setup Instructions

### 1. GitHub Secrets Configuration

Set up the following secrets in your GitHub repository (`Settings > Secrets and variables > Actions`):

```bash
# Required secrets
GCP_PROJECT_ID          # Your GCP project ID
GCP_SA_KEY              # Service account JSON key (base64 encoded)
```

### 2. Service Account Setup

Create a GCP service account with the following roles:
- `Cloud Run Admin`
- `Artifact Registry Administrator`
- `Compute Admin`
- `Security Admin`
- `Service Usage Admin`

```bash
# Create service account
gcloud iam service-accounts create github-actions \
    --description="Service account for GitHub Actions" \
    --display-name="GitHub Actions"

# Grant necessary roles
for role in \
    "roles/run.admin" \
    "roles/artifactregistry.admin" \
    "roles/compute.admin" \
    "roles/cloudsql.admin" \
    "roles/iam.serviceAccountUser" \
    "roles/serviceusage.serviceUsageAdmin"
do
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:github-actions@$PROJECT_ID.iam.gserviceaccount.com" \
        --role="$role"
done

# Create and download key
gcloud iam service-accounts keys create github-actions-key.json \
    --iam-account=github-actions@$PROJECT_ID.iam.gserviceaccount.com
```

### 3. Local Development Setup

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Install development dependencies
pip install -e ".[dev]"

# Run local quality checks
make lint        # Run all code quality checks
make test        # Run all tests
make format      # Auto-format code
```

## 📋 Workflow Details

### PR Validation Workflow (`.github/workflows/pr-check.yml`)

**Triggers:** Pull requests to `main` or `develop` branches

**Steps:**
1. **Code Quality Checks**
   - Black formatting validation
   - isort import sorting
   - flake8 linting
   - mypy type checking

2. **Testing**
   - Unit tests with coverage reporting
   - Coverage threshold enforcement (70%)

3. **Docker Validation**
   - Build Docker image
   - Test API functionality
   - Validate health checks

4. **Infrastructure Validation**
   - Terraform formatting check
   - Terraform configuration validation

**Success Criteria:** All checks must pass for PR approval

### Main CI/CD Workflow (`.github/workflows/ci-cd.yml`)

**Triggers:** Push to `main` branch

#### Job 1: Test & Code Quality 🧪
- Comprehensive code quality checks
- Unit test execution with coverage
- Coverage reporting to Codecov

#### Job 2: Security Scanning 🔒
- Trivy vulnerability scanning
- Results uploaded to GitHub Security tab
- SARIF format for integration

#### Job 3: Build & Test Docker 🐳
- Docker image build for testing
- Container functionality testing
- Health check validation
- Cleanup of test resources

#### Job 4: Production Build & Push 🚀
- Authenticate with GCP
- Build production Docker image
- Push to Artifact Registry
- Tag with both `latest` and commit SHA

#### Job 5: Infrastructure Deployment 🏗️
- Terraform initialization
- Infrastructure planning
- Automated deployment
- Output service URL for testing

#### Job 6: Integration Testing 🧪
- Wait for service readiness
- Health endpoint validation
- Prediction endpoint testing
- Response validation

#### Job 7: Performance Testing ⚡
- Load testing with Locust
- Performance report generation
- Artifact upload for analysis

#### Job 8: Notifications 📢
- Success/failure notifications
- Deployment status reporting
- Service URL sharing

## 🛠️ Development Workflow

### Recommended Git Flow

```bash
# 1. Create feature branch
git checkout -b feature/your-feature

# 2. Make changes and test locally
make test
make lint

# 3. Commit changes (pre-commit hooks will run)
git add .
git commit -m "feat: your feature description"

# 4. Push and create PR
git push origin feature/your-feature
# Create PR on GitHub

# 5. PR validation runs automatically
# Fix any issues and push again

# 6. After approval, merge to main
# CD pipeline runs automatically
```

### Code Quality Standards

- **Black** for code formatting (88 character line length)
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking
- **pytest** for testing (minimum 70% coverage)
- **bandit** for security scanning

### Testing Strategy

```bash
# Run specific test types
pytest tests/unit/              # Unit tests only
pytest tests/integration/       # Integration tests only
pytest -m "not slow"           # Skip slow tests
pytest --cov=src --cov-report=html  # Coverage report
```

## 🚨 Troubleshooting

### Common Issues

#### 1. GCP Authentication Failures
```bash
# Check service account permissions
gcloud auth list
gcloud config get-value project

# Verify service account key
echo $GCP_SA_KEY | base64 -d | jq .
```

#### 2. Test Failures
```bash
# Run tests locally first
make test

# Check specific test failures
pytest tests/unit/test_api.py -v

# Debug Docker build issues
docker build -f infrastructure/docker/Dockerfile.api .
```

#### 3. Terraform Issues
```bash
# Validate configuration
cd infrastructure/terraform
terraform init
terraform validate
terraform plan
```

#### 4. Pre-commit Hook Issues
```bash
# Update hooks
pre-commit autoupdate

# Run specific hooks
pre-commit run black --all-files
pre-commit run flake8 --all-files

# Skip hooks temporarily (not recommended)
git commit --no-verify
```

### Pipeline Monitoring

- **GitHub Actions**: Monitor workflow runs in the Actions tab
- **GCP Console**: Check Cloud Run deployments and logs
- **Artifact Registry**: Verify Docker image pushes
- **Cloud Monitoring**: Review application metrics

## 📊 Metrics & Monitoring

### CI/CD Metrics
- Build success rate
- Deployment frequency
- Lead time for changes
- Mean time to recovery

### Quality Metrics
- Test coverage percentage
- Security vulnerability count
- Code quality score
- Performance benchmarks

## 🔄 Continuous Improvement

### Optimization Opportunities
1. **Caching Strategy**: Improve build times with better caching
2. **Parallel Execution**: Run more jobs in parallel
3. **Selective Testing**: Run only affected tests
4. **Advanced Security**: Add more security scanning tools

### Future Enhancements
- [ ] Staging environment deployment
- [ ] Blue-green deployment strategy
- [ ] Automated rollback capabilities
- [ ] Performance regression detection
- [ ] Advanced monitoring alerts

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Google Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Terraform Documentation](https://www.terraform.io/docs)
- [Pre-commit Documentation](https://pre-commit.com/)

---

For questions or issues with the CI/CD pipeline, please create an issue in the repository or contact the development team. 