# 🚀 CI/CD Pipeline Setup Guide

This guide will get your CI/CD pipeline up and running in **under 10 minutes**.

## ✅ What You Get

Your pipeline includes:
- 🧪 **Automated Testing** (unit, integration, coverage)
- 🔒 **Security Scanning** (vulnerabilities, secrets)
- 🐳 **Docker Build & Test** (multi-stage, optimized)
- ☁️ **GCP Deployment** (Cloud Run, Terraform)
- ⚡ **Performance Testing** (load testing with Locust)
- 📊 **Monitoring & Alerts** (custom metrics, dashboards)

## 🏃‍♂️ Quick Setup (5 minutes)

### 1. Create GitHub Secrets

Go to your GitHub repo → `Settings` → `Secrets and variables` → `Actions` → `New repository secret`:

```bash
GCP_PROJECT_ID    # Your GCP project ID (e.g., voize-462411)
GCP_SA_KEY        # Service account JSON key (see step 2)
```

### 2. Create GCP Service Account

```bash
# Set your project ID
export PROJECT_ID="your-project-id"

# Create service account
gcloud iam service-accounts create github-actions \
    --description="GitHub Actions CI/CD" \
    --display-name="GitHub Actions"

# Grant necessary permissions
for role in \
    "roles/run.admin" \
    "roles/artifactregistry.admin" \
    "roles/compute.admin" \
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

# Copy the JSON content and paste it as GCP_SA_KEY secret in GitHub
cat github-actions-key.json
```

### 3. Test Locally (Optional)

```bash
# Setup development environment
make setup-dev

# Run the full CI pipeline locally
make ci-local
```

### 4. Trigger Your First Deployment

```bash
# Create a feature branch
git checkout -b feature/test-ci-cd

# Make a small change
echo "# CI/CD Test" >> test-file.md

# Commit and push
git add .
git commit -m "test: trigger CI/CD pipeline"
git push origin feature/test-ci-cd

# Create PR on GitHub - this will trigger PR validation
# Merge PR - this will trigger full deployment
```

## 🎯 What Happens Next

### On Pull Request:
- ✅ Code quality checks (formatting, linting, types)
- ✅ Security scanning  
- ✅ Unit tests with coverage
- ✅ Docker build validation
- ✅ Terraform validation

### On Merge to Main:
- 🚀 Full CI pipeline (all PR checks)
- 🐳 Production Docker build & push
- ☁️ Terraform deployment to GCP
- 🧪 Integration testing on live service
- ⚡ Performance/load testing
- 📢 Success notification with service URL

## 📊 Monitoring Your Pipeline

### GitHub Actions Dashboard
- View all workflow runs: `Your Repo` → `Actions`
- See detailed logs for each step
- Download artifacts (test reports, performance reports)

### GCP Console
- **Cloud Run**: Monitor your deployed service
- **Artifact Registry**: See pushed Docker images
- **Cloud Logging**: View application logs
- **Cloud Monitoring**: Check custom metrics and alerts

## 🛠️ Local Development Commands

```bash
# Code quality
make format      # Auto-format code  
make lint        # Run all linting
make security    # Security scanning

# Testing
make test        # All tests
make test-coverage  # Tests with coverage report

# CI/CD
make ci-local    # Run full pipeline locally
make pre-commit  # Setup pre-commit hooks

# Deployment
make deploy-prod # Deploy to production
make test-prod   # Test production deployment
make logs-prod   # View production logs
```

## 🚨 Troubleshooting

### Common Issues

**❌ "GCP authentication failed"**
```bash
# Verify your service account key
echo $GCP_SA_KEY | base64 -d | jq .
```

**❌ "Tests failing locally"**
```bash
# Run tests to see detailed output
make test-coverage
```

**❌ "Docker build fails"**
```bash
# Test Docker build locally
docker build -f infrastructure/docker/Dockerfile.api .
```

**❌ "Terraform validation fails"**
```bash
# Check Terraform configuration
make tf-validate
```

### Get Help

1. Check the [full CI/CD documentation](docs/CICD.md)
2. View workflow logs in GitHub Actions
3. Check GCP logs in Cloud Console
4. Create an issue in the repository

## 🎉 Success Indicators

You'll know everything is working when:

1. ✅ **PR Checks Pass**: All green checkmarks on your PR
2. ✅ **Deployment Succeeds**: Workflow completes without errors  
3. ✅ **Service Responds**: Integration tests pass
4. ✅ **Performance OK**: Load tests complete
5. ✅ **Monitoring Active**: Metrics appear in GCP Console

## 🔄 Next Steps

Once your pipeline is working:

- [ ] **Add Slack/Email notifications** for deployment status
- [ ] **Set up staging environment** for safer deployments  
- [ ] **Configure advanced monitoring** with custom alerts
- [ ] **Add automated rollback** capabilities
- [ ] **Implement blue-green deployments** for zero downtime

---

**🎊 Congratulations!** You now have a production-grade CI/CD pipeline that would impress any ML engineering team!

Your setup demonstrates:
- ✅ **DevOps Best Practices** (IaC, automated testing, security)
- ✅ **MLOps Maturity** (model deployment, monitoring, performance testing)  
- ✅ **Production Readiness** (logging, alerting, scalability)
- ✅ **Professional Standards** (code quality, documentation, automation)

This is exactly what senior ML engineers are expected to build! 🚀 