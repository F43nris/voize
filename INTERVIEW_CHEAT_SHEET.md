# 🎯 Medical Document Classifier - Interview Cheat Sheet

## 📋 **Project Overview**
**What it is:** A production-ready ML Ops system for classifying medical documents using Multinomial Naive Bayes, deployed as a scalable FastAPI service on Google Cloud Run.

**Key Value:** Demonstrates end-to-end ML Ops capabilities - from data pipeline to production deployment with monitoring, CI/CD, and infrastructure automation.

---

## 🏗️ **Architecture Overview**

### **High-Level Components:**
1. **Data Pipeline** - Automated data processing and model training workflows
2. **ML API Service** - FastAPI application serving the trained model
3. **Monitoring System** - Business metrics, prediction logging, and observability
4. **Infrastructure** - Terraform-managed GCP resources with Docker containers
5. **CI/CD Pipeline** - GitHub Actions for automated testing and deployment

### **Tech Stack at a Glance:**
- **ML:** scikit-learn, pandas, numpy, NLTK
- **API:** FastAPI, Pydantic, uvicorn
- **Infrastructure:** GCP (Cloud Run, Artifact Registry), Terraform
- **CI/CD:** GitHub Actions, Docker
- **Monitoring:** Custom metrics tracking, structured logging
- **Data Pipeline:** Custom DAG scheduler with task orchestration

---

## 🔧 **Technical Deep Dive**

### **1. Data Pipeline & ML Training**

**Files:** `data-pipeline/` directory
- **`ml_dags.py`** - DAG definitions for ML workflows
- **`dag_scheduler.py`** - Custom task orchestration engine
- **`preprocess.py`** - Feature engineering pipeline
- **`optimize_multinomial_nb.py`** - Model training and hyperparameter optimization

**Key Features:**
- **DAG-based Workflows:** Custom scheduler for task dependencies (data quality → cleaning → feature engineering → training → validation)
- **Feature Engineering:** TF-IDF vectorization with NLTK preprocessing
- **Model Training:** Multinomial Naive Bayes with hyperparameter optimization
- **Data Quality Checks:** Automated validation of data completeness and quality

**Why These Choices:**
- **Multinomial NB:** Fast training, good performance on text classification, interpretable
- **Custom DAG Scheduler:** Demonstrates understanding of workflow orchestration concepts
- **TF-IDF Features:** Standard for text classification, computationally efficient

### **2. ML API Service**

**File:** `src/inference/api.py` (1057 lines)

**Core Endpoints:**
- `POST /predict` - Model inference with confidence scores
- `GET /health` - Health checks with model status
- `GET /monitoring/*` - Business metrics and performance data
- `GET /dags/*` - Pipeline management and status

**Key Technical Features:**
- **Model Loading:** Automatic model loading from local files or GCP Cloud Storage
- **Feature Engineering Pipeline:** Consistent preprocessing using saved `MedicalTextFeatureEngine`
- **Request Validation:** Pydantic models for input/output validation
- **Error Handling:** Comprehensive error responses with logging
- **Middleware:** Request tracking for business metrics

**Production-Ready Features:**
- **Health Checks:** Deep health checks including model load status
- **Monitoring Integration:** All requests tracked for business metrics
- **Structured Logging:** JSON-formatted logs for cloud environments
- **Resource Management:** Proper memory management for model loading

### **3. Monitoring & Observability**

**Files:** `src/monitoring/`

**Business Metrics (`business_metrics.py`):**
- **Real-time Metrics:** Request volume, response times, error rates
- **Performance Tracking:** P95/P99 response time percentiles
- **Anomaly Detection:** Automatic detection of unusual patterns
- **Hourly Trends:** Historical performance analysis

**Prediction Logging (`prediction_logger.py`):**
- **Request Tracking:** Every prediction logged with metadata
- **Model Performance:** Confidence distribution tracking
- **Drift Detection:** Data drift monitoring capabilities

**Why This Matters:**
- **Production Readiness:** Shows understanding of ML model monitoring needs
- **Business Value:** Tracking metrics that matter to stakeholders
- **Troubleshooting:** Enables quick identification of issues

### **4. Infrastructure as Code**

**Directory:** `infrastructure/terraform/`

**Key Resources:**
- **Cloud Run Service:** Managed container platform with auto-scaling
- **Artifact Registry:** Docker image storage and versioning
- **IAM & Security:** Proper service account permissions
- **Monitoring Setup:** Cloud logging and monitoring integration

**Infrastructure Choices:**
- **Cloud Run:** Serverless, pay-per-use, automatic scaling to zero
- **Terraform:** Infrastructure versioning and reproducible deployments
- **Modular Design:** Reusable Terraform modules for different environments

### **5. CI/CD Pipeline**

**File:** `.github/workflows/ci-cd.yml` (585 lines)

**Pipeline Stages:**
1. **Code Quality:** Black, isort, flake8, mypy type checking
2. **Security Scanning:** Trivy vulnerability scanning
3. **Testing:** Unit tests, integration tests, coverage reporting
4. **Docker Build:** Multi-stage builds with testing
5. **Deployment:** Automated deployment to GCP Cloud Run
6. **Performance Testing:** Load testing with validation

**Advanced Features:**
- **Parallel Jobs:** Optimized pipeline execution
- **Caching:** Pip dependencies and Docker layer caching
- **Security Integration:** SARIF upload to GitHub Security tab
- **Environment Promotion:** Different pipelines for PR vs main branch

---

## 💡 **Key Technical Decisions & Rationale**

### **Why FastAPI?**
- **Performance:** High-performance async framework
- **Developer Experience:** Automatic OpenAPI docs, type hints
- **Production Features:** Built-in validation, middleware support

### **Why Multinomial Naive Bayes?**
- **Interpretability:** Easy to understand and explain predictions
- **Performance:** Fast training and inference for text classification
- **Resource Efficiency:** Low memory footprint for production deployment

### **Why Custom DAG Scheduler?**
- **Learning Demonstration:** Shows understanding of workflow orchestration
- **Lightweight:** No heavyweight dependencies like Airflow for this demo
- **Customizable:** Tailored to specific ML pipeline needs

### **Why GCP Cloud Run?**
- **Serverless Benefits:** Auto-scaling, pay-per-request
- **Container Support:** Deploy any containerized application
- **Integration:** Native GCP logging and monitoring

### **Why Terraform?**
- **Infrastructure as Code:** Version-controlled, reproducible infrastructure
- **Multi-cloud:** Not vendor-locked, transferable skills
- **Team Collaboration:** Infrastructure changes through code review

---

## 🚀 **Deployment & Operations**

### **Local Development:**
```bash
make setup    # Build and start locally
make test     # Run all tests
make health   # Check API status
```

### **Production Deployment:**
```bash
# Automatic via GitHub Actions on merge to main
# Manual deployment:
cd infrastructure/terraform
terraform plan
terraform apply
```

### **Monitoring in Production:**
- **Health Endpoint:** `/health` for load balancer checks
- **Metrics Dashboard:** `/monitoring/metrics` for real-time stats
- **Performance Analysis:** `/monitoring/performance` for detailed metrics

---

## 📊 **Testing Strategy**

### **Test Coverage:**
- **Unit Tests:** Individual component testing
- **Integration Tests:** End-to-end API testing
- **Docker Tests:** Container functionality validation
- **Load Testing:** Performance validation under load

### **Quality Gates:**
- **Code Coverage:** Tracked with pytest-cov
- **Code Quality:** Multiple linters and formatters
- **Security Scanning:** Vulnerability detection
- **Performance Testing:** Response time validation

---

## 🎨 **Demonstration Points**

### **Show the Data Pipeline:**
```bash
# View DAG definitions
curl http://localhost:8000/dags

# Trigger model retraining
curl -X POST http://localhost:8000/dags/model_retraining/run
```

### **Show API Functionality:**
```bash
# Health check
curl http://localhost:8000/health

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient presents with chest pain and shortness of breath."}'

# View business metrics
curl http://localhost:8000/monitoring/metrics
```

### **Show Infrastructure:**
```bash
# View Terraform state
cd infrastructure/terraform && terraform show

# View deployed resources in GCP Console
```

### **Show CI/CD:**
- **GitHub Actions:** Demonstrate the workflow runs
- **Security Scanning:** Show security reports
- **Automated Deployment:** Show the deployment process

---

## 🤔 **Expected Interview Questions & Answers**

### **"How would you scale this system for 10x traffic?"**
- **Cloud Run:** Auto-scales containers based on request volume
- **Model Optimization:** Could switch to ONNX or TensorRT for faster inference
- **Caching:** Add Redis for frequent predictions
- **Load Balancing:** Already handled by Cloud Run
- **Database:** Add proper database for prediction logging at scale

### **"How do you handle model drift?"**
- **Monitoring:** Prediction confidence distribution tracking
- **Data Quality:** Automated checks in the DAG pipeline
- **Retraining Pipeline:** Automated model retraining workflow
- **A/B Testing:** Could implement shadow deployments for new models

### **"What about security?"**
- **Container Scanning:** Trivy vulnerability scanning in CI/CD
- **IAM:** Least-privilege service accounts
- **API Security:** Input validation with Pydantic
- **Secrets Management:** GCP Secret Manager integration
- **Network Security:** VPC and firewall rules (could be added)

### **"How do you ensure data quality?"**
- **Pipeline Validation:** Data quality checks in DAG workflow
- **Schema Validation:** Pydantic models for API requests
- **Monitoring:** Anomaly detection in business metrics
- **Testing:** Comprehensive test coverage for data processing

### **"What would you improve?"**
- **Model Store:** ML model registry like MLflow
- **Feature Store:** Centralized feature management
- **A/B Testing:** Framework for model experimentation
- **Advanced Monitoring:** More sophisticated drift detection
- **Multi-model:** Support for multiple model versions

---

## ⚡ **Key Strengths to Highlight**

1. **End-to-End Implementation:** Complete ML Ops pipeline from data to production
2. **Production-Ready:** Proper monitoring, logging, error handling
3. **Cloud-Native:** Leverages modern cloud services and patterns
4. **Automated Testing:** Comprehensive CI/CD with quality gates
5. **Observability:** Built-in monitoring and metrics tracking
6. **Infrastructure as Code:** Reproducible, version-controlled infrastructure
7. **Best Practices:** Code quality, security scanning, documentation

---

## 🎯 **Closing Statement**

*"This project demonstrates my understanding of the complete ML Ops lifecycle. I've built a production-ready system that handles the complexity of deploying, monitoring, and maintaining ML models in production. The architecture showcases scalability, observability, and maintainability - all critical for successful ML Ops in enterprise environments."* 