terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.0"
    }
  }
}

# Configure the Google Cloud Provider
provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

# Enable required APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "cloudbuild.googleapis.com",
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "logging.googleapis.com",
    "monitoring.googleapis.com",
    "cloudresourcemanager.googleapis.com"
  ])
  
  service = each.key
  
  disable_dependent_services = true
  disable_on_destroy         = false
}

# Artifact Registry for container images
resource "google_artifact_registry_repository" "ml_models" {
  location      = var.region
  repository_id = "ml-models"
  description   = "Repository for ML model containers"
  format        = "DOCKER"
  
  depends_on = [google_project_service.required_apis]
}

# Cloud Run service
module "cloud_run" {
  source = "./modules/cloud_run"
  
  project_id      = var.project_id
  region          = var.region
  service_name    = "medical-classifier-api"
  image_url       = "${var.region}-docker.pkg.dev/${var.project_id}/ml-models/medical-classifier:latest"
  
  # Resource limits
  cpu_limit    = "1000m"
  memory_limit = "4Gi"
  
  # Scaling configuration
  min_instances = 0
  max_instances = 10
  
  # Environment variables
  env_vars = {
    ENV = "production"
  }
  
  depends_on = [
    google_project_service.required_apis,
    google_artifact_registry_repository.ml_models
  ]
}

# Monitoring and logging
module "monitoring" {
  source = "./modules/monitoring"
  
  project_id   = var.project_id
  service_name = module.cloud_run.service_name
  
  depends_on = [module.cloud_run]
}

# Load testing resources (optional for demo) - commented out for initial deployment
# module "load_testing" {
#   source = "./modules/load_testing"
#   
#   project_id  = var.project_id
#   region      = var.region
#   target_url  = module.cloud_run.service_url
#   
#   depends_on = [module.cloud_run]
# } 