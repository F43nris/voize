resource "google_cloud_run_v2_service" "ml_api" {
  name     = var.service_name
  location = var.region
  
  template {
    # Scaling configuration
    scaling {
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }
    
    containers {
      image = var.image_url
      
      # Resource limits
      resources {
        limits = {
          cpu    = var.cpu_limit
          memory = var.memory_limit
        }
        cpu_idle = false  # Keep CPU available during model loading
        startup_cpu_boost = true
      }
      
      # Environment variables
      dynamic "env" {
        for_each = var.env_vars
        content {
          name  = env.key
          value = env.value
        }
      }
      
      # Health check - more generous for ML model loading
      liveness_probe {
        http_get {
          path = "/health"
          port = 8000
        }
        initial_delay_seconds = 180
        timeout_seconds      = 30
        period_seconds       = 60
        failure_threshold    = 5
      }
      
      # Very lenient startup probe for ML model loading - up to 10 minutes
      startup_probe {
        http_get {
          path = "/health"
          port = 8000
        }
        initial_delay_seconds = 120  # Wait 2 minutes before first check
        timeout_seconds      = 30    # Each check can take 30s
        period_seconds       = 30    # Check every 30s
        failure_threshold    = 15    # Allow 15 failures = 120 + (30 * 15) = 570 seconds = 9.5 minutes total
      }
      
      ports {
        container_port = 8000
      }
    }
    
    # Execution environment
    execution_environment = "EXECUTION_ENVIRONMENT_GEN2"
    
    # Timeout
    timeout = "300s"
  }
  
  # Traffic configuration
  traffic {
    percent = 100
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
  }
  
  # Labels
  labels = {
    environment = var.environment
    service     = "ml-api"
    managed-by  = "terraform"
  }
}

# IAM policy to allow public access (adjust as needed for production)
resource "google_cloud_run_service_iam_member" "public_access" {
  service  = google_cloud_run_v2_service.ml_api.name
  location = google_cloud_run_v2_service.ml_api.location
  role     = "roles/run.invoker"
  member   = "allUsers"
} 