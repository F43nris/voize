# Cloud Build trigger for load testing
resource "google_cloudbuild_trigger" "load_test" {
  name        = "ml-api-load-test"
  description = "Load test the ML API"
  
  # Webhook trigger (can be triggered via API)
  webhook_config {
    secret = google_secret_manager_secret_version.webhook_secret.secret_data
  }
  
  # Build configuration
  build {
    step {
      name = "gcr.io/cloud-builders/python"
      entrypoint = "bash"
      args = [
        "-c",
        <<EOF
pip install locust requests
cat > locustfile.py << 'EOL'
from locust import HttpUser, task, between
import json

class MLAPIUser(HttpUser):
    wait_time = between(1, 3)
    
    @task(3)
    def health_check(self):
        self.client.get("/health")
    
    @task(1)
    def predict(self):
        payload = {
            "text": "Patient presents with chest pain and shortness of breath."
        }
        self.client.post("/predict", json=payload)

if __name__ == "__main__":
    import subprocess
    import sys
    
    # Run a quick load test
    subprocess.run([
        sys.executable, "-m", "locust",
        "--host", "${var.target_url}",
        "--users", "10",
        "--spawn-rate", "2",
        "--run-time", "60s",
        "--headless"
    ])
EOL
python locustfile.py
EOF
      ]
    }
    
    # Store results
    step {
      name = "gcr.io/cloud-builders/gsutil"
      args = [
        "cp", 
        "locust_*.html",
        "gs://${google_storage_bucket.load_test_results.name}/results/"
      ]
    }
    
    options {
      logging = "CLOUD_LOGGING_ONLY"
    }
  }
}

# Secret for webhook trigger
resource "google_secret_manager_secret" "webhook_secret" {
  secret_id = "load-test-webhook-secret"
  
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "webhook_secret" {
  secret = google_secret_manager_secret.webhook_secret.id
  secret_data = "load-test-secret-key"
}

# Storage bucket for load test results
resource "google_storage_bucket" "load_test_results" {
  name     = "${var.project_id}-load-test-results"
  location = var.region
  
  uniform_bucket_level_access = true
  
  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "Delete"
    }
  }
}

# Cloud Scheduler job for regular load testing (optional)
resource "google_cloud_scheduler_job" "weekly_load_test" {
  name      = "weekly-ml-api-load-test"
  schedule  = "0 2 * * 1"  # Every Monday at 2 AM
  time_zone = "UTC"
  
  http_target {
    uri         = "https://cloudbuild.googleapis.com/v1/projects/${var.project_id}/triggers/${google_cloudbuild_trigger.load_test.trigger_id}:run"
    http_method = "POST"
    
    oauth_token {
      service_account_email = google_service_account.scheduler.email
    }
    
    body = base64encode(jsonencode({
      branchName = "main"
    }))
    
    headers = {
      "Content-Type" = "application/json"
    }
  }
}

# Service account for Cloud Scheduler
resource "google_service_account" "scheduler" {
  account_id   = "scheduler-load-test"
  display_name = "Scheduler Load Test Service Account"
}

# IAM binding for Cloud Scheduler to trigger builds
resource "google_project_iam_member" "scheduler_cloudbuild" {
  project = var.project_id
  role    = "roles/cloudbuild.builds.editor"
  member  = "serviceAccount:${google_service_account.scheduler.email}"
} 