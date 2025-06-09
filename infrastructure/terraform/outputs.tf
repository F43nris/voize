output "cloud_run_url" {
  description = "URL of the deployed Cloud Run service"
  value       = module.cloud_run.service_url
}

output "artifact_registry_url" {
  description = "URL of the Artifact Registry repository"
  value       = google_artifact_registry_repository.ml_models.name
}

output "docker_image_url" {
  description = "Full Docker image URL for pushing"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/ml-models/medical-classifier:latest"
}

output "project_id" {
  description = "The GCP project ID"
  value       = var.project_id
}

output "region" {
  description = "The GCP region"
  value       = var.region
} 