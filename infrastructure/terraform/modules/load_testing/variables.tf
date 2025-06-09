variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "region" {
  description = "The GCP region"
  type        = string
}

variable "target_url" {
  description = "URL of the service to load test"
  type        = string
} 