variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "service_name" {
  description = "Name of the service to monitor"
  type        = string
}

variable "notification_channels" {
  description = "List of notification channels for alerts"
  type        = list(string)
  default     = []
} 