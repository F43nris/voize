# Log-based metric for prediction requests
resource "google_logging_metric" "prediction_requests" {
  name   = "ml_api_prediction_requests"
  filter = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\" AND httpRequest.requestUrl~\"/predict\""
  
  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    display_name = "ML API Prediction Requests"
  }
  
  label_extractors = {
    "status_code" = "EXTRACT(httpRequest.status)"
  }
}

# Log-based metric for errors
resource "google_logging_metric" "api_errors" {
  name   = "ml_api_errors"
  filter = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\" AND severity=\"ERROR\""
  
  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    display_name = "ML API Errors"
  }
}

# Alerting policy for high error rate
resource "google_monitoring_alert_policy" "high_error_rate" {
  display_name = "High Error Rate - ML API"
  combiner     = "OR"
  
  conditions {
    display_name = "Error rate > 5%"
    
    condition_threshold {
      filter          = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0.05
      
      aggregations {
        alignment_period   = "300s"
        per_series_aligner = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_MEAN"
        group_by_fields = ["resource.labels.service_name"]
      }
    }
  }
  
  notification_channels = var.notification_channels
}

# Alerting policy for high latency
resource "google_monitoring_alert_policy" "high_latency" {
  display_name = "High Latency - ML API"
  combiner     = "OR"
  
  conditions {
    display_name = "Response time > 5 seconds"
    
    condition_threshold {
      filter          = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 5000
      
      aggregations {
        alignment_period   = "300s"
        per_series_aligner = "ALIGN_MEAN"
        cross_series_reducer = "REDUCE_MEAN"
        group_by_fields = ["resource.labels.service_name"]
      }
    }
  }
  
  notification_channels = var.notification_channels
}

# Dashboard for monitoring
resource "google_monitoring_dashboard" "ml_api_dashboard" {
  dashboard_json = jsonencode({
    displayName = "ML API Monitoring Dashboard"
    
    gridLayout = {
      widgets = [
        {
          title = "Request Rate"
          xyChart = {
            dataSets = [{
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "metric.type=\"logging.googleapis.com/user/ml_api_prediction_requests\" AND resource.type=\"cloud_run_revision\""
                  aggregation = {
                    alignmentPeriod = "60s"
                    perSeriesAligner = "ALIGN_RATE"
                    crossSeriesReducer = "REDUCE_SUM"
                  }
                }
              }
              plotType = "LINE"
            }]
            yAxis = {
              label = "Requests per second"
            }
          }
        },
        {
          title = "Error Rate"
          xyChart = {
            dataSets = [{
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "metric.type=\"logging.googleapis.com/user/ml_api_errors\" AND resource.type=\"cloud_run_revision\""
                  aggregation = {
                    alignmentPeriod = "60s"
                    perSeriesAligner = "ALIGN_RATE"
                    crossSeriesReducer = "REDUCE_SUM"
                  }
                }
              }
              plotType = "LINE"
            }]
            yAxis = {
              label = "Errors per second"
            }
          }
        },
        {
          title = "Response Time"
          xyChart = {
            dataSets = [{
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "metric.type=\"run.googleapis.com/request_latencies\" AND resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\""
                  aggregation = {
                    alignmentPeriod = "60s"
                    perSeriesAligner = "ALIGN_MEAN"
                    crossSeriesReducer = "REDUCE_MEAN"
                  }
                }
              }
              plotType = "LINE"
            }]
            yAxis = {
              label = "Latency (ms)"
            }
          }
        },
        {
          title = "Instance Count"
          xyChart = {
            dataSets = [{
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "metric.type=\"run.googleapis.com/container/instance_count\" AND resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\""
                  aggregation = {
                    alignmentPeriod = "60s"
                    perSeriesAligner = "ALIGN_MEAN"
                    crossSeriesReducer = "REDUCE_MEAN"
                  }
                }
              }
              plotType = "LINE"
            }]
            yAxis = {
              label = "Instance Count"
            }
          }
        }
      ]
    }
  })
} 