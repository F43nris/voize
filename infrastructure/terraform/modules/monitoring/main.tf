# Enhanced prediction logging metrics
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
    "method" = "EXTRACT(httpRequest.requestMethod)"
  }
}

# Enhanced error tracking
resource "google_logging_metric" "api_errors" {
  name   = "ml_api_errors"
  filter = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\" AND (severity=\"ERROR\" OR httpRequest.status>=400)"
  
  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    display_name = "ML API Errors"
  }
  
  label_extractors = {
    "status_code" = "EXTRACT(httpRequest.status)"
    "error_type" = "EXTRACT(jsonPayload.error_type)"
  }
}

# Model performance metrics
resource "google_logging_metric" "model_predictions" {
  name   = "ml_model_predictions_logged"
  filter = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\" AND jsonPayload.event_type=\"prediction_logged\""
  
  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    display_name = "ML Model Predictions Logged"
  }
  
  label_extractors = {
    "model_version" = "EXTRACT(jsonPayload.model_version)"
    "prediction" = "EXTRACT(jsonPayload.prediction)"
  }
}

# Response time distribution metric
resource "google_logging_metric" "response_time_distribution" {
  name   = "ml_api_response_times"
  filter = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\" AND httpRequest.latency!=\"\""
  
  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "DISTRIBUTION"
    display_name = "ML API Response Time Distribution"
  }
  
  value_extractor = "EXTRACT(httpRequest.latency)"
  
  bucket_options {
    exponential_buckets {
      num_finite_buckets = 64
      growth_factor     = 2.0
      scale            = 0.01
    }
  }
}

# Business metrics - Request volume
resource "google_logging_metric" "request_volume" {
  name   = "ml_api_request_volume"
  filter = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\" AND httpRequest.requestUrl!=\"\""
  
  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    display_name = "ML API Request Volume"
  }
  
  label_extractors = {
    "endpoint" = "EXTRACT(httpRequest.requestUrl)"
    "user_agent" = "EXTRACT(httpRequest.userAgent)"
  }
}

# Model confidence metrics
resource "google_logging_metric" "low_confidence_predictions" {
  name   = "ml_model_low_confidence"
  filter = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\" AND jsonPayload.confidence<0.7"
  
  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    display_name = "ML Model Low Confidence Predictions"
  }
  
  label_extractors = {
    "model_version" = "EXTRACT(jsonPayload.model_version)"
  }
}

# Enhanced alerting policies

# High error rate alert
resource "google_monitoring_alert_policy" "high_error_rate" {
  display_name = "High Error Rate - ML API"
  combiner     = "OR"
  enabled      = true
  
  conditions {
    display_name = "Error rate > 5%"
    
    condition_threshold {
      filter          = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0.05
      
      aggregations {
        alignment_period     = "300s"
        per_series_aligner   = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_MEAN"
        group_by_fields      = ["resource.labels.service_name"]
      }
      
      trigger {
        count = 1
      }
    }
  }
  
  notification_channels = var.notification_channels
  
  documentation {
    content = "Error rate for ML API has exceeded 5% for 5 minutes. Check logs and model performance."
  }
}

# High response time alert
resource "google_monitoring_alert_policy" "high_response_time" {
  display_name = "High Response Time - ML API"
  combiner     = "OR"
  enabled      = true
  
  conditions {
    display_name = "P95 response time > 5 seconds"
    
    condition_threshold {
      filter          = "metric.type=\"logging.googleapis.com/user/ml_api_response_times\" AND resource.type=\"cloud_run_revision\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 5000  # 5 seconds in milliseconds
      
      aggregations {
        alignment_period     = "300s"
        per_series_aligner   = "ALIGN_DELTA"
        cross_series_reducer = "REDUCE_PERCENTILE_95"
        group_by_fields      = ["resource.labels.service_name"]
      }
      
      trigger {
        count = 1
      }
    }
  }
  
  notification_channels = var.notification_channels
  
  documentation {
    content = "P95 response time for ML API has exceeded 5 seconds. Check system resources and model performance."
  }
}

# Low confidence rate alert (model drift indicator)
resource "google_monitoring_alert_policy" "high_low_confidence_rate" {
  display_name = "High Low Confidence Rate - Model Drift Alert"
  combiner     = "OR"
  enabled      = true
  
  conditions {
    display_name = "Low confidence predictions > 30%"
    
    condition_threshold {
      filter          = "metric.type=\"logging.googleapis.com/user/ml_model_low_confidence\" AND resource.type=\"cloud_run_revision\""
      duration        = "600s"  # 10 minutes
      comparison      = "COMPARISON_GT"
      threshold_value = 0.3
      
      aggregations {
        alignment_period     = "600s"
        per_series_aligner   = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_MEAN"
        group_by_fields      = ["resource.labels.service_name"]
      }
      
      trigger {
        count = 1
      }
    }
  }
  
  notification_channels = var.notification_channels
  
  documentation {
    content = "Model is producing >30% low confidence predictions. This may indicate model drift or data quality issues. Consider retraining."
  }
}

# Low request volume alert (potential downtime)
resource "google_monitoring_alert_policy" "low_request_volume" {
  display_name = "Low Request Volume - Potential Downtime"
  combiner     = "OR"
  enabled      = true
  
  conditions {
    display_name = "Request volume < 1 req/min for 10 minutes"
    
    condition_threshold {
      filter          = "metric.type=\"logging.googleapis.com/user/ml_api_request_volume\" AND resource.type=\"cloud_run_revision\""
      duration        = "600s"
      comparison      = "COMPARISON_LT"
      threshold_value = 1.0
      
      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_SUM"
        group_by_fields      = ["resource.labels.service_name"]
      }
      
      trigger {
        count = 10  # 10 consecutive minutes
      }
    }
  }
  
  notification_channels = var.notification_channels
  
  documentation {
    content = "Request volume is very low (<1 req/min for 10 minutes). Check if service is accessible and functioning."
  }
}

# Model staleness alert
resource "google_monitoring_alert_policy" "model_staleness" {
  display_name = "Model Staleness Warning"
  combiner     = "OR"
  enabled      = true
  
  conditions {
    display_name = "No predictions logged for 24 hours"
    
    condition_absent {
      filter   = "metric.type=\"logging.googleapis.com/user/ml_model_predictions_logged\" AND resource.type=\"cloud_run_revision\""
      duration = "86400s"  # 24 hours
      
      aggregations {
        alignment_period     = "3600s"
        per_series_aligner   = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_SUM"
        group_by_fields      = ["resource.labels.service_name"]
      }
      
      trigger {
        count = 1
      }
    }
  }
  
  notification_channels = var.notification_channels
  
  documentation {
    content = "No model predictions have been logged for 24 hours. Check service health and usage patterns."
  }
}

# Enhanced monitoring dashboard
resource "google_monitoring_dashboard" "ml_api_dashboard" {
  dashboard_json = jsonencode({
    displayName = "ML API Production Monitoring Dashboard"
    
    gridLayout = {
      widgets = [
        {
          title = "Request Rate & Volume"
          xyChart = {
            dataSets = [
              {
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"logging.googleapis.com/user/ml_api_request_volume\" AND resource.type=\"cloud_run_revision\""
                    aggregation = {
                      alignmentPeriod = "60s"
                      perSeriesAligner = "ALIGN_RATE"
                      crossSeriesReducer = "REDUCE_SUM"
                    }
                  }
                }
                plotType = "LINE"
                targetAxis = "Y1"
              }
            ]
            yAxis = {
              label = "Requests per second"
              scale = "LINEAR"
            }
          }
        },
        {
          title = "Error Rate"
          xyChart = {
            dataSets = [
              {
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"logging.googleapis.com/user/ml_api_errors\" AND resource.type=\"cloud_run_revision\""
                    aggregation = {
                      alignmentPeriod = "300s"
                      perSeriesAligner = "ALIGN_RATE"
                      crossSeriesReducer = "REDUCE_SUM"
                    }
                  }
                }
                plotType = "LINE"
                targetAxis = "Y1"
              }
            ]
            yAxis = {
              label = "Errors per second"
              scale = "LINEAR"
            }
          }
        },
        {
          title = "Response Time Distribution"
          xyChart = {
            dataSets = [
              {
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"logging.googleapis.com/user/ml_api_response_times\" AND resource.type=\"cloud_run_revision\""
                    aggregation = {
                      alignmentPeriod = "300s"
                      perSeriesAligner = "ALIGN_DELTA"
                      crossSeriesReducer = "REDUCE_PERCENTILE_95"
                    }
                  }
                }
                plotType = "LINE"
                targetAxis = "Y1"
              },
              {
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"logging.googleapis.com/user/ml_api_response_times\" AND resource.type=\"cloud_run_revision\""
                    aggregation = {
                      alignmentPeriod = "300s"
                      perSeriesAligner = "ALIGN_DELTA"
                      crossSeriesReducer = "REDUCE_PERCENTILE_50"
                    }
                  }
                }
                plotType = "LINE"
                targetAxis = "Y1"
              }
            ]
            yAxis = {
              label = "Response Time (ms)"
              scale = "LINEAR"
            }
          }
        },
        {
          title = "Model Performance - Predictions by Class"
          xyChart = {
            dataSets = [
              {
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"logging.googleapis.com/user/ml_model_predictions_logged\" AND resource.type=\"cloud_run_revision\""
                    aggregation = {
                      alignmentPeriod = "300s"
                      perSeriesAligner = "ALIGN_RATE"
                      crossSeriesReducer = "REDUCE_SUM"
                      groupByFields = ["metric.labels.prediction"]
                    }
                  }
                }
                plotType = "STACKED_AREA"
                targetAxis = "Y1"
              }
            ]
            yAxis = {
              label = "Predictions per second"
              scale = "LINEAR"
            }
          }
        },
        {
          title = "Model Confidence - Low Confidence Rate"
          xyChart = {
            dataSets = [
              {
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"logging.googleapis.com/user/ml_model_low_confidence\" AND resource.type=\"cloud_run_revision\""
                    aggregation = {
                      alignmentPeriod = "300s"
                      perSeriesAligner = "ALIGN_RATE"
                      crossSeriesReducer = "REDUCE_SUM"
                    }
                  }
                }
                plotType = "LINE"
                targetAxis = "Y1"
              }
            ]
            yAxis = {
              label = "Low confidence predictions/sec"
              scale = "LINEAR"
            }
          }
        },
        {
          title = "System Resources - Memory Usage"
          xyChart = {
            dataSets = [
              {
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"run.googleapis.com/container/memory/utilizations\" AND resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\""
                    aggregation = {
                      alignmentPeriod = "60s"
                      perSeriesAligner = "ALIGN_MEAN"
                      crossSeriesReducer = "REDUCE_MEAN"
                    }
                  }
                }
                plotType = "LINE"
                targetAxis = "Y1"
              }
            ]
            yAxis = {
              label = "Memory Utilization (%)"
              scale = "LINEAR"
            }
          }
        },
        {
          title = "System Resources - CPU Usage"
          xyChart = {
            dataSets = [
              {
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"run.googleapis.com/container/cpu/utilizations\" AND resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.service_name}\""
                    aggregation = {
                      alignmentPeriod = "60s"
                      perSeriesAligner = "ALIGN_MEAN"
                      crossSeriesReducer = "REDUCE_MEAN"
                    }
                  }
                }
                plotType = "LINE"
                targetAxis = "Y1"
              }
            ]
            yAxis = {
              label = "CPU Utilization (%)"
              scale = "LINEAR"
            }
          }
        },
        {
          title = "Instance Count"
          xyChart = {
            dataSets = [
              {
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
                targetAxis = "Y1"
              }
            ]
            yAxis = {
              label = "Instance Count"
              scale = "LINEAR"
            }
          }
        }
      ]
    }
  })
} 