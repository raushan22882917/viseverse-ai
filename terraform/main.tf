# Terraform configuration for VisaVerse Guardian AI infrastructure
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

# Variables
variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "prod"
}

variable "gemini_api_key" {
  description = "Gemini API Key"
  type        = string
  sensitive   = true
}

variable "neo4j_password" {
  description = "Neo4j Database Password"
  type        = string
  sensitive   = true
}

# Provider configuration
provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

# Enable required APIs
resource "google_project_service" "apis" {
  for_each = toset([
    "run.googleapis.com",
    "cloudbuild.googleapis.com",
    "containerregistry.googleapis.com",
    "secretmanager.googleapis.com",
    "storage.googleapis.com",
    "aiplatform.googleapis.com",
    "monitoring.googleapis.com",
    "logging.googleapis.com",
    "cloudtrace.googleapis.com",
    "compute.googleapis.com"
  ])

  service = each.value
  project = var.project_id

  disable_dependent_services = false
  disable_on_destroy         = false
}

# Service Account for the application
resource "google_service_account" "visaverse_sa" {
  account_id   = "visaverse-guardian-ai"
  display_name = "VisaVerse Guardian AI Service Account"
  description  = "Service account for VisaVerse Guardian AI application"
  project      = var.project_id

  depends_on = [google_project_service.apis]
}

# IAM roles for the service account
resource "google_project_iam_member" "visaverse_roles" {
  for_each = toset([
    "roles/storage.objectViewer",
    "roles/aiplatform.user",
    "roles/secretmanager.secretAccessor",
    "roles/monitoring.metricWriter",
    "roles/logging.logWriter",
    "roles/cloudtrace.agent"
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.visaverse_sa.email}"

  depends_on = [google_service_account.visaverse_sa]
}

# Secret Manager secrets
resource "google_secret_manager_secret" "gemini_api_key" {
  secret_id = "gemini-api-key"
  project   = var.project_id

  replication {
    auto {}
  }

  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret_version" "gemini_api_key_version" {
  secret      = google_secret_manager_secret.gemini_api_key.id
  secret_data = var.gemini_api_key
}

resource "google_secret_manager_secret" "neo4j_password" {
  secret_id = "neo4j-password"
  project   = var.project_id

  replication {
    auto {}
  }

  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret_version" "neo4j_password_version" {
  secret      = google_secret_manager_secret.neo4j_password.id
  secret_data = var.neo4j_password
}

# Cloud Storage bucket for document uploads
resource "google_storage_bucket" "document_uploads" {
  name     = "${var.project_id}-visaverse-documents"
  location = var.region
  project  = var.project_id

  uniform_bucket_level_access = true
  
  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "Delete"
    }
  }

  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  depends_on = [google_project_service.apis]
}

# IAM for Cloud Storage
resource "google_storage_bucket_iam_member" "document_uploads_access" {
  bucket = google_storage_bucket.document_uploads.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.visaverse_sa.email}"
}

# Neo4j Compute Engine instance
resource "google_compute_instance" "neo4j" {
  name         = "visaverse-neo4j-${var.environment}"
  machine_type = "e2-standard-2"
  zone         = "${var.region}-a"
  project      = var.project_id

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = 50
      type  = "pd-ssd"
    }
  }

  network_interface {
    network = "default"
    access_config {
      // Ephemeral public IP
    }
  }

  metadata_startup_script = templatefile("${path.module}/scripts/neo4j-startup.sh", {
    neo4j_password = var.neo4j_password
  })

  service_account {
    email  = google_service_account.visaverse_sa.email
    scopes = ["cloud-platform"]
  }

  tags = ["neo4j", "visaverse"]

  depends_on = [google_project_service.apis]
}

# Firewall rule for Neo4j
resource "google_compute_firewall" "neo4j_firewall" {
  name    = "visaverse-neo4j-firewall"
  network = "default"
  project = var.project_id

  allow {
    protocol = "tcp"
    ports    = ["7474", "7687"]
  }

  source_ranges = ["10.0.0.0/8"]  # Internal GCP networks only
  target_tags   = ["neo4j"]

  depends_on = [google_project_service.apis]
}

# Cloud Run service
resource "google_cloud_run_v2_service" "visaverse_api" {
  name     = "visaverse-guardian-ai"
  location = var.region
  project  = var.project_id

  template {
    service_account = google_service_account.visaverse_sa.email
    
    scaling {
      min_instance_count = 1
      max_instance_count = 10
    }

    containers {
      image = "gcr.io/${var.project_id}/visaverse-guardian-ai:latest"
      
      ports {
        container_port = 8000
      }

      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
        cpu_idle = true
      }

      env {
        name  = "GCP_PROJECT_ID"
        value = var.project_id
      }

      env {
        name  = "ENVIRONMENT"
        value = var.environment
      }

      env {
        name  = "NEO4J_URI"
        value = "bolt://${google_compute_instance.neo4j.network_interface[0].network_ip}:7687"
      }

      env {
        name = "GEMINI_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.gemini_api_key.secret_id
            version = "latest"
          }
        }
      }

      env {
        name = "NEO4J_PASSWORD"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.neo4j_password.secret_id
            version = "latest"
          }
        }
      }

      env {
        name  = "DOCUMENT_BUCKET"
        value = google_storage_bucket.document_uploads.name
      }
    }
  }

  traffic {
    percent = 100
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
  }

  depends_on = [
    google_project_service.apis,
    google_service_account.visaverse_sa,
    google_project_iam_member.visaverse_roles
  ]
}

# Cloud Run IAM for public access
resource "google_cloud_run_service_iam_member" "public_access" {
  service  = google_cloud_run_v2_service.visaverse_api.name
  location = google_cloud_run_v2_service.visaverse_api.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Cloud Monitoring alerts
resource "google_monitoring_alert_policy" "high_error_rate" {
  display_name = "VisaVerse High Error Rate"
  project      = var.project_id
  
  conditions {
    display_name = "Error rate > 5%"
    
    condition_threshold {
      filter          = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"visaverse-guardian-ai\""
      duration        = "300s"
      comparison      = "COMPARISON_GREATER_THAN"
      threshold_value = 0.05
      
      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_RATE"
      }
    }
  }

  notification_channels = []
  
  alert_strategy {
    auto_close = "1800s"
  }

  depends_on = [google_project_service.apis]
}

resource "google_monitoring_alert_policy" "high_latency" {
  display_name = "VisaVerse High Latency"
  project      = var.project_id
  
  conditions {
    display_name = "95th percentile latency > 5s"
    
    condition_threshold {
      filter          = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"visaverse-guardian-ai\""
      duration        = "300s"
      comparison      = "COMPARISON_GREATER_THAN"
      threshold_value = 5000
      
      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_DELTA"
        cross_series_reducer = "REDUCE_PERCENTILE_95"
      }
    }
  }

  notification_channels = []
  
  alert_strategy {
    auto_close = "1800s"
  }

  depends_on = [google_project_service.apis]
}

# Outputs
output "cloud_run_url" {
  description = "URL of the deployed Cloud Run service"
  value       = google_cloud_run_v2_service.visaverse_api.uri
}

output "neo4j_internal_ip" {
  description = "Internal IP address of Neo4j instance"
  value       = google_compute_instance.neo4j.network_interface[0].network_ip
}

output "document_bucket_name" {
  description = "Name of the document storage bucket"
  value       = google_storage_bucket.document_uploads.name
}

output "service_account_email" {
  description = "Email of the service account"
  value       = google_service_account.visaverse_sa.email
}