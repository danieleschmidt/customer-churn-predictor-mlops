# Variables for ML Platform Terraform configuration

variable "environment" {
  description = "Environment name (production, staging, development)"
  type        = string
  default     = "production"
  
  validation {
    condition     = can(regex("^(production|staging|development)$", var.environment))
    error_message = "Environment must be production, staging, or development."
  }
}

variable "project_name" {
  description = "Name of the ML platform project"
  type        = string
  default     = "ml-platform"
}

variable "owner" {
  description = "Owner of the infrastructure"
  type        = string
  default     = "ml-team"
}

variable "cost_center" {
  description = "Cost center for billing purposes"
  type        = string
  default     = "engineering"
}

# Cloud Provider Configuration
variable "cloud_providers" {
  description = "Configuration for different cloud providers"
  type = object({
    aws = object({
      enabled            = bool
      region             = string
      availability_zones = list(string)
      vpc_cidr          = string
      enable_eks        = bool
      enable_ecs        = bool
    })
    azure = object({
      enabled             = bool
      region              = string
      resource_group_name = string
      vnet_address_space  = list(string)
      enable_aks          = bool
      enable_aci          = bool
    })
    gcp = object({
      enabled          = bool
      project_id       = string
      region           = string
      vpc_cidr         = string
      enable_gke       = bool
      enable_cloud_run = bool
    })
  })
  
  default = {
    aws = {
      enabled            = true
      region             = "us-west-2"
      availability_zones = ["us-west-2a", "us-west-2b", "us-west-2c"]
      vpc_cidr          = "10.0.0.0/16"
      enable_eks        = true
      enable_ecs        = false
    }
    azure = {
      enabled             = false
      region              = "East US 2"
      resource_group_name = "ml-platform-rg"
      vnet_address_space  = ["10.1.0.0/16"]
      enable_aks          = true
      enable_aci          = false
    }
    gcp = {
      enabled          = false
      project_id       = "ml-platform-project"
      region           = "us-central1"
      vpc_cidr         = "10.2.0.0/16"
      enable_gke       = true
      enable_cloud_run = true
    }
  }
}

# ML-specific Configuration
variable "ml_configuration" {
  description = "ML-specific infrastructure configuration"
  type = object({
    enable_gpu         = bool
    model_storage_size = string
    instance_types = object({
      aws = object({
        api_instances     = list(string)
        training_instances = list(string)
        gpu_instances      = list(string)
      })
      azure = object({
        api_instances     = list(string)
        training_instances = list(string)
        gpu_instances      = list(string)
      })
      gcp = object({
        api_instances     = list(string)
        training_instances = list(string)
        gpu_instances      = list(string)
      })
    })
    auto_scaling = object({
      min_replicas                   = number
      max_replicas                   = number
      cpu_target_utilization        = number
      memory_target_utilization     = number
      custom_metrics_enabled        = bool
    })
  })
  
  default = {
    enable_gpu         = false
    model_storage_size = "100Gi"
    instance_types = {
      aws = {
        api_instances     = ["m5.large", "m5.xlarge"]
        training_instances = ["c5.2xlarge", "c5.4xlarge"]
        gpu_instances      = ["p3.2xlarge", "g4dn.xlarge"]
      }
      azure = {
        api_instances     = ["Standard_D2s_v3", "Standard_D4s_v3"]
        training_instances = ["Standard_F8s_v2", "Standard_F16s_v2"]
        gpu_instances      = ["Standard_NC6s_v3", "Standard_ND6s"]
      }
      gcp = {
        api_instances     = ["n1-standard-2", "n1-standard-4"]
        training_instances = ["c2-standard-8", "c2-standard-16"]
        gpu_instances      = ["n1-standard-4", "n1-standard-8"]
      }
    }
    auto_scaling = {
      min_replicas                   = 3
      max_replicas                   = 50
      cpu_target_utilization        = 70
      memory_target_utilization     = 80
      custom_metrics_enabled        = true
    }
  }
}

# Database Configuration
variable "database_configuration" {
  description = "Database configuration settings"
  type = object({
    engine_version      = string
    instance_class      = string
    allocated_storage   = number
    max_allocated_storage = number
    backup_retention_period = number
    backup_window      = string
    maintenance_window = string
    multi_az          = bool
    encrypted         = bool
    performance_insights_enabled = bool
  })
  
  default = {
    engine_version      = "13.13"
    instance_class      = "db.r5.xlarge"
    allocated_storage   = 100
    max_allocated_storage = 1000
    backup_retention_period = 30
    backup_window      = "03:00-04:00"
    maintenance_window = "sun:04:00-sun:05:00"
    multi_az          = true
    encrypted         = true
    performance_insights_enabled = true
  }
}

# Cache Configuration
variable "cache_configuration" {
  description = "Redis cache configuration"
  type = object({
    node_type          = string
    num_cache_nodes    = number
    parameter_group_name = string
    engine_version     = string
    port              = number
    at_rest_encryption_enabled = bool
    transit_encryption_enabled = bool
  })
  
  default = {
    node_type          = "cache.r6g.large"
    num_cache_nodes    = 2
    parameter_group_name = "default.redis6.x"
    engine_version     = "6.2"
    port              = 6379
    at_rest_encryption_enabled = true
    transit_encryption_enabled = true
  }
}

# Monitoring Configuration
variable "monitoring" {
  description = "Monitoring and observability configuration"
  type = object({
    enable_prometheus    = bool
    enable_grafana      = bool
    enable_jaeger       = bool
    enable_elk_stack    = bool
    retention_days      = number
    slack_webhook_url   = string
    email_notifications = list(string)
    pagerduty_token    = string
    custom_dashboards  = list(string)
  })
  
  default = {
    enable_prometheus    = true
    enable_grafana      = true
    enable_jaeger       = true
    enable_elk_stack    = true
    retention_days      = 90
    slack_webhook_url   = ""
    email_notifications = []
    pagerduty_token    = ""
    custom_dashboards  = []
  }
}

# Security Configuration
variable "security" {
  description = "Security configuration settings"
  type = object({
    enable_vault            = bool
    enable_cert_manager     = bool
    enable_falco           = bool
    enable_opa_gatekeeper  = bool
    domain_name            = string
    certificate_email      = string
    enable_network_policies = bool
    enable_pod_security_policies = bool
    enable_image_scanning  = bool
    vulnerability_scan_on_push = bool
  })
  
  default = {
    enable_vault            = true
    enable_cert_manager     = true
    enable_falco           = true
    enable_opa_gatekeeper  = true
    domain_name            = "ml-platform.example.com"
    certificate_email      = "admin@example.com"
    enable_network_policies = true
    enable_pod_security_policies = true
    enable_image_scanning  = true
    vulnerability_scan_on_push = true
  }
}

# Disaster Recovery Configuration
variable "disaster_recovery" {
  description = "Disaster recovery configuration"
  type = object({
    primary_cloud            = string
    secondary_cloud          = string
    backup_retention_days    = number
    backup_schedule          = string
    enable_cross_region_backup = bool
    rpo_hours               = number
    rto_hours               = number
    enable_automated_failover = bool
  })
  
  default = {
    primary_cloud            = "aws"
    secondary_cloud          = "azure"
    backup_retention_days    = 90
    backup_schedule          = "0 2 * * *"
    enable_cross_region_backup = true
    rpo_hours               = 1
    rto_hours               = 4
    enable_automated_failover = false
  }
}

# Cost Optimization
variable "cost_optimization" {
  description = "Cost optimization settings"
  type = object({
    enable_spot_instances    = bool
    spot_instance_percentage = number
    enable_scheduled_scaling = bool
    scale_down_schedule     = string
    scale_up_schedule       = string
    enable_resource_tagging = bool
    budget_alerts_enabled   = bool
    monthly_budget_limit    = number
  })
  
  default = {
    enable_spot_instances    = true
    spot_instance_percentage = 30
    enable_scheduled_scaling = true
    scale_down_schedule     = "0 18 * * 1-5"  # Scale down at 6 PM on weekdays
    scale_up_schedule       = "0 8 * * 1-5"   # Scale up at 8 AM on weekdays
    enable_resource_tagging = true
    budget_alerts_enabled   = true
    monthly_budget_limit    = 10000
  }
}

# Feature Flags
variable "feature_flags" {
  description = "Feature flags for enabling/disabling features"
  type = object({
    enable_canary_deployments = bool
    enable_blue_green_deployments = bool
    enable_a_b_testing       = bool
    enable_chaos_engineering = bool
    enable_load_testing      = bool
    enable_synthetic_monitoring = bool
  })
  
  default = {
    enable_canary_deployments = true
    enable_blue_green_deployments = true
    enable_a_b_testing       = true
    enable_chaos_engineering = false
    enable_load_testing      = true
    enable_synthetic_monitoring = true
  }
}