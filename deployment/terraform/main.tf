# Main Terraform configuration for ML Platform infrastructure
terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }

  backend "s3" {
    bucket         = "ml-platform-terraform-state"
    key            = "production/terraform.tfstate"
    region         = "us-west-2"
    encrypt        = true
    dynamodb_table = "ml-platform-terraform-locks"
  }
}

# Local variables
locals {
  common_tags = {
    Environment   = var.environment
    Project      = var.project_name
    ManagedBy    = "terraform"
    Owner        = var.owner
    CostCenter   = var.cost_center
    Application  = "ml-platform"
  }

  # Multi-cloud configuration
  deploy_aws   = var.cloud_providers.aws.enabled
  deploy_azure = var.cloud_providers.azure.enabled
  deploy_gcp   = var.cloud_providers.gcp.enabled
}

# Random password generation for databases
resource "random_password" "db_password" {
  length  = 32
  special = true
}

resource "random_password" "jwt_secret" {
  length  = 64
  special = true
}

# AWS Resources (conditional)
module "aws_infrastructure" {
  count  = local.deploy_aws ? 1 : 0
  source = "./modules/aws"

  # Pass common variables
  environment     = var.environment
  project_name    = var.project_name
  common_tags     = local.common_tags
  
  # AWS-specific variables
  aws_region              = var.cloud_providers.aws.region
  vpc_cidr               = var.cloud_providers.aws.vpc_cidr
  availability_zones     = var.cloud_providers.aws.availability_zones
  enable_eks             = var.cloud_providers.aws.enable_eks
  enable_ecs             = var.cloud_providers.aws.enable_ecs
  
  # Database configuration
  db_password = random_password.db_password.result
  jwt_secret  = random_password.jwt_secret.result
  
  # ML-specific configuration
  ml_instance_types = var.ml_configuration.instance_types.aws
  enable_gpu        = var.ml_configuration.enable_gpu
  model_storage_size = var.ml_configuration.model_storage_size
}

# Azure Resources (conditional)
module "azure_infrastructure" {
  count  = local.deploy_azure ? 1 : 0
  source = "./modules/azure"

  # Pass common variables
  environment     = var.environment
  project_name    = var.project_name
  common_tags     = local.common_tags
  
  # Azure-specific variables
  azure_region           = var.cloud_providers.azure.region
  resource_group_name    = var.cloud_providers.azure.resource_group_name
  vnet_address_space     = var.cloud_providers.azure.vnet_address_space
  enable_aks             = var.cloud_providers.azure.enable_aks
  enable_aci             = var.cloud_providers.azure.enable_aci
  
  # Database configuration
  db_password = random_password.db_password.result
  jwt_secret  = random_password.jwt_secret.result
  
  # ML-specific configuration
  ml_instance_types = var.ml_configuration.instance_types.azure
  enable_gpu        = var.ml_configuration.enable_gpu
  model_storage_size = var.ml_configuration.model_storage_size
}

# GCP Resources (conditional)
module "gcp_infrastructure" {
  count  = local.deploy_gcp ? 1 : 0
  source = "./modules/gcp"

  # Pass common variables
  environment     = var.environment
  project_name    = var.project_name
  common_tags     = local.common_tags
  
  # GCP-specific variables
  gcp_project_id         = var.cloud_providers.gcp.project_id
  gcp_region             = var.cloud_providers.gcp.region
  vpc_cidr               = var.cloud_providers.gcp.vpc_cidr
  enable_gke             = var.cloud_providers.gcp.enable_gke
  enable_cloud_run       = var.cloud_providers.gcp.enable_cloud_run
  
  # Database configuration
  db_password = random_password.db_password.result
  jwt_secret  = random_password.jwt_secret.result
  
  # ML-specific configuration
  ml_instance_types = var.ml_configuration.instance_types.gcp
  enable_gpu        = var.ml_configuration.enable_gpu
  model_storage_size = var.ml_configuration.model_storage_size
}

# Monitoring and Observability
module "monitoring" {
  source = "./modules/monitoring"

  environment    = var.environment
  project_name   = var.project_name
  common_tags    = local.common_tags
  
  # Multi-cloud monitoring configuration
  aws_enabled   = local.deploy_aws
  azure_enabled = local.deploy_azure
  gcp_enabled   = local.deploy_gcp
  
  # Monitoring configuration
  enable_prometheus     = var.monitoring.enable_prometheus
  enable_grafana       = var.monitoring.enable_grafana
  enable_jaeger        = var.monitoring.enable_jaeger
  enable_elk_stack     = var.monitoring.enable_elk_stack
  
  # Alert configuration
  slack_webhook_url    = var.monitoring.slack_webhook_url
  email_notifications  = var.monitoring.email_notifications
  pagerduty_token     = var.monitoring.pagerduty_token
}

# Security
module "security" {
  source = "./modules/security"

  environment    = var.environment
  project_name   = var.project_name
  common_tags    = local.common_tags
  
  # Multi-cloud security configuration
  aws_enabled   = local.deploy_aws
  azure_enabled = local.deploy_azure
  gcp_enabled   = local.deploy_gcp
  
  # Security configuration
  enable_vault               = var.security.enable_vault
  enable_cert_manager        = var.security.enable_cert_manager
  enable_falco              = var.security.enable_falco
  enable_opa_gatekeeper      = var.security.enable_opa_gatekeeper
  
  # Certificate configuration
  domain_name               = var.security.domain_name
  certificate_email         = var.security.certificate_email
}

# Disaster Recovery
module "disaster_recovery" {
  source = "./modules/disaster-recovery"

  environment    = var.environment
  project_name   = var.project_name
  common_tags    = local.common_tags
  
  # Multi-cloud DR configuration
  primary_cloud   = var.disaster_recovery.primary_cloud
  secondary_cloud = var.disaster_recovery.secondary_cloud
  
  # Backup configuration
  backup_retention_days     = var.disaster_recovery.backup_retention_days
  backup_schedule          = var.disaster_recovery.backup_schedule
  enable_cross_region_backup = var.disaster_recovery.enable_cross_region_backup
}

# Outputs
output "infrastructure_endpoints" {
  description = "Infrastructure endpoints across all clouds"
  value = {
    aws = local.deploy_aws ? {
      eks_cluster_endpoint = module.aws_infrastructure[0].eks_cluster_endpoint
      ecs_cluster_name     = module.aws_infrastructure[0].ecs_cluster_name
      rds_endpoint         = module.aws_infrastructure[0].rds_endpoint
      elasticache_endpoint = module.aws_infrastructure[0].elasticache_endpoint
      load_balancer_dns    = module.aws_infrastructure[0].load_balancer_dns
    } : {}
    
    azure = local.deploy_azure ? {
      aks_cluster_fqdn      = module.azure_infrastructure[0].aks_cluster_fqdn
      container_registry_url = module.azure_infrastructure[0].container_registry_url
      postgres_fqdn         = module.azure_infrastructure[0].postgres_fqdn
      redis_hostname        = module.azure_infrastructure[0].redis_hostname
      application_gateway_ip = module.azure_infrastructure[0].application_gateway_ip
    } : {}
    
    gcp = local.deploy_gcp ? {
      gke_cluster_endpoint = module.gcp_infrastructure[0].gke_cluster_endpoint
      cloud_run_url        = module.gcp_infrastructure[0].cloud_run_url
      cloud_sql_ip         = module.gcp_infrastructure[0].cloud_sql_ip
      redis_ip             = module.gcp_infrastructure[0].redis_ip
      load_balancer_ip     = module.gcp_infrastructure[0].load_balancer_ip
    } : {}
  }
}

output "monitoring_endpoints" {
  description = "Monitoring and observability endpoints"
  value = {
    prometheus_url = module.monitoring.prometheus_url
    grafana_url    = module.monitoring.grafana_url
    jaeger_url     = module.monitoring.jaeger_url
    kibana_url     = module.monitoring.kibana_url
  }
}

output "security_outputs" {
  description = "Security-related outputs"
  sensitive   = true
  value = {
    vault_url           = module.security.vault_url
    certificate_status  = module.security.certificate_status
    policy_violations   = module.security.policy_violations
  }
}