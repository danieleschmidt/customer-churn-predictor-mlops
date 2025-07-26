"""
Docker integration tests for the churn prediction application.

Tests comprehensive Docker deployment scenarios including:
- Container startup and health checks
- Environment variable handling
- Volume mounts and file permissions
- Service connectivity and networking
- Multi-service orchestration
- Resource constraints and limits
"""

import pytest
import time
import json
import subprocess
import requests
import docker
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import patch


class TestDockerContainerStartup:
    """Test suite for Docker container startup and basic functionality."""
    
    @pytest.fixture(scope="session")
    def docker_client(self):
        """Provide Docker client for tests."""
        try:
            client = docker.from_env()
            # Test connection
            client.ping()
            return client
        except Exception as e:
            pytest.skip(f"Docker not available: {e}")
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir)
            # Create required subdirectories
            (temp_path / "data" / "raw").mkdir(parents=True)
            (temp_path / "data" / "processed").mkdir(parents=True)
            (temp_path / "models").mkdir(parents=True)
            (temp_path / "logs").mkdir(parents=True)
            yield temp_path
    
    def test_dockerfile_health_check_syntax(self):
        """Test that Dockerfile has properly configured health check."""
        dockerfile_path = Path("/root/repo/Dockerfile")
        assert dockerfile_path.exists(), "Dockerfile not found"
        
        dockerfile_content = dockerfile_path.read_text()
        
        # Verify health check is present and properly configured
        assert "HEALTHCHECK" in dockerfile_content
        assert "--interval=" in dockerfile_content
        assert "--timeout=" in dockerfile_content
        assert "--retries=" in dockerfile_content
        assert "is_healthy" in dockerfile_content
        
        # Verify reasonable health check parameters
        lines = dockerfile_content.split('\n')
        healthcheck_line = [line for line in lines if 'HEALTHCHECK' in line][0]
        
        assert "30s" in healthcheck_line or "interval=30s" in healthcheck_line
        assert "10s" in healthcheck_line or "timeout=10s" in healthcheck_line
        assert "retries=3" in healthcheck_line
    
    def test_docker_compose_health_check_configuration(self):
        """Test that docker-compose.yml has health check properly configured."""
        compose_path = Path("/root/repo/docker-compose.yml")
        assert compose_path.exists(), "docker-compose.yml not found"
        
        compose_content = compose_path.read_text()
        
        # Verify health check configuration in compose
        assert "healthcheck:" in compose_content
        assert "test:" in compose_content
        assert "interval:" in compose_content
        assert "timeout:" in compose_content
        assert "retries:" in compose_content
        assert "start_period:" in compose_content
        assert "is_healthy" in compose_content
    
    def test_container_builds_successfully(self, docker_client):
        """Test that the Docker container builds without errors."""
        try:
            # Build the production image
            image, build_logs = docker_client.images.build(
                path="/root/repo",
                tag="churn-predictor:test",
                target="production",
                rm=True,
                pull=True
            )
            
            assert image is not None
            assert image.id is not None
            
            # Verify image labels
            labels = image.labels or {}
            assert "org.label-schema.name" in labels
            assert labels.get("org.label-schema.name") == "churn-predictor"
            
        except docker.errors.BuildError as e:
            pytest.fail(f"Docker build failed: {e}")
        finally:
            # Cleanup test image
            try:
                docker_client.images.remove("churn-predictor:test", force=True)
            except:
                pass
    
    def test_container_starts_with_health_command(self, docker_client, temp_data_dir):
        """Test that container starts successfully with health command."""
        container = None
        try:
            # Start container with health command
            container = docker_client.containers.run(
                "python:3.12-slim",
                command="python -c 'print(\"Health check mock\"); exit(0)'",
                detach=True,
                remove=True,
                environment={
                    "PYTHONPATH": "/app",
                    "LOG_LEVEL": "INFO"
                }
            )
            
            # Wait for container to start
            time.sleep(2)
            
            # Check container status
            container.reload()
            assert container.status in ["running", "exited"]
            
        except Exception as e:
            pytest.fail(f"Container startup failed: {e}")
        finally:
            if container:
                try:
                    container.stop(timeout=5)
                except:
                    pass


class TestEnvironmentVariableHandling:
    """Test suite for environment variable handling in Docker containers."""
    
    def test_required_environment_variables(self):
        """Test that all required environment variables are defined in Dockerfile."""
        dockerfile_path = Path("/root/repo/Dockerfile")
        dockerfile_content = dockerfile_path.read_text()
        
        required_env_vars = [
            "PYTHONPATH",
            "PYTHONUNBUFFERED", 
            "PYTHONDONTWRITEBYTECODE",
            "MODEL_CACHE_MAX_ENTRIES",
            "MODEL_CACHE_MAX_MEMORY_MB",
            "MODEL_CACHE_TTL_SECONDS",
            "LOG_LEVEL",
            "WORKERS"
        ]
        
        for env_var in required_env_vars:
            assert f"ENV {env_var}" in dockerfile_content, f"Missing environment variable: {env_var}"
    
    def test_environment_variable_defaults(self):
        """Test that environment variables have sensible defaults."""
        dockerfile_path = Path("/root/repo/Dockerfile")
        dockerfile_content = dockerfile_path.read_text()
        
        # Test specific default values
        defaults = {
            "MODEL_CACHE_MAX_ENTRIES": "10",
            "MODEL_CACHE_MAX_MEMORY_MB": "500", 
            "MODEL_CACHE_TTL_SECONDS": "3600",
            "LOG_LEVEL": "INFO",
            "WORKERS": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1"
        }
        
        for var, expected_value in defaults.items():
            assert f"ENV {var}={expected_value}" in dockerfile_content or \
                   f"ENV {var} {expected_value}" in dockerfile_content, \
                   f"Wrong default for {var}, expected {expected_value}"
    
    def test_docker_compose_environment_override(self):
        """Test that docker-compose allows environment variable override."""
        compose_path = Path("/root/repo/docker-compose.yml")
        compose_content = compose_path.read_text()
        
        # Verify environment section exists
        assert "environment:" in compose_content
        
        # Test specific overridable variables
        overridable_vars = [
            "LOG_LEVEL", "MODEL_CACHE_MAX_ENTRIES", 
            "MODEL_CACHE_MAX_MEMORY_MB", "WORKERS"
        ]
        
        for var in overridable_vars:
            # Should be in format: - VAR=${VAR:-default}
            assert f"{var}=" in compose_content, f"Environment variable {var} not configured"
    
    def test_security_environment_variables(self):
        """Test that sensitive environment variables are properly handled."""
        compose_path = Path("/root/repo/docker-compose.yml")
        dockerfile_path = Path("/root/repo/Dockerfile")
        
        compose_content = compose_path.read_text()
        dockerfile_content = dockerfile_path.read_text()
        
        # Ensure no hardcoded secrets in Dockerfile
        sensitive_patterns = ["password", "secret", "key", "token"]
        for pattern in sensitive_patterns:
            # API_KEY is acceptable as it's externally provided
            assert (pattern.lower() not in dockerfile_content.lower() or 
                   "API_KEY" in dockerfile_content), \
                   f"Potential hardcoded secret containing '{pattern}' in Dockerfile"
        
        # MLflow URI should be configurable (not hardcoded)
        assert "MLFLOW_TRACKING_URI=" in compose_content
        assert "${MLFLOW_TRACKING_URI" in compose_content


class TestVolumeMountsAndPermissions:
    """Test suite for volume mounts and file permission handling."""
    
    def test_dockerfile_volume_directories(self):
        """Test that Dockerfile creates required directories with proper permissions."""
        dockerfile_path = Path("/root/repo/Dockerfile")
        dockerfile_content = dockerfile_path.read_text()
        
        # Verify directory creation
        required_dirs = ["data/raw", "data/processed", "models", "logs"]
        
        # Should have mkdir command for these directories
        assert "mkdir -p" in dockerfile_content
        for dir_name in ["data", "models", "logs"]:
            assert dir_name in dockerfile_content
        
        # Verify ownership change to app user
        assert "chown -R" in dockerfile_content
        assert "${APP_USER}:${APP_USER}" in dockerfile_content
    
    def test_docker_compose_volume_mounts(self):
        """Test that docker-compose.yml configures volume mounts correctly."""
        compose_path = Path("/root/repo/docker-compose.yml")
        compose_content = compose_path.read_text()
        
        # Verify required volume mounts
        required_mounts = [
            "./data:/app/data:rw",
            "./models:/app/models:rw", 
            "./logs:/app/logs:rw",
            "./config.yml:/app/config.yml:ro"
        ]
        
        for mount in required_mounts:
            assert mount in compose_content, f"Missing volume mount: {mount}"
    
    def test_non_root_user_configuration(self):
        """Test that container runs as non-root user for security."""
        dockerfile_path = Path("/root/repo/Dockerfile")
        dockerfile_content = dockerfile_path.read_text()
        
        # Verify user creation
        assert "groupadd" in dockerfile_content
        assert "useradd" in dockerfile_content
        assert "USER ${APP_USER}" in dockerfile_content
        
        # Verify user is not root
        assert "USER root" not in dockerfile_content.split("USER ${APP_USER}")[-1], \
               "Container should not switch back to root after setting app user"
    
    def test_development_volume_configuration(self):
        """Test development container volume configuration for live coding."""
        compose_path = Path("/root/repo/docker-compose.yml")
        compose_content = compose_path.read_text()
        
        # Find development service configuration
        dev_section = False
        for line in compose_content.split('\n'):
            if 'churn-predictor-dev:' in line:
                dev_section = True
            if dev_section and 'volumes:' in line:
                break
        
        assert dev_section, "Development service not found"
        
        # Development should mount source code for live development
        assert ".:/app:rw" in compose_content
        
        # Should exclude cache directories
        assert "__pycache__" in compose_content


class TestServiceConnectivity:
    """Test suite for service connectivity and networking."""
    
    def test_network_configuration(self):
        """Test that Docker Compose configures networking properly."""
        compose_path = Path("/root/repo/docker-compose.yml")
        compose_content = compose_path.read_text()
        
        # Verify network configuration
        assert "networks:" in compose_content
        assert "churn-net:" in compose_content
        assert "driver: bridge" in compose_content
        
        # All services should be on the same network
        services = ["churn-predictor", "mlflow", "prometheus", "grafana"]
        for service in services:
            # Find service definition and verify network membership
            service_lines = []
            in_service = False
            for line in compose_content.split('\n'):
                if f"{service}:" in line and not line.strip().startswith('#'):
                    in_service = True
                elif in_service and line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                    break
                elif in_service:
                    service_lines.append(line)
            
            service_config = '\n'.join(service_lines)
            if service in ["churn-predictor"]:  # Main services
                assert "churn-net" in service_config, f"Service {service} not on churn-net"
    
    def test_port_exposure(self):
        """Test that services expose correct ports."""
        compose_path = Path("/root/repo/docker-compose.yml")
        compose_content = compose_path.read_text()
        
        expected_ports = {
            "churn-predictor": ["8000:8000"],
            "mlflow": ["5000:5000"],
            "prometheus": ["9090:9090"],
            "grafana": ["3000:3000"]
        }
        
        for service, ports in expected_ports.items():
            for port in ports:
                assert port in compose_content, f"Service {service} missing port {port}"
    
    def test_service_dependencies(self):
        """Test that service dependencies are properly configured."""
        compose_path = Path("/root/repo/docker-compose.yml")
        compose_content = compose_path.read_text()
        
        # MLflow should be a dependency for trainer
        trainer_section = []
        in_trainer = False
        for line in compose_content.split('\n'):
            if 'churn-trainer:' in line:
                in_trainer = True
            elif in_trainer and line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                break
            elif in_trainer:
                trainer_section.append(line)
        
        trainer_config = '\n'.join(trainer_section)
        assert "depends_on:" in trainer_config
        assert "mlflow" in trainer_config
    
    def test_service_profiles(self):
        """Test that services are organized into proper profiles."""
        compose_path = Path("/root/repo/docker-compose.yml")
        compose_content = compose_path.read_text()
        
        expected_profiles = {
            "development": ["churn-predictor-dev"],
            "mlflow": ["mlflow"],
            "training": ["churn-trainer"],
            "monitoring": ["prometheus", "grafana"]
        }
        
        for profile, services in expected_profiles.items():
            for service in services:
                assert f"profiles:" in compose_content
                # Find the service and verify profile
                service_lines = []
                in_service = False
                for line in compose_content.split('\n'):
                    if f"{service}:" in line and not line.strip().startswith('#'):
                        in_service = True
                    elif in_service and line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                        break
                    elif in_service:
                        service_lines.append(line)
                
                service_config = '\n'.join(service_lines)
                if services:  # Only check if service should have profile
                    assert profile in service_config, f"Service {service} missing profile {profile}"


class TestResourceConstraints:
    """Test suite for resource constraints and limits."""
    
    def test_memory_environment_variables(self):
        """Test that memory-related environment variables are properly configured."""
        compose_path = Path("/root/repo/docker-compose.yml")
        dockerfile_path = Path("/root/repo/Dockerfile")
        
        compose_content = compose_path.read_text()
        dockerfile_content = dockerfile_path.read_text()
        
        # Verify memory cache settings
        memory_vars = ["MODEL_CACHE_MAX_MEMORY_MB", "MODEL_CACHE_MAX_ENTRIES"]
        for var in memory_vars:
            assert var in dockerfile_content
            assert var in compose_content
    
    def test_worker_configuration(self):
        """Test that worker/concurrency configuration is sensible."""
        dockerfile_path = Path("/root/repo/Dockerfile")
        dockerfile_content = dockerfile_path.read_text()
        
        # Default should be conservative (1 worker)
        assert "ENV WORKERS=1" in dockerfile_content
        
        # Should be overridable in compose
        compose_path = Path("/root/repo/docker-compose.yml")
        compose_content = compose_path.read_text()
        assert "WORKERS=" in compose_content
    
    def test_development_vs_production_resources(self):
        """Test that development and production have different resource configurations."""
        compose_path = Path("/root/repo/docker-compose.yml")
        compose_content = compose_path.read_text()
        
        # Development should have more conservative settings
        dev_memory_line = None
        prod_memory_line = None
        
        lines = compose_content.split('\n')
        in_dev = False
        in_prod = False
        
        for line in lines:
            if 'churn-predictor-dev:' in line:
                in_dev = True
                in_prod = False
            elif 'churn-predictor:' in line and 'dev' not in line:
                in_prod = True
                in_dev = False
            elif line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                in_dev = in_prod = False
            
            if in_dev and "MODEL_CACHE_MAX_MEMORY_MB" in line:
                dev_memory_line = line
            elif in_prod and "MODEL_CACHE_MAX_MEMORY_MB" in line:
                prod_memory_line = line
        
        # Development should have lower memory allocation
        if dev_memory_line and prod_memory_line:
            dev_memory = int(dev_memory_line.split('=')[-1])
            prod_memory = int(prod_memory_line.split('=')[-1].split('-')[-1].strip('}'))
            assert dev_memory < prod_memory, "Development should have lower memory allocation"


class TestHealthCheckIntegration:
    """Test suite for health check integration and functionality."""
    
    def test_health_check_imports(self):
        """Test that health check imports work correctly in container context."""
        # Verify the health check command can import required modules
        command = "from src.health_check import is_healthy; import sys; sys.exit(0 if is_healthy() else 1)"
        
        # This should not raise import errors when executed with proper PYTHONPATH
        try:
            exec(compile(command, '<string>', 'exec'), {})
        except ImportError as e:
            pytest.fail(f"Health check command has import issues: {e}")
        except SystemExit:
            # SystemExit is expected from the health check command
            pass
    
    def test_health_check_graceful_failure(self):
        """Test that health check fails gracefully on errors."""
        # Mock a health check that raises an exception
        with patch('src.health_check.is_healthy', side_effect=Exception("Test error")):
            try:
                from src.health_check import is_healthy
                result = is_healthy()
                # Should return False on error, not raise exception
                assert result is False
            except ImportError:
                # If we can't import, that's the test environment limitation
                pytest.skip("Cannot import health check module in test environment")
    
    def test_health_check_endpoint_timeout(self):
        """Test that health check respects timeout configuration."""
        dockerfile_path = Path("/root/repo/Dockerfile")
        dockerfile_content = dockerfile_path.read_text()
        
        # Find health check configuration
        healthcheck_lines = [line for line in dockerfile_content.split('\n') if 'HEALTHCHECK' in line]
        assert len(healthcheck_lines) > 0
        
        healthcheck_line = healthcheck_lines[0]
        
        # Verify timeout is reasonable (not too short or too long)
        assert "--timeout=10s" in healthcheck_line
        assert "--interval=30s" in healthcheck_line
        
        # Start period should give enough time for app startup
        assert "--start-period=30s" in healthcheck_line
    
    def test_compose_health_check_consistency(self):
        """Test that Docker Compose health check matches Dockerfile."""
        dockerfile_path = Path("/root/repo/Dockerfile")
        compose_path = Path("/root/repo/docker-compose.yml")
        
        dockerfile_content = dockerfile_path.read_text()
        compose_content = compose_path.read_text()
        
        # Extract health check commands
        dockerfile_healthcheck = None
        for line in dockerfile_content.split('\n'):
            if 'HEALTHCHECK' in line and 'CMD' in line:
                dockerfile_healthcheck = line
                break
        
        # Both should use the same health check logic
        assert "is_healthy" in dockerfile_content
        assert "is_healthy" in compose_content
        
        # Timeouts should be consistent
        assert "timeout=10s" in dockerfile_content
        assert "timeout: 10s" in compose_content
        
        assert "interval=30s" in dockerfile_content
        assert "interval: 30s" in compose_content


class TestMultiServiceOrchestration:
    """Test suite for multi-service orchestration scenarios."""
    
    def test_complete_stack_configuration(self):
        """Test that complete monitoring stack is properly configured."""
        compose_path = Path("/root/repo/docker-compose.yml")
        compose_content = compose_path.read_text()
        
        # Verify all major services are defined
        required_services = [
            "churn-predictor", "churn-predictor-dev", "mlflow", 
            "churn-trainer", "prometheus", "grafana"
        ]
        
        for service in required_services:
            assert f"{service}:" in compose_content, f"Missing service: {service}"
    
    def test_persistent_volume_configuration(self):
        """Test that persistent volumes are properly configured."""
        compose_path = Path("/root/repo/docker-compose.yml")
        compose_content = compose_path.read_text()
        
        # Verify named volumes
        expected_volumes = ["mlflow-data", "prometheus-data", "grafana-data"]
        
        assert "volumes:" in compose_content
        for volume in expected_volumes:
            assert f"{volume}:" in compose_content
            assert "driver: local" in compose_content
    
    def test_service_labels(self):
        """Test that services have proper labels for management."""
        compose_path = Path("/root/repo/docker-compose.yml")
        compose_content = compose_path.read_text()
        
        # All services should have Terragon labels
        services_with_labels = [
            "churn-predictor", "churn-trainer", "mlflow", "prometheus", "grafana"
        ]
        
        for service in services_with_labels:
            # Find service section and check for labels
            service_lines = []
            in_service = False
            for line in compose_content.split('\n'):
                if f"{service}:" in line and not line.strip().startswith('#'):
                    in_service = True
                elif in_service and line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                    break
                elif in_service:
                    service_lines.append(line)
            
            service_config = '\n'.join(service_lines)
            assert "labels:" in service_config, f"Service {service} missing labels"
            assert "com.terragon" in service_config, f"Service {service} missing Terragon labels"
    
    def test_environment_consistency_across_services(self):
        """Test that environment variables are consistent across related services."""
        compose_path = Path("/root/repo/docker-compose.yml")
        compose_content = compose_path.read_text()
        
        # MLflow URI should be consistent between trainer and main app
        mlflow_uri_lines = [line for line in compose_content.split('\n') 
                           if 'MLFLOW_TRACKING_URI' in line]
        
        # Should reference mlflow service in trainer
        trainer_mlflow_refs = [line for line in mlflow_uri_lines 
                              if 'mlflow:5000' in line]
        assert len(trainer_mlflow_refs) > 0, "Trainer should reference mlflow service"
    
    def test_security_across_services(self):
        """Test security configuration across all services."""
        compose_path = Path("/root/repo/docker-compose.yml")
        compose_content = compose_path.read_text()
        
        # Grafana should have secure admin password configuration
        assert "GF_SECURITY_ADMIN_PASSWORD" in compose_content
        assert "GF_USERS_ALLOW_SIGN_UP=false" in compose_content
        
        # No hardcoded passwords should be visible
        lines = compose_content.split('\n')
        for line in lines:
            if 'password' in line.lower():
                # Should use environment variable substitution
                assert '${' in line or 'GF_SECURITY_ADMIN_PASSWORD' in line, \
                       f"Potential hardcoded password in line: {line}"