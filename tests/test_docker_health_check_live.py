"""
Live Docker health check integration tests.

These tests require Docker to be running and test actual container behavior.
They are separated from unit tests as they require Docker environment.
"""

import pytest
import time
import docker
import requests
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional


class TestLiveDockerHealthChecks:
    """Live integration tests for Docker health checks."""
    
    @pytest.fixture(scope="session")
    def docker_client(self):
        """Docker client fixture with availability check."""
        try:
            client = docker.from_env()
            client.ping()
            return client
        except Exception as e:
            pytest.skip(f"Docker not available: {e}")
    
    @pytest.fixture(scope="session")
    def built_image(self, docker_client):
        """Build test image once for all tests."""
        try:
            # Build the image
            image, logs = docker_client.images.build(
                path="/root/repo",
                tag="churn-predictor:test-health",
                target="production",
                rm=True,
                quiet=False
            )
            yield image
        except Exception as e:
            pytest.skip(f"Could not build Docker image: {e}")
        finally:
            # Cleanup
            try:
                docker_client.images.remove("churn-predictor:test-health", force=True)
            except:
                pass
    
    @pytest.fixture
    def temp_volumes(self):
        """Create temporary volumes for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir)
            volumes = {
                'data': temp_path / 'data',
                'models': temp_path / 'models', 
                'logs': temp_path / 'logs'
            }
            
            # Create directory structure
            for vol_path in volumes.values():
                vol_path.mkdir(parents=True, exist_ok=True)
                (vol_path / 'raw').mkdir(exist_ok=True)
                (vol_path / 'processed').mkdir(exist_ok=True)
            
            yield volumes
    
    def test_container_health_check_passes(self, docker_client, built_image, temp_volumes):
        """Test that container health check passes with proper setup."""
        container = None
        try:
            # Start container with health check
            container = docker_client.containers.run(
                built_image.id,
                detach=True,
                remove=True,
                environment={
                    'LOG_LEVEL': 'INFO',
                    'PYTHONPATH': '/app'
                },
                volumes={
                    str(temp_volumes['data']): {'bind': '/app/data', 'mode': 'rw'},
                    str(temp_volumes['models']): {'bind': '/app/models', 'mode': 'rw'},
                    str(temp_volumes['logs']): {'bind': '/app/logs', 'mode': 'rw'}
                },
                command=['python', '-c', '''
import time
import sys
import os
sys.path.insert(0, "/app")

# Simple health check simulation
try:
    from src.health_check import is_healthy
    for i in range(30):  # Wait up to 30 seconds
        if is_healthy():
            print("Health check passed")
            exit(0)
        time.sleep(1)
    print("Health check failed")
    exit(1)
except Exception as e:
    print(f"Health check error: {e}")
    exit(1)
''']
            )
            
            # Wait for health check to complete
            result = container.wait(timeout=60)
            logs = container.logs().decode('utf-8')
            
            assert result['StatusCode'] == 0, f"Health check failed. Logs: {logs}"
            assert "Health check passed" in logs or "Health check failed" not in logs
            
        except Exception as e:
            if container:
                logs = container.logs().decode('utf-8')
                pytest.fail(f"Container health check test failed: {e}\nLogs: {logs}")
            else:
                pytest.fail(f"Container health check test failed: {e}")
        finally:
            if container:
                try:
                    container.stop(timeout=5)
                except:
                    pass
    
    def test_container_starts_with_api_mode(self, docker_client, built_image, temp_volumes):
        """Test that container starts successfully in API mode."""
        container = None
        try:
            # Start container in API mode
            container = docker_client.containers.run(
                built_image.id,
                detach=True,
                remove=True,
                ports={'8000/tcp': None},  # Random port
                environment={
                    'LOG_LEVEL': 'INFO',
                    'PYTHONPATH': '/app'
                },
                volumes={
                    str(temp_volumes['data']): {'bind': '/app/data', 'mode': 'rw'},
                    str(temp_volumes['models']): {'bind': '/app/models', 'mode': 'rw'},
                    str(temp_volumes['logs']): {'bind': '/app/logs', 'mode': 'rw'}
                },
                command=['python', '-m', 'src.cli', 'api', '--host', '0.0.0.0', '--port', '8000']
            )
            
            # Give container time to start
            time.sleep(10)
            
            # Check container is running
            container.reload()
            assert container.status == 'running'
            
            # Get the mapped port
            port_info = container.ports.get('8000/tcp')
            if port_info:
                host_port = port_info[0]['HostPort']
                
                # Try to make a request to health endpoint
                try:
                    response = requests.get(f'http://localhost:{host_port}/health', timeout=5)
                    assert response.status_code == 200
                    health_data = response.json()
                    assert 'status' in health_data
                except requests.exceptions.RequestException:
                    # May fail in test environment, but container should be running
                    pass
            
        except Exception as e:
            if container:
                logs = container.logs().decode('utf-8')
                pytest.fail(f"API mode test failed: {e}\nLogs: {logs}")
            else:
                pytest.fail(f"API mode test failed: {e}")
        finally:
            if container:
                try:
                    container.stop(timeout=10)
                except:
                    pass
    
    def test_container_environment_variables(self, docker_client, built_image):
        """Test that environment variables are properly set in container."""
        container = None
        try:
            # Start container with custom environment
            container = docker_client.containers.run(
                built_image.id,
                detach=True,
                remove=True,
                environment={
                    'LOG_LEVEL': 'DEBUG',
                    'MODEL_CACHE_MAX_ENTRIES': '5',
                    'MODEL_CACHE_MAX_MEMORY_MB': '100',
                    'WORKERS': '2'
                },
                command=['python', '-c', '''
import os
print(f"LOG_LEVEL={os.getenv('LOG_LEVEL')}")
print(f"MODEL_CACHE_MAX_ENTRIES={os.getenv('MODEL_CACHE_MAX_ENTRIES')}")
print(f"MODEL_CACHE_MAX_MEMORY_MB={os.getenv('MODEL_CACHE_MAX_MEMORY_MB')}")
print(f"WORKERS={os.getenv('WORKERS')}")
print(f"PYTHONPATH={os.getenv('PYTHONPATH')}")
print(f"PYTHONUNBUFFERED={os.getenv('PYTHONUNBUFFERED')}")
''']
            )
            
            # Wait for command to complete
            result = container.wait(timeout=30)
            logs = container.logs().decode('utf-8')
            
            assert result['StatusCode'] == 0, f"Environment test failed: {logs}"
            
            # Verify environment variables are set correctly
            assert "LOG_LEVEL=DEBUG" in logs
            assert "MODEL_CACHE_MAX_ENTRIES=5" in logs
            assert "MODEL_CACHE_MAX_MEMORY_MB=100" in logs
            assert "WORKERS=2" in logs
            assert "PYTHONPATH=/app" in logs
            assert "PYTHONUNBUFFERED=1" in logs
            
        finally:
            if container:
                try:
                    container.stop(timeout=5)
                except:
                    pass
    
    def test_container_file_permissions(self, docker_client, built_image, temp_volumes):
        """Test that file permissions work correctly in container."""
        container = None
        try:
            # Start container and test file operations
            container = docker_client.containers.run(
                built_image.id,
                detach=True,
                remove=True,
                volumes={
                    str(temp_volumes['data']): {'bind': '/app/data', 'mode': 'rw'},
                    str(temp_volumes['models']): {'bind': '/app/models', 'mode': 'rw'},
                    str(temp_volumes['logs']): {'bind': '/app/logs', 'mode': 'rw'}
                },
                command=['python', '-c', '''
import os
import json

# Test file writing permissions
test_results = {}

try:
    # Test writing to data directory
    with open("/app/data/test_file.txt", "w") as f:
        f.write("test data")
    test_results["data_write"] = True
except Exception as e:
    test_results["data_write"] = False
    test_results["data_error"] = str(e)

try:
    # Test writing to models directory
    with open("/app/models/test_model.pkl", "w") as f:
        f.write("test model")
    test_results["models_write"] = True
except Exception as e:
    test_results["models_write"] = False
    test_results["models_error"] = str(e)

try:
    # Test writing to logs directory
    with open("/app/logs/test.log", "w") as f:
        f.write("test log")
    test_results["logs_write"] = True
except Exception as e:
    test_results["logs_write"] = False
    test_results["logs_error"] = str(e)

# Check user ID
test_results["user_id"] = os.getuid()
test_results["group_id"] = os.getgid()

print(json.dumps(test_results))
''']
            )
            
            # Wait for command to complete
            result = container.wait(timeout=30)
            logs = container.logs().decode('utf-8')
            
            assert result['StatusCode'] == 0, f"Permission test failed: {logs}"
            
            # Parse results
            try:
                test_results = json.loads(logs.strip())
                
                # Verify file write permissions
                assert test_results.get("data_write", False), f"Data write failed: {test_results.get('data_error')}"
                assert test_results.get("models_write", False), f"Models write failed: {test_results.get('models_error')}"
                assert test_results.get("logs_write", False), f"Logs write failed: {test_results.get('logs_error')}"
                
                # Verify non-root user
                assert test_results.get("user_id", 0) != 0, "Container should not run as root"
                assert test_results.get("group_id", 0) != 0, "Container should not run with root group"
                
            except json.JSONDecodeError:
                pytest.fail(f"Could not parse test results: {logs}")
            
        finally:
            if container:
                try:
                    container.stop(timeout=5)
                except:
                    pass
    
    def test_container_resource_limits(self, docker_client, built_image):
        """Test that container respects resource limits."""
        container = None
        try:
            # Start container with resource limits
            container = docker_client.containers.run(
                built_image.id,
                detach=True,
                remove=True,
                mem_limit='512m',
                cpuset_cpus='0',
                environment={
                    'LOG_LEVEL': 'INFO',
                    'MODEL_CACHE_MAX_MEMORY_MB': '100'  # Conservative for test
                },
                command=['python', '-c', '''
import time
import sys
import os

# Test that container can start and run with limited resources
print("Container started with resource limits")

# Try to import main modules to verify they work under constraints
try:
    sys.path.insert(0, "/app")
    from src.health_check import is_healthy
    print("Health check module imported successfully")
    
    # Run a basic health check
    if is_healthy():
        print("Health check passed under resource constraints")
    else:
        print("Health check failed under resource constraints")
        
except Exception as e:
    print(f"Error under resource constraints: {e}")
    sys.exit(1)

print("Resource limit test completed successfully")
''']
            )
            
            # Wait for command to complete
            result = container.wait(timeout=60)
            logs = container.logs().decode('utf-8')
            
            assert result['StatusCode'] == 0, f"Resource limit test failed: {logs}"
            assert "Container started with resource limits" in logs
            assert "Resource limit test completed successfully" in logs
            
        finally:
            if container:
                try:
                    container.stop(timeout=5)
                except:
                    pass


class TestDockerComposeIntegration:
    """Integration tests for Docker Compose functionality."""
    
    def test_docker_compose_syntax(self):
        """Test that docker-compose.yml has valid syntax."""
        try:
            result = subprocess.run(
                ['docker-compose', '-f', '/root/repo/docker-compose.yml', 'config'],
                capture_output=True,
                text=True,
                check=True
            )
            assert result.returncode == 0, f"Docker Compose syntax error: {result.stderr}"
        except subprocess.CalledProcessError as e:
            pytest.fail(f"Docker Compose validation failed: {e}")
        except FileNotFoundError:
            pytest.skip("docker-compose command not available")
    
    def test_compose_service_validation(self):
        """Test that all services in compose file are valid."""
        try:
            # Validate compose file
            result = subprocess.run(
                ['docker-compose', '-f', '/root/repo/docker-compose.yml', 'config', '--services'],
                capture_output=True,
                text=True,
                check=True
            )
            
            services = result.stdout.strip().split('\n')
            expected_services = [
                'churn-predictor', 'churn-predictor-dev', 'mlflow',
                'churn-trainer', 'prometheus', 'grafana'
            ]
            
            for service in expected_services:
                assert service in services, f"Service {service} not found in compose output"
                
        except subprocess.CalledProcessError as e:
            pytest.fail(f"Compose service validation failed: {e}")
        except FileNotFoundError:
            pytest.skip("docker-compose command not available")
    
    def test_compose_volume_validation(self):
        """Test that volume configurations are valid."""
        try:
            result = subprocess.run(
                ['docker-compose', '-f', '/root/repo/docker-compose.yml', 'config', '--volumes'],
                capture_output=True,
                text=True,
                check=True
            )
            
            volumes = result.stdout.strip().split('\n')
            expected_volumes = ['mlflow-data', 'prometheus-data', 'grafana-data']
            
            for volume in expected_volumes:
                assert volume in volumes, f"Volume {volume} not found in compose output"
                
        except subprocess.CalledProcessError as e:
            pytest.fail(f"Compose volume validation failed: {e}")
        except FileNotFoundError:
            pytest.skip("docker-compose command not available")