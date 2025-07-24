"""
Tests for Docker security hardening functionality.

This module tests security measures implemented for container deployment
including vulnerability scanning, security policies, and runtime hardening.
"""

import unittest
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from src.security import (
    SecurityScanner,
    get_security_report,
    check_container_security,
    validate_image_signature,
    get_security_policies,
    SecurityVulnerability,
    SecurityScanResult
)


class TestSecurityVulnerability(unittest.TestCase):
    """Test SecurityVulnerability dataclass."""
    
    def test_vulnerability_creation(self):
        """Test vulnerability object creation."""
        vuln = SecurityVulnerability(
            id="CVE-2023-1234",
            severity="HIGH",
            package="test-package",
            version="1.0.0",
            fixed_version="1.0.1",
            description="Test vulnerability"
        )
        
        self.assertEqual(vuln.id, "CVE-2023-1234")
        self.assertEqual(vuln.severity, "HIGH")
        self.assertEqual(vuln.package, "test-package")
        self.assertEqual(vuln.version, "1.0.0")
        self.assertEqual(vuln.fixed_version, "1.0.1")
        self.assertEqual(vuln.description, "Test vulnerability")
    
    def test_vulnerability_to_dict(self):
        """Test vulnerability serialization."""
        vuln = SecurityVulnerability(
            id="CVE-2023-1234",
            severity="MEDIUM",
            package="example",
            version="2.0.0",
            fixed_version="2.1.0",
            description="Example vulnerability"
        )
        
        result = vuln.to_dict()
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result["id"], "CVE-2023-1234")
        self.assertEqual(result["severity"], "MEDIUM")
        self.assertEqual(result["package"], "example")


class TestSecurityScanResult(unittest.TestCase):
    """Test SecurityScanResult functionality."""
    
    def test_scan_result_creation(self):
        """Test scan result creation."""
        vulnerabilities = [
            SecurityVulnerability("CVE-1", "HIGH", "pkg1", "1.0", "1.1", "Test 1"),
            SecurityVulnerability("CVE-2", "MEDIUM", "pkg2", "2.0", "2.1", "Test 2")
        ]
        
        result = SecurityScanResult(
            scan_time="2023-01-01T12:00:00Z",
            image="test:latest",
            vulnerabilities=vulnerabilities,
            total_vulnerabilities=2,
            high_severity_count=1,
            medium_severity_count=1,
            low_severity_count=0
        )
        
        self.assertEqual(result.total_vulnerabilities, 2)
        self.assertEqual(result.high_severity_count, 1)
        self.assertEqual(result.medium_severity_count, 1)
        self.assertEqual(len(result.vulnerabilities), 2)
    
    def test_scan_result_is_secure(self):
        """Test security assessment logic."""
        # No vulnerabilities - secure
        result_secure = SecurityScanResult(
            scan_time="2023-01-01T12:00:00Z",
            image="secure:latest",
            vulnerabilities=[],
            total_vulnerabilities=0,
            high_severity_count=0,
            medium_severity_count=0,
            low_severity_count=0
        )
        
        self.assertTrue(result_secure.is_secure())
        
        # High severity vulnerabilities - not secure
        result_insecure = SecurityScanResult(
            scan_time="2023-01-01T12:00:00Z",
            image="insecure:latest",
            vulnerabilities=[
                SecurityVulnerability("CVE-1", "HIGH", "pkg", "1.0", "1.1", "High severity")
            ],
            total_vulnerabilities=1,
            high_severity_count=1,
            medium_severity_count=0,
            low_severity_count=0
        )
        
        self.assertFalse(result_insecure.is_secure())
    
    def test_scan_result_to_dict(self):
        """Test scan result serialization."""
        vulnerabilities = [
            SecurityVulnerability("CVE-1", "HIGH", "pkg1", "1.0", "1.1", "Test")
        ]
        
        result = SecurityScanResult(
            scan_time="2023-01-01T12:00:00Z",
            image="test:latest",
            vulnerabilities=vulnerabilities,
            total_vulnerabilities=1,
            high_severity_count=1,
            medium_severity_count=0,
            low_severity_count=0
        )
        
        result_dict = result.to_dict()
        
        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict["total_vulnerabilities"], 1)
        self.assertEqual(result_dict["high_severity_count"], 1)
        self.assertIn("vulnerabilities", result_dict)


class TestSecurityScanner(unittest.TestCase):
    """Test SecurityScanner class functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scanner = SecurityScanner()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_scanner_initialization(self):
        """Test scanner initialization."""
        self.assertIsNotNone(self.scanner.trivy_path)
        self.assertIsNotNone(self.scanner.cosign_path)
        self.assertEqual(self.scanner.scan_timeout, 300)
    
    @patch('subprocess.run')
    def test_scan_image_success(self, mock_run):
        """Test successful image scanning."""
        # Mock Trivy output
        mock_trivy_output = {
            "SchemaVersion": 2,
            "ArtifactName": "test:latest",
            "ArtifactType": "container_image",
            "Results": [
                {
                    "Target": "test:latest",
                    "Class": "os-pkgs",
                    "Type": "debian",
                    "Vulnerabilities": [
                        {
                            "VulnerabilityID": "CVE-2023-1234",
                            "Severity": "HIGH",
                            "PkgName": "libssl",
                            "InstalledVersion": "1.0.0",
                            "FixedVersion": "1.0.1",
                            "Description": "SSL vulnerability"
                        }
                    ]
                }
            ]
        }
        
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(mock_trivy_output)
        )
        
        result = self.scanner.scan_image("test:latest")
        
        self.assertIsInstance(result, SecurityScanResult)
        self.assertEqual(result.total_vulnerabilities, 1)
        self.assertEqual(result.high_severity_count, 1)
        self.assertEqual(result.image, "test:latest")
        
        # Verify Trivy was called correctly
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        self.assertIn("trivy", call_args[0])
        self.assertIn("test:latest", call_args)
    
    @patch('subprocess.run')
    def test_scan_image_no_vulnerabilities(self, mock_run):
        """Test scanning image with no vulnerabilities."""
        mock_trivy_output = {
            "SchemaVersion": 2,
            "ArtifactName": "secure:latest",
            "ArtifactType": "container_image",
            "Results": []
        }
        
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(mock_trivy_output)
        )
        
        result = self.scanner.scan_image("secure:latest")
        
        self.assertEqual(result.total_vulnerabilities, 0)
        self.assertEqual(result.high_severity_count, 0)
        self.assertTrue(result.is_secure())
    
    @patch('subprocess.run')
    def test_scan_image_trivy_error(self, mock_run):
        """Test handling Trivy scan errors."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stderr="Trivy scan failed"
        )
        
        with self.assertRaises(Exception) as context:
            self.scanner.scan_image("error:latest")
        
        self.assertIn("Trivy scan failed", str(context.exception))
    
    @patch('subprocess.run')
    def test_verify_image_signature_success(self, mock_run):
        """Test successful image signature verification."""
        mock_run.return_value = MagicMock(returncode=0)
        
        result = self.scanner.verify_image_signature("signed:latest")
        
        self.assertTrue(result)
        mock_run.assert_called_once()
    
    @patch('subprocess.run')
    def test_verify_image_signature_failure(self, mock_run):
        """Test image signature verification failure."""
        mock_run.return_value = MagicMock(returncode=1)
        
        result = self.scanner.verify_image_signature("unsigned:latest")
        
        self.assertFalse(result)
    
    def test_generate_security_policies(self):
        """Test security policy generation."""
        policies = self.scanner.generate_security_policies()
        
        self.assertIsInstance(policies, dict)
        self.assertIn("container_policies", policies)
        self.assertIn("network_policies", policies)
        self.assertIn("runtime_policies", policies)
        
        # Check container policies
        container_policies = policies["container_policies"]
        self.assertIn("run_as_non_root", container_policies)
        self.assertIn("read_only_root_filesystem", container_policies)
        self.assertIn("no_privileged_containers", container_policies)
    
    def test_check_dockerfile_security(self):
        """Test Dockerfile security analysis."""
        # Create a test Dockerfile with security issues
        dockerfile_content = """
FROM ubuntu:latest
USER root
RUN apt-get update && apt-get install -y curl
COPY . /app
WORKDIR /app
EXPOSE 8080
CMD ["python", "app.py"]
"""
        
        dockerfile_path = Path(self.temp_dir) / "Dockerfile"
        dockerfile_path.write_text(dockerfile_content)
        
        issues = self.scanner.check_dockerfile_security(dockerfile_path)
        
        self.assertIsInstance(issues, list)
        # Should find security issues like using latest tag, running as root
        issue_types = [issue["type"] for issue in issues]
        self.assertIn("latest_tag", issue_types)
        self.assertIn("root_user", issue_types)


class TestSecurityFunctions(unittest.TestCase):
    """Test module-level security functions."""
    
    @patch('src.security.SecurityScanner.scan_image')
    def test_get_security_report(self, mock_scan):
        """Test security report generation."""
        mock_result = SecurityScanResult(
            scan_time="2023-01-01T12:00:00Z",
            image="test:latest",
            vulnerabilities=[],
            total_vulnerabilities=0,
            high_severity_count=0,
            medium_severity_count=0,
            low_severity_count=0
        )
        mock_scan.return_value = mock_result
        
        report = get_security_report("test:latest")
        
        self.assertIsInstance(report, dict)
        self.assertIn("scan_result", report)
        self.assertIn("security_score", report)
        self.assertIn("recommendations", report)
    
    @patch('src.security.SecurityScanner.scan_image')
    def test_check_container_security(self, mock_scan):
        """Test container security check."""
        mock_result = SecurityScanResult(
            scan_time="2023-01-01T12:00:00Z",
            image="test:latest",
            vulnerabilities=[],
            total_vulnerabilities=0,
            high_severity_count=0,
            medium_severity_count=0,
            low_severity_count=0
        )
        mock_scan.return_value = mock_result
        
        is_secure = check_container_security("test:latest")
        
        self.assertTrue(is_secure)
    
    @patch('src.security.SecurityScanner.verify_image_signature')
    def test_validate_image_signature(self, mock_verify):
        """Test image signature validation."""
        mock_verify.return_value = True
        
        result = validate_image_signature("signed:latest")
        
        self.assertTrue(result)
        mock_verify.assert_called_once_with("signed:latest")
    
    def test_get_security_policies(self):
        """Test security policies retrieval."""
        policies = get_security_policies()
        
        self.assertIsInstance(policies, dict)
        self.assertIn("container_policies", policies)


class TestSecurityIntegration(unittest.TestCase):
    """Integration tests for security system."""
    
    def test_security_workflow(self):
        """Test complete security workflow."""
        scanner = SecurityScanner()
        
        # Generate policies
        policies = scanner.generate_security_policies()
        self.assertIsInstance(policies, dict)
        
        # Check Dockerfile security (if exists)
        dockerfile_path = Path("Dockerfile")
        if dockerfile_path.exists():
            issues = scanner.check_dockerfile_security(dockerfile_path)
            self.assertIsInstance(issues, list)
    
    @patch('subprocess.run')
    def test_end_to_end_security_check(self, mock_run):
        """Test end-to-end security checking."""
        # Mock successful Trivy scan
        mock_trivy_output = {
            "SchemaVersion": 2,
            "ArtifactName": "churn-predictor:latest",
            "ArtifactType": "container_image",
            "Results": []
        }
        
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(mock_trivy_output)
        )
        
        # Test complete security check
        report = get_security_report("churn-predictor:latest")
        
        self.assertIsInstance(report, dict)
        self.assertIn("scan_result", report)
        self.assertTrue(report["scan_result"]["is_secure"])


if __name__ == "__main__":
    unittest.main()