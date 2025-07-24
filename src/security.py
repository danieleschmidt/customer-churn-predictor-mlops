"""
Docker security hardening and vulnerability scanning functionality.

This module provides comprehensive security measures for container deployment
including vulnerability scanning, image verification, security policies,
and runtime hardening recommendations.
"""

import subprocess
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SecurityVulnerability:
    """
    Represents a security vulnerability found during scanning.
    
    Attributes:
        id: Vulnerability identifier (e.g., CVE-2023-1234)
        severity: Severity level (CRITICAL, HIGH, MEDIUM, LOW)
        package: Affected package name
        version: Current package version
        fixed_version: Version that fixes the vulnerability
        description: Vulnerability description
    """
    id: str
    severity: str
    package: str
    version: str
    fixed_version: Optional[str]
    description: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert vulnerability to dictionary."""
        return asdict(self)


@dataclass
class SecurityScanResult:
    """
    Results from a security vulnerability scan.
    
    Attributes:
        scan_time: When the scan was performed
        image: Docker image that was scanned
        vulnerabilities: List of found vulnerabilities
        total_vulnerabilities: Total number of vulnerabilities
        high_severity_count: Number of high severity vulnerabilities
        medium_severity_count: Number of medium severity vulnerabilities
        low_severity_count: Number of low severity vulnerabilities
    """
    scan_time: str
    image: str
    vulnerabilities: List[SecurityVulnerability]
    total_vulnerabilities: int
    high_severity_count: int
    medium_severity_count: int
    low_severity_count: int
    
    def is_secure(self, max_high: int = 0, max_medium: int = 5) -> bool:
        """
        Determine if the scan result indicates a secure image.
        
        Args:
            max_high: Maximum allowed high severity vulnerabilities
            max_medium: Maximum allowed medium severity vulnerabilities
            
        Returns:
            True if image meets security criteria
        """
        return (self.high_severity_count <= max_high and 
                self.medium_severity_count <= max_medium)
    
    def get_security_score(self) -> float:
        """
        Calculate security score (0-100, higher is better).
        
        Returns:
            Security score based on vulnerability counts and severity
        """
        if self.total_vulnerabilities == 0:
            return 100.0
        
        # Weight vulnerabilities by severity
        weighted_score = (
            self.high_severity_count * 10 +
            self.medium_severity_count * 5 +
            self.low_severity_count * 1
        )
        
        # Convert to 0-100 scale (lower weighted score = higher security)
        max_possible_score = self.total_vulnerabilities * 10  # All high severity
        if max_possible_score == 0:
            return 100.0
        
        security_score = max(0, 100 - (weighted_score / max_possible_score * 100))
        return round(security_score, 2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert scan result to dictionary."""
        result = asdict(self)
        result["vulnerabilities"] = [vuln.to_dict() for vuln in self.vulnerabilities]
        result["is_secure"] = self.is_secure()
        result["security_score"] = self.get_security_score()
        return result


class SecurityScanner:
    """
    Comprehensive security scanner for Docker images and configurations.
    
    Provides vulnerability scanning using Trivy, image signature verification
    using Cosign, security policy generation, and Dockerfile security analysis.
    """
    
    def __init__(self, trivy_path: str = "trivy", cosign_path: str = "cosign"):
        """
        Initialize security scanner.
        
        Args:
            trivy_path: Path to Trivy binary
            cosign_path: Path to Cosign binary
        """
        self.trivy_path = trivy_path
        self.cosign_path = cosign_path
        self.scan_timeout = 300  # 5 minutes
        
        logger.info("SecurityScanner initialized")
    
    def scan_image(self, image: str, output_format: str = "json") -> SecurityScanResult:
        """
        Scan Docker image for vulnerabilities using Trivy.
        
        Args:
            image: Docker image to scan (e.g., "nginx:latest")
            output_format: Output format (json, table, sarif)
            
        Returns:
            SecurityScanResult containing vulnerability information
            
        Raises:
            Exception: If scan fails or Trivy is not available
        """
        logger.info(f"Starting security scan of image: {image}")
        
        # Prepare Trivy command
        cmd = [
            self.trivy_path,
            "image",
            "--format", output_format,
            "--quiet",
            "--no-progress",
            image
        ]
        
        try:
            # Run Trivy scan
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.scan_timeout,
                check=False
            )
            
            if result.returncode != 0:
                error_msg = f"Trivy scan failed for {image}: {result.stderr}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            # Parse JSON output
            scan_data = json.loads(result.stdout)
            
            # Extract vulnerabilities
            vulnerabilities = []
            total_count = 0
            high_count = 0
            medium_count = 0
            low_count = 0
            
            # Process Trivy results
            for result_item in scan_data.get("Results", []):
                for vuln in result_item.get("Vulnerabilities", []):
                    vulnerability = SecurityVulnerability(
                        id=vuln.get("VulnerabilityID", ""),
                        severity=vuln.get("Severity", "UNKNOWN"),
                        package=vuln.get("PkgName", ""),
                        version=vuln.get("InstalledVersion", ""),
                        fixed_version=vuln.get("FixedVersion"),
                        description=vuln.get("Description", "")
                    )
                    vulnerabilities.append(vulnerability)
                    total_count += 1
                    
                    # Count by severity
                    severity = vulnerability.severity.upper()
                    if severity in ["CRITICAL", "HIGH"]:
                        high_count += 1
                    elif severity == "MEDIUM":
                        medium_count += 1
                    else:
                        low_count += 1
            
            scan_result = SecurityScanResult(
                scan_time=datetime.utcnow().isoformat(),
                image=image,
                vulnerabilities=vulnerabilities,
                total_vulnerabilities=total_count,
                high_severity_count=high_count,
                medium_severity_count=medium_count,
                low_severity_count=low_count
            )
            
            logger.info(f"Security scan completed for {image}: "
                       f"{total_count} vulnerabilities found "
                       f"({high_count} high, {medium_count} medium, {low_count} low)")
            
            return scan_result
            
        except subprocess.TimeoutExpired:
            error_msg = f"Trivy scan timed out after {self.scan_timeout} seconds"
            logger.error(error_msg)
            raise Exception(error_msg)
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse Trivy output: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            logger.error(f"Security scan failed: {e}")
            raise
    
    def verify_image_signature(self, image: str) -> bool:
        """
        Verify Docker image signature using Cosign.
        
        Args:
            image: Docker image to verify
            
        Returns:
            True if signature is valid, False otherwise
        """
        logger.info(f"Verifying signature for image: {image}")
        
        try:
            cmd = [
                self.cosign_path,
                "verify",
                image
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                check=False
            )
            
            if result.returncode == 0:
                logger.info(f"Image signature verified successfully: {image}")
                return True
            else:
                logger.warning(f"Image signature verification failed: {image}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Signature verification timed out")
            return False
        except Exception as e:
            logger.error(f"Signature verification error: {e}")
            return False
    
    def generate_security_policies(self) -> Dict[str, Any]:
        """
        Generate security policies for container deployment.
        
        Returns:
            Dictionary containing security policies and recommendations
        """
        policies = {
            "container_policies": {
                "run_as_non_root": {
                    "enabled": True,
                    "description": "Containers must run as non-root user",
                    "enforcement": "required"
                },
                "read_only_root_filesystem": {
                    "enabled": True,
                    "description": "Root filesystem should be read-only",
                    "enforcement": "recommended"
                },
                "no_privileged_containers": {
                    "enabled": True,
                    "description": "Privileged containers are not allowed",
                    "enforcement": "required"
                },
                "drop_all_capabilities": {
                    "enabled": True,
                    "description": "Drop all Linux capabilities by default",
                    "enforcement": "recommended"
                },
                "no_host_network": {
                    "enabled": True,
                    "description": "Containers cannot use host network",
                    "enforcement": "required"
                },
                "resource_limits": {
                    "enabled": True,
                    "description": "All containers must have resource limits",
                    "enforcement": "required",
                    "limits": {
                        "memory": "1Gi",
                        "cpu": "1000m"
                    }
                }
            },
            "network_policies": {
                "default_deny": {
                    "enabled": True,
                    "description": "Default deny all network traffic",
                    "enforcement": "required"
                },
                "ingress_whitelist": {
                    "enabled": True,
                    "description": "Only allow whitelisted ingress traffic",
                    "allowed_ports": [8000, 8080]
                },
                "egress_restrictions": {
                    "enabled": True,
                    "description": "Restrict egress to necessary services only"
                }
            },
            "runtime_policies": {
                "seccomp_profile": {
                    "enabled": True,
                    "profile": "runtime/default",
                    "description": "Use secure computing mode profile"
                },
                "apparmor_profile": {
                    "enabled": True,
                    "profile": "docker-default",
                    "description": "Use AppArmor security profile"
                },
                "selinux_options": {
                    "enabled": False,
                    "description": "SELinux security context options"
                }
            },
            "image_policies": {
                "vulnerability_scanning": {
                    "enabled": True,
                    "max_high_severity": 0,
                    "max_medium_severity": 5,
                    "description": "Images must pass vulnerability scanning"
                },
                "signature_verification": {
                    "enabled": True,
                    "description": "Images must be signed and verified"
                },
                "base_image_restrictions": {
                    "enabled": True,
                    "allowed_registries": [
                        "docker.io/library",
                        "gcr.io",
                        "quay.io"
                    ],
                    "prohibited_tags": ["latest", "master", "main"]
                }
            }
        }
        
        return policies
    
    def check_dockerfile_security(self, dockerfile_path: Path) -> List[Dict[str, Any]]:
        """
        Analyze Dockerfile for security issues.
        
        Args:
            dockerfile_path: Path to Dockerfile
            
        Returns:
            List of security issues found
        """
        logger.info(f"Analyzing Dockerfile security: {dockerfile_path}")
        
        if not dockerfile_path.exists():
            return [{"type": "file_not_found", "message": "Dockerfile not found"}]
        
        issues = []
        
        try:
            content = dockerfile_path.read_text()
            lines = content.split('\n')
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip().upper()
                
                # Check for latest tag usage
                if line.startswith('FROM') and ':LATEST' in line:
                    issues.append({
                        "type": "latest_tag",
                        "line": line_num,
                        "severity": "MEDIUM",
                        "message": "Using 'latest' tag is not recommended for production",
                        "recommendation": "Use specific version tags"
                    })
                
                # Check for running as root
                if line.startswith('USER ROOT') or (line.startswith('USER') and '0' in line):
                    issues.append({
                        "type": "root_user",
                        "line": line_num,
                        "severity": "HIGH",
                        "message": "Running as root user",
                        "recommendation": "Create and use non-root user"
                    })
                
                # Check for ADD usage (prefer COPY)
                if line.startswith('ADD ') and not line.startswith('ADD --'):
                    issues.append({
                        "type": "add_usage",
                        "line": line_num,
                        "severity": "LOW",
                        "message": "ADD instruction can introduce security risks",
                        "recommendation": "Use COPY instead of ADD when possible"
                    })
                
                # Check for curl/wget without verification
                if 'CURL' in line or 'WGET' in line:
                    if '-K' not in line and '--INSECURE' in line:
                        issues.append({
                            "type": "insecure_download",
                            "line": line_num,
                            "severity": "HIGH",
                            "message": "Insecure download without certificate verification",
                            "recommendation": "Remove --insecure flag and verify certificates"
                        })
                
                # Check for exposed privileged ports
                if line.startswith('EXPOSE'):
                    ports = [p.strip() for p in line.replace('EXPOSE', '').split()]
                    for port in ports:
                        if port.isdigit() and int(port) < 1024:
                            issues.append({
                                "type": "privileged_port",
                                "line": line_num,
                                "severity": "MEDIUM",
                                "message": f"Exposing privileged port {port}",
                                "recommendation": "Use non-privileged ports (>1024)"
                            })
            
            logger.info(f"Dockerfile analysis completed: {len(issues)} issues found")
            return issues
            
        except Exception as e:
            logger.error(f"Failed to analyze Dockerfile: {e}")
            return [{"type": "analysis_error", "message": str(e)}]
    
    def generate_security_report(self, image: str) -> Dict[str, Any]:
        """
        Generate comprehensive security report for an image.
        
        Args:
            image: Docker image to analyze
            
        Returns:
            Comprehensive security report
        """
        logger.info(f"Generating security report for: {image}")
        
        report = {
            "image": image,
            "timestamp": datetime.utcnow().isoformat(),
            "scan_result": None,
            "signature_verified": False,
            "dockerfile_issues": [],
            "security_policies": self.generate_security_policies(),
            "recommendations": []
        }
        
        try:
            # Vulnerability scan
            scan_result = self.scan_image(image)
            report["scan_result"] = scan_result.to_dict()
            
            # Signature verification
            report["signature_verified"] = self.verify_image_signature(image)
            
            # Dockerfile analysis (if available)
            dockerfile_path = Path("Dockerfile")
            if dockerfile_path.exists():
                report["dockerfile_issues"] = self.check_dockerfile_security(dockerfile_path)
            
            # Generate recommendations
            recommendations = []
            
            if not scan_result.is_secure():
                recommendations.append("Address high and medium severity vulnerabilities")
            
            if not report["signature_verified"]:
                recommendations.append("Sign and verify image signatures")
            
            if report["dockerfile_issues"]:
                recommendations.append("Fix Dockerfile security issues")
            
            recommendations.extend([
                "Run containers as non-root user",
                "Use read-only root filesystem",
                "Apply resource limits",
                "Use security profiles (AppArmor/SELinux)",
                "Implement network policies",
                "Regular security scanning in CI/CD"
            ])
            
            report["recommendations"] = recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate security report: {e}")
            report["error"] = str(e)
        
        return report


# Global security scanner instance
_global_scanner: Optional[SecurityScanner] = None


def get_security_scanner() -> SecurityScanner:
    """
    Get global security scanner instance.
    
    Returns:
        Global SecurityScanner instance
    """
    global _global_scanner
    
    if _global_scanner is None:
        _global_scanner = SecurityScanner()
    
    return _global_scanner


def get_security_report(image: str) -> Dict[str, Any]:
    """
    Get comprehensive security report for an image.
    
    Args:
        image: Docker image to analyze
        
    Returns:
        Security report dictionary
    """
    scanner = get_security_scanner()
    return scanner.generate_security_report(image)


def check_container_security(image: str, max_high: int = 0, max_medium: int = 5) -> bool:
    """
    Check if container image meets security requirements.
    
    Args:
        image: Docker image to check
        max_high: Maximum allowed high severity vulnerabilities
        max_medium: Maximum allowed medium severity vulnerabilities
        
    Returns:
        True if image meets security criteria
    """
    try:
        scanner = get_security_scanner()
        scan_result = scanner.scan_image(image)
        return scan_result.is_secure(max_high, max_medium)
    except Exception as e:
        logger.error(f"Security check failed: {e}")
        return False


def validate_image_signature(image: str) -> bool:
    """
    Validate Docker image signature.
    
    Args:
        image: Docker image to validate
        
    Returns:
        True if signature is valid
    """
    scanner = get_security_scanner()
    return scanner.verify_image_signature(image)


def get_security_policies() -> Dict[str, Any]:
    """
    Get security policies for container deployment.
    
    Returns:
        Security policies dictionary
    """
    scanner = get_security_scanner()
    return scanner.generate_security_policies()