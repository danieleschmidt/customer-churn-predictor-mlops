"""
Tests for advanced security framework.
"""

import pytest
import json
import time
import tempfile
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from src.advanced_security import (
    AdvancedEncryption, ThreatDetectionSystem, ComplianceValidator,
    VulnerabilityScanner, AdvancedSecurityOrchestrator, create_security_report
)


class TestAdvancedEncryption:
    """Test encryption functionality."""
    
    def test_encryption_initialization(self):
        """Test encryption system initialization."""
        encryption = AdvancedEncryption()
        assert encryption.password is not None
        assert encryption.fernet is not None
    
    def test_encrypt_decrypt_string(self):
        """Test string encryption and decryption."""
        encryption = AdvancedEncryption("test_password")
        
        original_data = "sensitive information"
        encrypted = encryption.encrypt_data(original_data)
        decrypted = encryption.decrypt_data(encrypted)
        
        assert encrypted != original_data
        assert decrypted == original_data
    
    def test_encrypt_decrypt_dict(self):
        """Test dictionary encryption and decryption."""
        encryption = AdvancedEncryption("test_password")
        
        original_data = {"user": "john", "balance": 1000}
        encrypted = encryption.encrypt_data(original_data)
        decrypted = encryption.decrypt_data(encrypted)
        
        assert encrypted != str(original_data)
        assert decrypted == original_data
    
    def test_generate_secure_token(self):
        """Test secure token generation."""
        encryption = AdvancedEncryption()
        
        token1 = encryption.generate_secure_token()
        token2 = encryption.generate_secure_token()
        
        assert len(token1) > 0
        assert len(token2) > 0
        assert token1 != token2  # Should be unique
    
    def test_hash_sensitive_data(self):
        """Test sensitive data hashing."""
        encryption = AdvancedEncryption()
        
        data = "password123"
        hashed1, salt1 = encryption.hash_sensitive_data(data)
        hashed2, salt2 = encryption.hash_sensitive_data(data)
        
        assert hashed1 != data
        assert hashed2 != data
        assert hashed1 != hashed2  # Different salts should produce different hashes
        assert salt1 != salt2


class TestThreatDetectionSystem:
    """Test threat detection functionality."""
    
    def test_threat_detector_initialization(self):
        """Test threat detector initialization."""
        detector = ThreatDetectionSystem()
        assert len(detector.threat_patterns) > 0
        assert 'sql_injection' in detector.threat_patterns
        assert 'xss' in detector.threat_patterns
    
    def test_detect_sql_injection(self):
        """Test SQL injection detection."""
        detector = ThreatDetectionSystem()
        
        request_data = {
            'method': 'POST',
            'path': '/login',
            'headers': {'user-agent': 'Mozilla/5.0'},
            'params': {'username': 'admin', 'password': "' OR '1'='1"},
            'source_ip': '192.168.1.100'
        }
        
        threats = detector.analyze_request(request_data)
        
        # Should detect SQL injection
        sql_threats = [t for t in threats if t.category == 'sql_injection']
        assert len(sql_threats) > 0
        assert any(t.severity in ['high', 'critical'] for t in sql_threats)
    
    def test_detect_xss(self):
        """Test XSS detection."""
        detector = ThreatDetectionSystem()
        
        request_data = {
            'method': 'POST',
            'path': '/comment',
            'headers': {'user-agent': 'Mozilla/5.0'},
            'params': {'comment': '<script>alert("XSS")</script>'},
            'source_ip': '10.0.0.1'
        }
        
        threats = detector.analyze_request(request_data)
        
        # Should detect XSS
        xss_threats = [t for t in threats if t.category == 'xss']
        assert len(xss_threats) > 0
    
    def test_detect_suspicious_user_agent(self):
        """Test suspicious user agent detection."""
        detector = ThreatDetectionSystem()
        
        request_data = {
            'method': 'GET',
            'path': '/',
            'headers': {'user-agent': 'sqlmap/1.4'},
            'params': {},
            'source_ip': '192.168.1.1'
        }
        
        threats = detector.analyze_request(request_data)
        
        # Should detect malicious user agent
        ua_threats = [t for t in threats if t.category == 'malicious_user_agent']
        assert len(ua_threats) > 0
    
    def test_clean_request(self):
        """Test clean request produces no threats."""
        detector = ThreatDetectionSystem()
        
        request_data = {
            'method': 'GET',
            'path': '/api/health',
            'headers': {'user-agent': 'Mozilla/5.0 (legitimate browser)'},
            'params': {'check': 'status'},
            'source_ip': '192.168.1.100'
        }
        
        threats = detector.analyze_request(request_data)
        
        # Should have minimal or no threats for clean request
        high_severity_threats = [t for t in threats if t.severity in ['high', 'critical']]
        assert len(high_severity_threats) == 0


class TestComplianceValidator:
    """Test compliance validation."""
    
    def test_validator_initialization(self):
        """Test compliance validator initialization."""
        validator = ComplianceValidator()
        assert 'GDPR' in validator.regulations
        assert 'HIPAA' in validator.regulations
        assert 'SOC2' in validator.regulations
    
    def test_gdpr_compliance_checks(self):
        """Test GDPR compliance validation."""
        validator = ComplianceValidator()
        
        config = {
            'encryption_at_rest': True,
            'data_deletion_api': True,
            'audit_logging': True
        }
        
        checks = validator.validate_compliance('GDPR', config)
        
        assert len(checks) > 0
        assert all(check.regulation == 'GDPR' for check in checks)
        
        # Should have compliant checks for good config
        compliant_checks = [c for c in checks if c.status == 'compliant']
        assert len(compliant_checks) > 0
    
    def test_gdpr_compliance_failures(self):
        """Test GDPR compliance with poor configuration."""
        validator = ComplianceValidator()
        
        config = {
            'encryption_at_rest': False,
            'data_deletion_api': False,
            'audit_logging': False
        }
        
        checks = validator.validate_compliance('GDPR', config)
        
        # Should have non-compliant checks
        non_compliant_checks = [c for c in checks if c.status == 'non_compliant']
        assert len(non_compliant_checks) > 0
    
    def test_hipaa_compliance_checks(self):
        """Test HIPAA compliance validation."""
        validator = ComplianceValidator()
        
        config = {
            'access_controls': True,
            'audit_logging': True,
            'data_integrity_controls': True
        }
        
        checks = validator.validate_compliance('HIPAA', config)
        
        assert len(checks) > 0
        assert all(check.regulation == 'HIPAA' for check in checks)
    
    def test_invalid_regulation(self):
        """Test validation with invalid regulation."""
        validator = ComplianceValidator()
        
        with pytest.raises(ValueError):
            validator.validate_compliance('INVALID_REGULATION', {})


class TestVulnerabilityScanner:
    """Test vulnerability scanning."""
    
    def test_scanner_initialization(self):
        """Test vulnerability scanner initialization."""
        scanner = VulnerabilityScanner()
        assert len(scanner.known_vulnerabilities) > 0
        assert 'weak_passwords' in scanner.known_vulnerabilities
    
    def test_scan_secure_system(self):
        """Test scanning secure system configuration."""
        scanner = VulnerabilityScanner()
        
        config = {
            'password_policy': {
                'min_length': 12,
                'require_special_chars': True
            },
            'encryption_at_rest': True,
            'dependencies': [],
            'user_permissions': {}
        }
        
        result = scanner.scan_system(config)
        
        assert 'vulnerabilities' in result
        assert 'risk_score' in result
        assert result['risk_score'] >= 0
    
    def test_scan_vulnerable_system(self):
        """Test scanning vulnerable system configuration."""
        scanner = VulnerabilityScanner()
        
        config = {
            'password_policy': {
                'min_length': 4,
                'require_special_chars': False
            },
            'encryption_at_rest': False,
            'dependencies': ['old-lib==1.0'],
            'user_permissions': {'user1': ['admin', 'root']}
        }
        
        result = scanner.scan_system(config)
        
        # Should find vulnerabilities
        assert result['vulnerabilities_found'] > 0
        assert result['risk_score'] > 0
        assert len(result['recommendations']) > 0


class TestAdvancedSecurityOrchestrator:
    """Test security orchestrator."""
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        orchestrator = AdvancedSecurityOrchestrator()
        
        assert orchestrator.config is not None
        assert orchestrator.encryption is not None
        assert orchestrator.threat_detector is not None
        assert orchestrator.compliance_validator is not None
        assert orchestrator.vulnerability_scanner is not None
    
    def test_security_audit(self):
        """Test comprehensive security audit."""
        orchestrator = AdvancedSecurityOrchestrator()
        
        audit_result = orchestrator.perform_security_audit()
        
        assert audit_result.audit_id is not None
        assert audit_result.overall_score >= 0
        assert audit_result.overall_score <= 1
        assert len(audit_result.recommendations) > 0
        assert audit_result.vulnerability_scan is not None
        assert len(audit_result.compliance_status) > 0
    
    def test_monitor_request_clean(self):
        """Test monitoring clean request."""
        orchestrator = AdvancedSecurityOrchestrator()
        
        request_data = {
            'method': 'GET',
            'path': '/api/health',
            'headers': {'user-agent': 'Mozilla/5.0'},
            'params': {},
            'source_ip': '192.168.1.1'
        }
        
        threats = orchestrator.monitor_request(request_data)
        
        # Should have minimal threats for clean request
        critical_threats = [t for t in threats if t.severity == 'critical']
        assert len(critical_threats) == 0
    
    def test_monitor_request_malicious(self):
        """Test monitoring malicious request."""
        orchestrator = AdvancedSecurityOrchestrator()
        
        request_data = {
            'method': 'POST',
            'path': '/login',
            'headers': {'user-agent': 'sqlmap/1.4'},
            'params': {'username': 'admin', 'password': "'; DROP TABLE users; --"},
            'source_ip': '192.168.1.100'
        }
        
        threats = orchestrator.monitor_request(request_data)
        
        # Should detect multiple threats
        assert len(threats) > 0
        high_severity_threats = [t for t in threats if t.severity in ['high', 'critical']]
        assert len(high_severity_threats) > 0
    
    def test_get_security_metrics(self):
        """Test getting security metrics."""
        orchestrator = AdvancedSecurityOrchestrator()
        
        # Generate some test events
        orchestrator.security_events.append({
            'event_type': 'threat_detected',
            'timestamp': datetime.utcnow().isoformat(),
            'severity': 'high'
        })
        
        metrics = orchestrator.get_security_metrics()
        
        assert 'events_last_7_days' in metrics
        assert 'threat_counts' in metrics
        assert 'config_status' in metrics
    
    def test_security_config_loading(self):
        """Test security configuration loading."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config = {
                'threat_detection_enabled': False,
                'encryption_at_rest': True,
                'custom_setting': 'test_value'
            }
            json.dump(config, f)
            config_path = f.name
        
        try:
            orchestrator = AdvancedSecurityOrchestrator(config_path=config_path)
            
            assert orchestrator.config['threat_detection_enabled'] is False
            assert orchestrator.config['encryption_at_rest'] is True
            assert orchestrator.config['custom_setting'] == 'test_value'
        finally:
            import os
            os.unlink(config_path)


class TestSecurityReporting:
    """Test security reporting functionality."""
    
    def test_create_security_report(self):
        """Test creating comprehensive security report."""
        orchestrator = AdvancedSecurityOrchestrator()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            report_file = create_security_report(orchestrator, temp_dir)
            
            assert report_file is not None
            assert temp_dir in report_file
            
            # Verify report file exists and has content
            with open(report_file, 'r') as f:
                report_data = json.load(f)
            
            assert 'report_metadata' in report_data
            assert 'audit_result' in report_data
            assert 'security_metrics' in report_data
    
    def test_audit_access_controls(self):
        """Test access control auditing."""
        orchestrator = AdvancedSecurityOrchestrator()
        
        audit_result = orchestrator._audit_access_controls()
        
        assert 'mfa_enabled' in audit_result
        assert 'session_timeout' in audit_result
        assert 'password_policy_compliant' in audit_result
    
    def test_audit_data_protection(self):
        """Test data protection auditing."""
        orchestrator = AdvancedSecurityOrchestrator()
        
        audit_result = orchestrator._audit_data_protection()
        
        assert 'encryption_at_rest' in audit_result
        assert 'encryption_in_transit' in audit_result
        assert 'key_management' in audit_result


@pytest.mark.integration
class TestSecurityIntegration:
    """Integration tests for security system."""
    
    def test_end_to_end_security_monitoring(self):
        """Test complete security monitoring workflow."""
        orchestrator = AdvancedSecurityOrchestrator()
        
        # Simulate series of requests
        requests = [
            {  # Clean request
                'method': 'GET',
                'path': '/api/health',
                'headers': {'user-agent': 'Mozilla/5.0'},
                'params': {},
                'source_ip': '192.168.1.1'
            },
            {  # Suspicious request
                'method': 'POST',
                'path': '/login',
                'headers': {'user-agent': 'sqlmap/1.4'},
                'params': {'username': 'admin', 'password': "' OR 1=1 --"},
                'source_ip': '10.0.0.1'
            }
        ]
        
        all_threats = []
        for request in requests:
            threats = orchestrator.monitor_request(request)
            all_threats.extend(threats)
        
        # Should have detected threats from suspicious request
        assert len(all_threats) > 0
        
        # Perform comprehensive audit
        audit_result = orchestrator.perform_security_audit()
        
        # Audit should include detected threats
        assert len(audit_result.threats_detected) > 0
        assert audit_result.overall_score <= 1.0
    
    def test_compliance_and_vulnerability_integration(self):
        """Test integration of compliance and vulnerability checking."""
        orchestrator = AdvancedSecurityOrchestrator()
        
        # Perform audit
        audit_result = orchestrator.perform_security_audit()
        
        # Should have both compliance and vulnerability results
        assert len(audit_result.compliance_status) > 0
        assert audit_result.vulnerability_scan is not None
        assert 'vulnerabilities_found' in audit_result.vulnerability_scan
        
        # Should generate actionable recommendations
        assert len(audit_result.recommendations) > 0
    
    def test_security_metrics_tracking(self):
        """Test security metrics tracking over time."""
        orchestrator = AdvancedSecurityOrchestrator()
        
        # Generate some security events
        for i in range(5):
            request_data = {
                'method': 'POST',
                'path': f'/test{i}',
                'headers': {'user-agent': 'test'},
                'params': {'param': f'<script>alert({i})</script>'},
                'source_ip': f'192.168.1.{100 + i}'
            }
            orchestrator.monitor_request(request_data)
        
        # Get metrics
        metrics = orchestrator.get_security_metrics()
        
        # Should track events
        assert metrics['events_last_7_days'] > 0
        assert metrics['total_threats_detected'] > 0