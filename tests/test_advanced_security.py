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
    VulnerabilityScanner, AdvancedSecurityOrchestrator, create_security_report,
    ZeroTrustAuthenticator, RealTimeSecurityMonitor, AdvancedDataProtection,
    SecurityPolicyEngine, SecureAuditLogger, IncidentResponseAutomation,
    AutomatedVulnerabilityAssessment, AuthenticationStatus, SecurityPolicy,
    IncidentSeverity, ThreatSeverity, ComplianceStatus
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


class TestZeroTrustAuthenticator:
    """Test zero-trust authentication system."""
    
    def test_authenticator_initialization(self):
        """Test authenticator initialization."""
        auth = ZeroTrustAuthenticator()
        assert auth.jwt_secret is not None
        assert auth.token_expiry == 3600
        assert len(auth.failed_attempts) == 0
        assert len(auth.blocked_ips) == 0
    
    def test_successful_authentication(self):
        """Test successful authentication."""
        auth = ZeroTrustAuthenticator()
        
        result = auth.authenticate(
            username="testuser",
            password="securepassword123",
            source_ip="192.168.1.100",
            user_agent="Mozilla/5.0 Test Browser"
        )
        
        assert result.status == AuthenticationStatus.SUCCESS
        assert result.username == "testuser"
        assert result.source_ip == "192.168.1.100"
        assert result.risk_score >= 0.0
    
    def test_failed_authentication(self):
        """Test failed authentication."""
        auth = ZeroTrustAuthenticator()
        
        result = auth.authenticate(
            username="testuser",
            password="weak",  # Too short
            source_ip="192.168.1.100",
            user_agent="Mozilla/5.0 Test Browser"
        )
        
        assert result.status == AuthenticationStatus.FAILURE
        assert result.failure_reason == "Invalid credentials"
    
    def test_mfa_required(self):
        """Test MFA requirement for high-risk scenarios."""
        auth = ZeroTrustAuthenticator()
        
        result = auth.authenticate(
            username="admin_user",  # Admin requires MFA
            password="securepassword123",
            source_ip="192.168.1.100",
            user_agent="Mozilla/5.0 Test Browser"
        )
        
        assert result.status == AuthenticationStatus.REQUIRES_MFA
    
    def test_mfa_verification(self):
        """Test MFA code verification."""
        auth = ZeroTrustAuthenticator()
        
        # Generate MFA token
        mfa_code = auth._generate_mfa_token("testuser")
        
        # Verify the code
        assert auth.verify_mfa("testuser", mfa_code)
        
        # Code should be consumed
        assert not auth.verify_mfa("testuser", mfa_code)
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        auth = ZeroTrustAuthenticator()
        
        # Simulate multiple failed attempts
        for _ in range(6):
            auth._record_failed_attempt("192.168.1.100")
        
        result = auth.authenticate(
            username="testuser",
            password="securepassword123",
            source_ip="192.168.1.100",
            user_agent="Mozilla/5.0 Test Browser"
        )
        
        assert result.status == AuthenticationStatus.BLOCKED


class TestRealTimeSecurityMonitor:
    """Test real-time security monitoring."""
    
    def test_monitor_initialization(self):
        """Test monitor initialization."""
        monitor = RealTimeSecurityMonitor()
        assert monitor.alert_threshold == 0.7
        assert len(monitor.active_incidents) == 0
        assert not monitor.running
    
    def test_event_processing(self):
        """Test security event processing."""
        monitor = RealTimeSecurityMonitor()
        
        event = {
            'event_type': 'test_threat',
            'source_ip': '192.168.1.100',
            'risk_score': 0.8
        }
        
        monitor.process_security_event(event)
        
        # Event should be in queue
        assert len(monitor.event_queue) == 1
        assert monitor.event_queue[0]['event_type'] == 'test_threat'
    
    def test_alert_subscription(self):
        """Test alert subscription functionality."""
        monitor = RealTimeSecurityMonitor()
        alerts_received = []
        
        def alert_callback(alert):
            alerts_received.append(alert)
        
        monitor.add_alert_subscriber(alert_callback)
        
        # Simulate high-risk event
        event = {
            'event_type': 'critical_threat',
            'source_ip': '192.168.1.100',
            'risk_score': 0.9
        }
        
        monitor._analyze_event(event)
        
        # Should have generated alert
        assert len(alerts_received) == 1
        assert alerts_received[0]['event_type'] == 'critical_threat'


class TestAdvancedDataProtection:
    """Test advanced data protection capabilities."""
    
    def test_data_classification(self):
        """Test data classification functionality."""
        protection = AdvancedDataProtection()
        
        data = {
            'user_email': 'test@example.com',
            'password': 'secret123',
            'public_info': 'This is public'
        }
        
        classifications = protection.classify_data(data)
        
        assert classifications['user_email'] == 'pii'
        assert classifications['password'] == 'confidential'
        assert classifications['public_info'] == 'public'
    
    def test_data_protection_application(self):
        """Test data protection policy application."""
        protection = AdvancedDataProtection()
        
        data = {
            'user_email': 'test@example.com',
            'credit_card': '1234567890123456',
            'name': 'John Doe'
        }
        
        protected_data = protection.apply_data_protection(data)
        
        # Should have protection metadata
        assert 'user_email_protected' in protected_data
        assert 'credit_card_protected' in protected_data
        assert 'name_protected' in protected_data
        
        # Data should be protected (not original values)
        assert protected_data['user_email'] != 'test@example.com'
        assert protected_data['credit_card'] != '1234567890123456'
    
    def test_data_protection_removal(self):
        """Test data protection removal."""
        protection = AdvancedDataProtection()
        
        original_data = {
            'user_email': 'test@example.com',
            'name': 'John Doe'
        }
        
        # Apply protection
        protected_data = protection.apply_data_protection(original_data)
        
        # Remove protection
        restored_data = protection.remove_data_protection(protected_data)
        
        # Should restore original values
        assert restored_data['user_email'] == 'test@example.com'
        assert restored_data['name'] == 'John Doe'
    
    def test_differential_privacy(self):
        """Test differential privacy application."""
        protection = AdvancedDataProtection()
        
        data = [
            {'age': 25, 'salary': 50000},
            {'age': 30, 'salary': 60000},
            {'age': 35, 'salary': 70000}
        ]
        
        private_data = protection.apply_differential_privacy(
            data, epsilon=1.0, fields=['age', 'salary']
        )
        
        # Data should be modified but structure preserved
        assert len(private_data) == 3
        assert all('age' in record and 'salary' in record for record in private_data)
        
        # Values should be different due to noise
        for i, record in enumerate(private_data):
            assert record['age'] != data[i]['age']
            assert record['salary'] != data[i]['salary']


class TestSecurityPolicyEngine:
    """Test security policy engine."""
    
    def test_policy_creation(self):
        """Test security policy creation."""
        engine = SecurityPolicyEngine()
        
        policy = SecurityPolicy(
            policy_id="test_policy",
            name="Test Security Policy",
            description="Test policy for unit tests",
            category="access_control",
            rules=[
                {
                    "type": "ip_based",
                    "allowed_ips": ["192.168.1.100"],
                    "name": "ip_whitelist"
                }
            ],
            enforcement_level="blocking",
            applicable_to=["api"],
            created_date=datetime.utcnow(),
            last_updated=datetime.utcnow(),
            version="1.0.0"
        )
        
        result = engine.create_policy(policy)
        assert result is True
        assert "test_policy" in engine.policies
    
    def test_policy_enforcement(self):
        """Test policy enforcement."""
        engine = SecurityPolicyEngine()
        
        # Create a test policy
        policy = SecurityPolicy(
            policy_id="ip_policy",
            name="IP Restriction Policy",
            description="Restrict access by IP",
            category="network_security",
            rules=[
                {
                    "type": "ip_based",
                    "allowed_ips": ["192.168.1.100"],
                    "name": "ip_restriction"
                }
            ],
            enforcement_level="blocking",
            applicable_to=["api"],
            created_date=datetime.utcnow(),
            last_updated=datetime.utcnow(),
            version="1.0.0"
        )
        
        engine.create_policy(policy)
        
        # Test allowed IP
        context = {
            "type": "api",
            "source_ip": "192.168.1.100"
        }
        
        result = engine.enforce_policy("ip_policy", context)
        assert result["allowed"] is True
        
        # Test blocked IP
        context = {
            "type": "api",
            "source_ip": "10.0.0.1"
        }
        
        result = engine.enforce_policy("ip_policy", context)
        assert result["allowed"] is False


class TestSecureAuditLogger:
    """Test secure audit logging with tamper protection."""
    
    def test_logger_initialization(self):
        """Test audit logger initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = os.path.join(temp_dir, "test_audit.log")
            logger = SecureAuditLogger(log_path)
            
            # Should have genesis entry
            assert len(logger.log_entries) == 1
            assert logger.log_entries[0].log_id == "GENESIS_LOG"
            assert logger.current_hash is not None
    
    def test_event_logging(self):
        """Test secure event logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = os.path.join(temp_dir, "test_audit.log")
            logger = SecureAuditLogger(log_path)
            
            log_id = logger.log_event(
                event_type="user_login",
                user_id="testuser",
                source_ip="192.168.1.100",
                resource="authentication_system",
                action="login",
                result="success",
                details={"session_id": "abc123"}
            )
            
            assert log_id is not None
            assert len(logger.log_entries) == 2  # Genesis + new entry
    
    def test_log_integrity_verification(self):
        """Test audit log integrity verification."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = os.path.join(temp_dir, "test_audit.log")
            logger = SecureAuditLogger(log_path)
            
            # Log several events
            for i in range(3):
                logger.log_event(
                    event_type="test_event",
                    user_id=f"user{i}",
                    source_ip="192.168.1.100",
                    resource="test_resource",
                    action="test_action",
                    result="success"
                )
            
            # Verify integrity
            verification = logger.verify_log_integrity()
            assert verification['valid'] is True
            assert len(verification['corrupted_entries']) == 0
    
    def test_audit_trail_filtering(self):
        """Test audit trail filtering."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = os.path.join(temp_dir, "test_audit.log")
            logger = SecureAuditLogger(log_path)
            
            # Log events with different types and users
            logger.log_event("login", "user1", "192.168.1.100", "auth", "login", "success")
            logger.log_event("logout", "user1", "192.168.1.100", "auth", "logout", "success")
            logger.log_event("login", "user2", "192.168.1.101", "auth", "login", "success")
            
            # Filter by event type
            login_events = logger.get_audit_trail(event_type="login")
            assert len(login_events) == 2
            
            # Filter by user
            user1_events = logger.get_audit_trail(user_id="user1")
            assert len(user1_events) == 2


class TestIncidentResponseAutomation:
    """Test automated incident response system."""
    
    def test_incident_creation(self):
        """Test security incident creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = os.path.join(temp_dir, "test_audit.log")
            audit_logger = SecureAuditLogger(log_path)
            response = IncidentResponseAutomation(audit_logger)
            
            incident_id = response.create_incident(
                title="Test Security Incident",
                description="This is a test incident",
                severity=IncidentSeverity.HIGH,
                affected_assets=["server1", "database1"]
            )
            
            assert incident_id is not None
            assert incident_id in response.incidents
            assert response.incidents[incident_id].status == "open"
    
    def test_incident_status_update(self):
        """Test incident status updates."""
        response = IncidentResponseAutomation()
        
        # Create incident
        incident_id = response.create_incident(
            title="Test Incident",
            description="Test incident for status update",
            severity=IncidentSeverity.MEDIUM,
            affected_assets=["server1"]
        )
        
        # Update status
        result = response.update_incident_status(
            incident_id, "investigating", "security_analyst", "Starting investigation"
        )
        
        assert result is True
        assert response.incidents[incident_id].status == "investigating"
        assert response.incidents[incident_id].assigned_to == "security_analyst"
    
    def test_containment_playbook_execution(self):
        """Test automated containment playbook execution."""
        response = IncidentResponseAutomation()
        
        # Create incident
        incident_id = response.create_incident(
            title="SQL Injection Attack",
            description="SQL injection detected",
            severity=IncidentSeverity.HIGH,
            affected_assets=["web_server", "database"]
        )
        
        # Execute SQL injection response playbook
        result = response.execute_containment_playbook(incident_id, "sql_injection_response")
        
        assert result["success"] is True
        assert len(result["actions_executed"]) > 0
        assert len(response.incidents[incident_id].containment_actions) > 0


class TestAutomatedVulnerabilityAssessment:
    """Test automated vulnerability assessment."""
    
    def test_vulnerability_scanner_initialization(self):
        """Test vulnerability scanner initialization."""
        scanner = AutomatedVulnerabilityAssessment()
        assert len(scanner.vulnerability_database) == 0
        assert len(scanner.cve_cache) == 0
        assert len(scanner.scan_history) == 0
    
    def test_dependency_file_finding(self):
        """Test finding dependency files."""
        scanner = AutomatedVulnerabilityAssessment()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test dependency files
            package_json = os.path.join(temp_dir, "package.json")
            requirements_txt = os.path.join(temp_dir, "requirements.txt")
            
            with open(package_json, 'w') as f:
                json.dump({"dependencies": {"test-pkg": "1.0.0"}}, f)
            
            with open(requirements_txt, 'w') as f:
                f.write("old-package==1.0.0\n")
            
            # Find dependency files
            dep_files = scanner._find_dependency_files(temp_dir)
            
            assert package_json in dep_files
            assert requirements_txt in dep_files
    
    def test_dependency_scanning(self):
        """Test dependency vulnerability scanning."""
        scanner = AutomatedVulnerabilityAssessment()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test requirements file with vulnerable package
            requirements_txt = os.path.join(temp_dir, "requirements.txt")
            with open(requirements_txt, 'w') as f:
                f.write("old-package==1.0.0\n")
            
            vulnerabilities = scanner.scan_dependencies(temp_dir)
            
            # Should find vulnerabilities in packages with 'old' or '1.0' patterns
            assert len(vulnerabilities) > 0
            assert len(scanner.scan_history) == 1
    
    def test_remediation_plan_generation(self):
        """Test vulnerability remediation plan generation."""
        scanner = AutomatedVulnerabilityAssessment()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            requirements_txt = os.path.join(temp_dir, "requirements.txt")
            with open(requirements_txt, 'w') as f:
                f.write("old-package==1.0.0\ncritical-old-lib==1.0.0\n")
            
            vulnerabilities = scanner.scan_dependencies(temp_dir)
            remediation_plan = scanner.generate_remediation_plan(vulnerabilities)
            
            assert 'total_vulnerabilities' in remediation_plan
            assert 'immediate_actions' in remediation_plan
            assert 'short_term_actions' in remediation_plan
            assert remediation_plan['total_vulnerabilities'] == len(vulnerabilities)


class TestAdvancedSecurityOrchestrator:
    """Test enhanced security orchestrator with all components."""
    
    def test_orchestrator_initialization_with_all_components(self):
        """Test orchestrator initialization with all security components."""
        orchestrator = AdvancedSecurityOrchestrator()
        
        # Verify all components are initialized
        assert orchestrator.encryption is not None
        assert orchestrator.threat_detector is not None
        assert orchestrator.compliance_validator is not None
        assert orchestrator.vulnerability_scanner is not None
        assert orchestrator.zero_trust_auth is not None
        assert orchestrator.realtime_monitor is not None
        assert orchestrator.data_protection is not None
        assert orchestrator.policy_engine is not None
        assert orchestrator.audit_logger is not None
        assert orchestrator.incident_response is not None
        assert orchestrator.vulnerability_assessment is not None
    
    def test_comprehensive_security_dashboard(self):
        """Test comprehensive security dashboard."""
        orchestrator = AdvancedSecurityOrchestrator()
        
        dashboard = orchestrator.get_comprehensive_security_dashboard()
        
        # Verify all dashboard sections
        assert 'timestamp' in dashboard
        assert 'threat_detection' in dashboard
        assert 'authentication' in dashboard
        assert 'incidents' in dashboard
        assert 'policies' in dashboard
        assert 'audit_logs' in dashboard
        assert 'real_time_monitoring' in dashboard
        assert 'vulnerability_assessment' in dashboard
        assert 'system_metrics' in dashboard
    
    def test_integrated_authentication_flow(self):
        """Test integrated authentication with orchestrator."""
        orchestrator = AdvancedSecurityOrchestrator()
        
        auth_result = orchestrator.authenticate_user(
            username="testuser",
            password="securepassword123",
            source_ip="192.168.1.100",
            user_agent="Mozilla/5.0 Test Browser"
        )
        
        assert auth_result.status == AuthenticationStatus.SUCCESS
        
        # Should have logged authentication event
        assert len(orchestrator.security_events) > 0
    
    def test_security_incident_creation(self):
        """Test security incident creation through orchestrator."""
        orchestrator = AdvancedSecurityOrchestrator()
        
        incident_id = orchestrator.create_security_incident(
            title="Test Security Breach",
            description="Simulated security incident for testing",
            severity="high",
            affected_assets=["web_server", "database"]
        )
        
        assert incident_id is not None
        assert incident_id in orchestrator.incident_response.incidents
    
    def test_vulnerability_scanning_integration(self):
        """Test vulnerability scanning through orchestrator."""
        orchestrator = AdvancedSecurityOrchestrator()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test dependency file
            requirements_txt = os.path.join(temp_dir, "requirements.txt")
            with open(requirements_txt, 'w') as f:
                f.write("old-package==1.0.0\n")
            
            scan_result = orchestrator.scan_project_vulnerabilities(temp_dir)
            
            assert 'vulnerabilities' in scan_result
            assert 'remediation_plan' in scan_result
            assert len(scan_result['vulnerabilities']) > 0
    
    def test_audit_log_integrity_verification(self):
        """Test audit log integrity through orchestrator."""
        orchestrator = AdvancedSecurityOrchestrator()
        
        # Generate some activity
        orchestrator.authenticate_user("testuser", "password123", "192.168.1.1", "TestAgent")
        orchestrator.create_security_incident("Test", "Test incident", "low", ["test"])
        
        # Verify audit integrity
        integrity_result = orchestrator.verify_audit_integrity()
        
        assert integrity_result['valid'] is True
        assert integrity_result['total_entries'] > 1  # Should have multiple entries


@pytest.mark.integration
class TestEnhancedSecurityIntegration:
    """Integration tests for enhanced security system."""
    
    def test_end_to_end_security_workflow(self):
        """Test complete security workflow with all components."""
        orchestrator = AdvancedSecurityOrchestrator()
        
        # 1. Authenticate user
        auth_result = orchestrator.authenticate_user(
            "testuser", "securepass123", "192.168.1.100", "Mozilla/5.0"
        )
        assert auth_result.status == AuthenticationStatus.SUCCESS
        
        # 2. Apply data protection
        sensitive_data = {
            'email': 'user@example.com',
            'credit_card': '1234567890123456'
        }
        protected_data = orchestrator.apply_data_protection_policy(sensitive_data)
        assert 'email_protected' in protected_data
        
        # 3. Monitor malicious request
        malicious_request = {
            'method': 'POST',
            'path': '/api/data',
            'headers': {'user-agent': 'sqlmap'},
            'params': {'query': 'SELECT * FROM users WHERE id=1; DROP TABLE users;--'},
            'source_ip': '10.0.0.1'
        }
        threats = orchestrator.monitor_request(malicious_request)
        assert len(threats) > 0
        
        # 4. Create security incident
        incident_id = orchestrator.create_security_incident(
            "Automated Threat Detection",
            "SQL injection attempt detected",
            "high",
            ["database", "api"]
        )
        assert incident_id is not None
        
        # 5. Get comprehensive dashboard
        dashboard = orchestrator.get_comprehensive_security_dashboard()
        assert dashboard['threat_detection']['total_threats'] > 0
        assert dashboard['incidents']['total_incidents'] > 0
        assert dashboard['audit_logs']['total_entries'] > 0
        
        # 6. Verify audit integrity
        integrity = orchestrator.verify_audit_integrity()
        assert integrity['valid'] is True