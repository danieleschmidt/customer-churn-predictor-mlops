"""
Advanced Security and Compliance Framework.

This module implements enterprise-grade security features including:
- Zero-trust security architecture with multi-layered authentication
- Advanced threat detection and anomaly monitoring
- Automated compliance checking (GDPR, HIPAA, SOC2)
- Real-time security event correlation and response
- ML model security validation and privacy protection
- Supply chain security and dependency scanning
"""

import os
import json
import time
import hashlib
import secrets
import hmac
import jwt
import threading
import asyncio
import sqlite3
try:
    import geoip2.database
    import geoip2.errors
    GEOIP_AVAILABLE = True
except ImportError:
    GEOIP_AVAILABLE = False
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union, Set, Callable
from dataclasses import dataclass, asdict, field
from pathlib import Path
from collections import defaultdict, deque
from enum import Enum
import numpy as np
import pandas as pd
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import base64
import re
import subprocess
import requests
from urllib.parse import urlparse
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .logging_config import get_logger
from .metrics import get_metrics_collector

logger = get_logger(__name__)
metrics = get_metrics_collector()


class ThreatSeverity(Enum):
    """Enumeration for threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuthenticationStatus(Enum):
    """Enumeration for authentication status."""
    SUCCESS = "success"
    FAILURE = "failure"
    BLOCKED = "blocked"
    REQUIRES_MFA = "requires_mfa"


class ComplianceStatus(Enum):
    """Enumeration for compliance status."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    WARNING = "warning"
    UNDER_REVIEW = "under_review"


class IncidentSeverity(Enum):
    """Enumeration for incident severity levels."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityThreat:
    """Representation of a detected security threat."""
    threat_id: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    category: str
    description: str
    source_ip: Optional[str]
    user_agent: Optional[str]
    endpoint: Optional[str]
    payload: Optional[str]
    timestamp: str
    mitigation_actions: List[str]
    risk_score: float


@dataclass
class ComplianceCheck:
    """Result of compliance validation."""
    regulation: str  # 'GDPR', 'HIPAA', 'SOC2', 'PCI-DSS'
    requirement: str
    status: str  # 'compliant', 'non_compliant', 'warning'
    details: str
    evidence: List[str]
    remediation_steps: List[str]
    last_checked: str


@dataclass
class SecurityAuditResult:
    """Comprehensive security audit result."""
    audit_id: str
    audit_timestamp: str
    overall_score: float
    threats_detected: List[SecurityThreat]
    compliance_status: List[ComplianceCheck]
    vulnerability_scan: Dict[str, Any]
    access_control_audit: Dict[str, Any]
    data_protection_audit: Dict[str, Any]
    recommendations: List[str]
    next_audit_date: str


@dataclass
class AuthenticationAttempt:
    """Representation of an authentication attempt."""
    user_id: str
    username: str
    source_ip: str
    user_agent: str
    timestamp: datetime
    status: AuthenticationStatus
    failure_reason: Optional[str] = None
    device_fingerprint: Optional[str] = None
    geolocation: Optional[Dict[str, Any]] = None
    risk_score: float = 0.0
    mfa_verified: bool = False


@dataclass
class SecurityIncident:
    """Representation of a security incident."""
    incident_id: str
    severity: IncidentSeverity
    category: str
    title: str
    description: str
    affected_assets: List[str]
    detection_timestamp: datetime
    status: str  # 'open', 'investigating', 'contained', 'resolved'
    assigned_to: Optional[str] = None
    containment_actions: List[str] = field(default_factory=list)
    resolution_actions: List[str] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    timeline: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class VulnerabilityAssessment:
    """Result of vulnerability assessment."""
    vulnerability_id: str
    cve_id: Optional[str]
    severity: str
    component: str
    version: str
    description: str
    impact: str
    remediation: str
    cvss_score: float
    exploitable: bool
    detection_date: datetime
    remediation_deadline: datetime


@dataclass
class SecurityPolicy:
    """Security policy definition."""
    policy_id: str
    name: str
    description: str
    category: str
    rules: List[Dict[str, Any]]
    enforcement_level: str  # 'advisory', 'warning', 'blocking'
    applicable_to: List[str]
    created_date: datetime
    last_updated: datetime
    version: str
    is_active: bool = True


@dataclass
class AuditLogEntry:
    """Tamper-proof audit log entry."""
    log_id: str
    timestamp: datetime
    event_type: str
    user_id: Optional[str]
    source_ip: str
    resource: str
    action: str
    result: str
    details: Dict[str, Any]
    integrity_hash: str
    previous_hash: Optional[str] = None


@dataclass
class TenantSecurityContext:
    """Security context for multi-tenant environment."""
    tenant_id: str
    tenant_name: str
    security_level: str  # 'basic', 'enhanced', 'enterprise'
    encryption_keys: Dict[str, str]
    access_policies: List[str]
    compliance_requirements: List[str]
    resource_quotas: Dict[str, Any]
    isolation_level: str  # 'logical', 'physical'


class AdvancedEncryption:
    """Advanced encryption and cryptographic utilities."""
    
    def __init__(self, password: str = None):
        self.password = password or self._generate_secure_password()
        self.fernet = self._initialize_fernet()
        
    def _generate_secure_password(self) -> str:
        """Generate cryptographically secure password."""
        return secrets.token_urlsafe(32)
    
    def _initialize_fernet(self) -> Fernet:
        """Initialize Fernet encryption with password-based key derivation."""
        password_bytes = self.password.encode()
        salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
        return Fernet(key)
    
    def encrypt_data(self, data: Union[str, bytes, Dict]) -> str:
        """Encrypt data with advanced encryption."""
        if isinstance(data, dict):
            data = json.dumps(data)
        if isinstance(data, str):
            data = data.encode()
        
        encrypted = self.fernet.encrypt(data)
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt_data(self, encrypted_data: str) -> Union[str, Dict]:
        """Decrypt data."""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted = self.fernet.decrypt(encrypted_bytes)
            
            # Try to parse as JSON
            try:
                return json.loads(decrypted.decode())
            except json.JSONDecodeError:
                return decrypted.decode()
                
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate secure random token."""
        return secrets.token_urlsafe(length)
    
    def hash_sensitive_data(self, data: str, salt: str = None) -> Tuple[str, str]:
        """Hash sensitive data with salt."""
        if salt is None:
            salt = secrets.token_hex(16)
        
        # Use PBKDF2 for password hashing
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt.encode(),
            iterations=100000,
        )
        hashed = kdf.derive(data.encode())
        return base64.urlsafe_b64encode(hashed).decode(), salt
    
    def generate_rsa_keypair(self, key_size: int = 2048) -> Tuple[bytes, bytes]:
        """Generate RSA key pair for asymmetric encryption."""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size
        )
        
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_key = private_key.public_key()
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        return private_pem, public_pem
    
    def encrypt_with_public_key(self, data: bytes, public_key_pem: bytes) -> bytes:
        """Encrypt data with RSA public key."""
        public_key = serialization.load_pem_public_key(public_key_pem)
        encrypted = public_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return encrypted
    
    def decrypt_with_private_key(self, encrypted_data: bytes, private_key_pem: bytes) -> bytes:
        """Decrypt data with RSA private key."""
        private_key = serialization.load_pem_private_key(
            private_key_pem,
            password=None
        )
        decrypted = private_key.decrypt(
            encrypted_data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return decrypted
    
    def encrypt_field_level(self, data: Dict[str, Any], sensitive_fields: List[str]) -> Dict[str, Any]:
        """Encrypt specific fields in a dictionary."""
        encrypted_data = data.copy()
        
        for field in sensitive_fields:
            if field in encrypted_data:
                if isinstance(encrypted_data[field], (str, int, float)):
                    encrypted_data[field] = self.encrypt_data(str(encrypted_data[field]))
                    encrypted_data[f"{field}_encrypted"] = True
                
        return encrypted_data
    
    def decrypt_field_level(self, encrypted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt field-level encrypted data."""
        decrypted_data = encrypted_data.copy()
        
        # Find encrypted fields
        encrypted_fields = [k.replace('_encrypted', '') for k in encrypted_data.keys() if k.endswith('_encrypted')]
        
        for field in encrypted_fields:
            if field in decrypted_data and decrypted_data.get(f"{field}_encrypted"):
                decrypted_data[field] = self.decrypt_data(decrypted_data[field])
                del decrypted_data[f"{field}_encrypted"]
        
        return decrypted_data
    
    def tokenize_sensitive_data(self, data: str) -> Tuple[str, str]:
        """Tokenize sensitive data and return token and encryption mapping."""
        token = f"TOKEN_{secrets.token_urlsafe(16)}"
        encrypted_value = self.encrypt_data(data)
        return token, encrypted_value


class ZeroTrustAuthenticator:
    """Zero-trust authentication system with multi-factor authentication."""
    
    def __init__(self, jwt_secret: str = None, token_expiry: int = 3600):
        self.jwt_secret = jwt_secret or secrets.token_urlsafe(32)
        self.token_expiry = token_expiry
        self.failed_attempts = defaultdict(list)
        self.blocked_ips = set()
        self.device_fingerprints = {}
        self.mfa_tokens = {}
        self.session_store = {}
        self._lock = threading.Lock()
        
        # Initialize GeoIP if available
        self.geoip_reader = None
        if GEOIP_AVAILABLE:
            try:
                # Try to load GeoIP database (would need to be downloaded separately)
                geoip_path = os.getenv('GEOIP_DATABASE_PATH', '/opt/geoip/GeoLite2-City.mmdb')
                if os.path.exists(geoip_path):
                    self.geoip_reader = geoip2.database.Reader(geoip_path)
            except Exception as e:
                logger.warning(f"Failed to initialize GeoIP: {e}")
    
    def authenticate(self, username: str, password: str, source_ip: str, 
                    user_agent: str, additional_factors: Dict[str, Any] = None) -> AuthenticationAttempt:
        """Perform zero-trust authentication."""
        
        # Check if IP is blocked
        if source_ip in self.blocked_ips:
            return self._create_auth_attempt(
                username, source_ip, user_agent, 
                AuthenticationStatus.BLOCKED, "IP address blocked"
            )
        
        # Check rate limiting
        if self._is_rate_limited(source_ip):
            return self._create_auth_attempt(
                username, source_ip, user_agent,
                AuthenticationStatus.BLOCKED, "Rate limit exceeded"
            )
        
        # Generate device fingerprint
        device_fingerprint = self._generate_device_fingerprint(user_agent, source_ip)
        
        # Get geolocation
        geolocation = self._get_geolocation(source_ip)
        
        # Calculate risk score
        risk_score = self._calculate_authentication_risk(
            username, source_ip, device_fingerprint, geolocation
        )
        
        # Simulate password verification (in practice, would verify against secure storage)
        password_valid = self._verify_password(username, password)
        
        if not password_valid:
            self._record_failed_attempt(source_ip)
            return self._create_auth_attempt(
                username, source_ip, user_agent,
                AuthenticationStatus.FAILURE, "Invalid credentials",
                device_fingerprint, geolocation, risk_score
            )
        
        # Check if MFA is required based on risk score
        if risk_score > 0.5 or self._requires_mfa(username):
            mfa_token = self._generate_mfa_token(username)
            return self._create_auth_attempt(
                username, source_ip, user_agent,
                AuthenticationStatus.REQUIRES_MFA, "MFA required",
                device_fingerprint, geolocation, risk_score
            )
        
        # Successful authentication
        session_token = self._create_session_token(username, source_ip, device_fingerprint)
        
        auth_attempt = self._create_auth_attempt(
            username, source_ip, user_agent,
            AuthenticationStatus.SUCCESS, None,
            device_fingerprint, geolocation, risk_score
        )
        
        # Store session
        self.session_store[session_token] = {
            'username': username,
            'created_at': datetime.utcnow(),
            'source_ip': source_ip,
            'device_fingerprint': device_fingerprint
        }
        
        return auth_attempt
    
    def verify_mfa(self, username: str, mfa_code: str) -> bool:
        """Verify multi-factor authentication code."""
        with self._lock:
            stored_code = self.mfa_tokens.get(username)
            if stored_code and stored_code['code'] == mfa_code:
                # Check if token is still valid (5 minutes)
                if datetime.utcnow() - stored_code['created_at'] < timedelta(minutes=5):
                    del self.mfa_tokens[username]
                    return True
                else:
                    del self.mfa_tokens[username]
        return False
    
    def validate_session_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate JWT session token."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            
            # Check if session exists
            if token in self.session_store:
                session = self.session_store[token]
                # Check if session is expired
                if datetime.utcnow() - session['created_at'] < timedelta(seconds=self.token_expiry):
                    return session
                else:
                    del self.session_store[token]
            
            return None
            
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid JWT token")
            return None
    
    def revoke_session(self, token: str) -> bool:
        """Revoke session token."""
        with self._lock:
            if token in self.session_store:
                del self.session_store[token]
                return True
        return False
    
    def _create_auth_attempt(self, username: str, source_ip: str, user_agent: str,
                           status: AuthenticationStatus, failure_reason: Optional[str] = None,
                           device_fingerprint: Optional[str] = None,
                           geolocation: Optional[Dict[str, Any]] = None,
                           risk_score: float = 0.0) -> AuthenticationAttempt:
        """Create authentication attempt record."""
        return AuthenticationAttempt(
            user_id=f"user_{hashlib.md5(username.encode()).hexdigest()[:8]}",
            username=username,
            source_ip=source_ip,
            user_agent=user_agent,
            timestamp=datetime.utcnow(),
            status=status,
            failure_reason=failure_reason,
            device_fingerprint=device_fingerprint,
            geolocation=geolocation,
            risk_score=risk_score,
            mfa_verified=False
        )
    
    def _generate_device_fingerprint(self, user_agent: str, source_ip: str) -> str:
        """Generate device fingerprint based on user agent and IP."""
        fingerprint_data = f"{user_agent}_{source_ip}"
        return hashlib.sha256(fingerprint_data.encode()).hexdigest()[:16]
    
    def _get_geolocation(self, source_ip: str) -> Optional[Dict[str, Any]]:
        """Get geolocation information for IP address."""
        if not self.geoip_reader:
            return None
        
        try:
            response = self.geoip_reader.city(source_ip)
            return {
                'country': response.country.name,
                'country_code': response.country.iso_code,
                'city': response.city.name,
                'latitude': float(response.location.latitude) if response.location.latitude else None,
                'longitude': float(response.location.longitude) if response.location.longitude else None
            }
        except (geoip2.errors.AddressNotFoundError, ValueError):
            return None
    
    def _calculate_authentication_risk(self, username: str, source_ip: str,
                                     device_fingerprint: str,
                                     geolocation: Optional[Dict[str, Any]]) -> float:
        """Calculate authentication risk score."""
        risk_score = 0.0
        
        # Check for unusual IP
        if source_ip not in [attempt.source_ip for attempt in 
                           self.failed_attempts.get(username, [])[-10:]]:
            risk_score += 0.2
        
        # Check for new device
        if device_fingerprint not in self.device_fingerprints.get(username, set()):
            risk_score += 0.3
        
        # Check for unusual location
        if geolocation:
            # Simplified logic - in practice would compare with user's typical locations
            if geolocation.get('country_code') not in ['US', 'CA', 'GB']:
                risk_score += 0.2
        
        # Check recent failed attempts
        recent_failures = len([
            attempt for attempt in self.failed_attempts.get(username, [])
            if datetime.utcnow() - attempt < timedelta(hours=1)
        ])
        if recent_failures > 3:
            risk_score += 0.4
        
        return min(1.0, risk_score)
    
    def _verify_password(self, username: str, password: str) -> bool:
        """Verify password (placeholder - would use secure storage in practice)."""
        # Simplified verification - in practice would use secure hashed storage
        return len(password) >= 8
    
    def _requires_mfa(self, username: str) -> bool:
        """Check if user requires MFA."""
        # Simplified logic - in practice would check user preferences/policy
        return username.startswith('admin') or username.endswith('_privileged')
    
    def _generate_mfa_token(self, username: str) -> str:
        """Generate MFA token."""
        mfa_code = f"{secrets.randbelow(1000000):06d}"
        with self._lock:
            self.mfa_tokens[username] = {
                'code': mfa_code,
                'created_at': datetime.utcnow()
            }
        # In practice, would send via SMS/email/authenticator app
        logger.info(f"MFA code generated for {username}: {mfa_code}")
        return mfa_code
    
    def _create_session_token(self, username: str, source_ip: str, device_fingerprint: str) -> str:
        """Create JWT session token."""
        payload = {
            'username': username,
            'source_ip': source_ip,
            'device_fingerprint': device_fingerprint,
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + timedelta(seconds=self.token_expiry)
        }
        return jwt.encode(payload, self.jwt_secret, algorithm='HS256')
    
    def _is_rate_limited(self, source_ip: str) -> bool:
        """Check if IP is rate limited."""
        now = datetime.utcnow()
        # Clean old attempts
        self.failed_attempts[source_ip] = [
            attempt for attempt in self.failed_attempts[source_ip]
            if now - attempt < timedelta(minutes=15)
        ]
        
        # Check if too many attempts
        return len(self.failed_attempts[source_ip]) > 5
    
    def _record_failed_attempt(self, source_ip: str):
        """Record failed authentication attempt."""
        with self._lock:
            self.failed_attempts[source_ip].append(datetime.utcnow())
            
            # Block IP if too many failures
            if len(self.failed_attempts[source_ip]) > 10:
                self.blocked_ips.add(source_ip)
                logger.warning(f"Blocked IP {source_ip} due to excessive failed attempts")


class ThreatDetectionSystem:
    """Advanced threat detection and monitoring."""
    
    def __init__(self):
        self.threat_patterns = self._load_threat_patterns()
        self.threat_history = []
        self.ip_reputation_cache = {}
        self.anomaly_baselines = {}
        
    def _load_threat_patterns(self) -> Dict[str, List[str]]:
        """Load threat detection patterns."""
        return {
            'sql_injection': [
                r"(\bunion\b.*\bselect\b)|(\bselect\b.*\bunion\b)",
                r"(\bdrop\b.*\btable\b)|(\btable\b.*\bdrop\b)",
                r"(--|#|/\*|\*/)",
                r"(\bor\b.*=.*\bor\b)|(\band\b.*=.*\band\b)",
                r"('.*'.*=.*'.*')"
            ],
            'xss': [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"on\w+\s*=",
                r"<iframe[^>]*>.*?</iframe>",
                r"<object[^>]*>.*?</object>"
            ],
            'path_traversal': [
                r"\.\./",
                r"\.\.\\",
                r"%2e%2e%2f",
                r"%2e%2e\\",
                r"\.\.%2f"
            ],
            'command_injection': [
                r"[;&|`]",
                r"\$\([^)]*\)",
                r"`[^`]*`",
                r"\|\s*\w+",
                r"&&\s*\w+"
            ],
            'suspicious_user_agents': [
                r"sqlmap",
                r"nikto",
                r"nessus",
                r"burp",
                r"nmap",
                r"masscan",
                r"hydra",
                r"gobuster"
            ]
        }
    
    def analyze_request(self, request_data: Dict[str, Any]) -> List[SecurityThreat]:
        """Analyze HTTP request for security threats."""
        threats = []
        
        # Extract request components
        method = request_data.get('method', 'GET')
        path = request_data.get('path', '')
        headers = request_data.get('headers', {})
        params = request_data.get('params', {})
        body = request_data.get('body', '')
        source_ip = request_data.get('source_ip', 'unknown')
        user_agent = headers.get('user-agent', '')
        
        # Check for malicious patterns
        for threat_type, patterns in self.threat_patterns.items():
            for pattern in patterns:
                # Check URL path
                if re.search(pattern, path, re.IGNORECASE):
                    threats.append(self._create_threat(
                        threat_type, f"Suspicious pattern in URL: {pattern}",
                        source_ip, user_agent, path, pattern
                    ))
                
                # Check parameters
                for param_name, param_value in params.items():
                    if isinstance(param_value, str) and re.search(pattern, param_value, re.IGNORECASE):
                        threats.append(self._create_threat(
                            threat_type, f"Suspicious pattern in parameter {param_name}",
                            source_ip, user_agent, path, param_value
                        ))
                
                # Check request body
                if isinstance(body, str) and re.search(pattern, body, re.IGNORECASE):
                    threats.append(self._create_threat(
                        threat_type, f"Suspicious pattern in request body",
                        source_ip, user_agent, path, body[:100]
                    ))
                
                # Check user agent
                if threat_type == 'suspicious_user_agents' and re.search(pattern, user_agent, re.IGNORECASE):
                    threats.append(self._create_threat(
                        'malicious_user_agent', f"Suspicious user agent detected",
                        source_ip, user_agent, path, user_agent
                    ))
        
        # Rate limiting analysis
        if self._detect_rate_limiting_violation(source_ip):
            threats.append(self._create_threat(
                'rate_limiting', "Rate limit exceeded - potential DDoS",
                source_ip, user_agent, path, f"IP: {source_ip}"
            ))
        
        # IP reputation check
        if self._check_ip_reputation(source_ip):
            threats.append(self._create_threat(
                'malicious_ip', "Request from known malicious IP",
                source_ip, user_agent, path, f"IP: {source_ip}"
            ))
        
        # Store threats in history
        self.threat_history.extend(threats)
        
        # Keep only last 1000 threats
        if len(self.threat_history) > 1000:
            self.threat_history = self.threat_history[-1000:]
        
        return threats
    
    def _create_threat(self, category: str, description: str, source_ip: str,
                      user_agent: str, endpoint: str, payload: str) -> SecurityThreat:
        """Create a security threat object."""
        
        # Determine severity
        severity_mapping = {
            'sql_injection': 'critical',
            'command_injection': 'critical',
            'xss': 'high',
            'path_traversal': 'high',
            'malicious_user_agent': 'medium',
            'rate_limiting': 'medium',
            'malicious_ip': 'high'
        }
        
        severity = severity_mapping.get(category, 'low')
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(category, source_ip)
        
        # Generate mitigation actions
        mitigation_actions = self._generate_mitigation_actions(category, severity)
        
        return SecurityThreat(
            threat_id=hashlib.md5(f"{category}{source_ip}{endpoint}{time.time()}".encode()).hexdigest(),
            severity=severity,
            category=category,
            description=description,
            source_ip=source_ip,
            user_agent=user_agent,
            endpoint=endpoint,
            payload=payload,
            timestamp=datetime.utcnow().isoformat(),
            mitigation_actions=mitigation_actions,
            risk_score=risk_score
        )
    
    def _detect_rate_limiting_violation(self, source_ip: str) -> bool:
        """Detect rate limiting violations."""
        # Simplified rate limiting check
        current_time = time.time()
        window_size = 60  # 1 minute window
        max_requests = 100
        
        # Count recent requests from this IP
        recent_requests = [
            t for t in self.threat_history 
            if t.source_ip == source_ip and 
               (current_time - time.mktime(time.strptime(t.timestamp[:19], "%Y-%m-%dT%H:%M:%S"))) < window_size
        ]
        
        return len(recent_requests) > max_requests
    
    def _check_ip_reputation(self, source_ip: str) -> bool:
        """Check IP reputation against threat intelligence."""
        # Check cache first
        if source_ip in self.ip_reputation_cache:
            cache_entry = self.ip_reputation_cache[source_ip]
            if time.time() - cache_entry['timestamp'] < 3600:  # 1 hour cache
                return cache_entry['is_malicious']
        
        # Simplified IP reputation check (in practice, would use real threat intel)
        is_malicious = False
        
        # Check for obvious indicators
        if (source_ip.startswith('10.') or 
            source_ip.startswith('192.168.') or 
            source_ip.startswith('172.')):
            is_malicious = False  # Private IP ranges
        elif source_ip in ['127.0.0.1', '::1']:
            is_malicious = False  # Loopback
        else:
            # Simulate threat intel check (would be real API call)
            # For demo, mark some IPs as suspicious based on pattern
            is_malicious = hash(source_ip) % 100 < 5  # 5% chance for demo
        
        # Cache result
        self.ip_reputation_cache[source_ip] = {
            'is_malicious': is_malicious,
            'timestamp': time.time()
        }
        
        return is_malicious
    
    def _calculate_risk_score(self, category: str, source_ip: str) -> float:
        """Calculate risk score for threat."""
        base_scores = {
            'sql_injection': 0.9,
            'command_injection': 0.95,
            'xss': 0.7,
            'path_traversal': 0.6,
            'malicious_user_agent': 0.4,
            'rate_limiting': 0.3,
            'malicious_ip': 0.5
        }
        
        base_score = base_scores.get(category, 0.2)
        
        # Adjust based on IP reputation
        if self._check_ip_reputation(source_ip):
            base_score *= 1.5
        
        # Adjust based on frequency
        recent_threats = [
            t for t in self.threat_history 
            if t.source_ip == source_ip and 
               (datetime.utcnow() - datetime.fromisoformat(t.timestamp.replace('Z', '+00:00'))).seconds < 300
        ]
        
        if len(recent_threats) > 3:
            base_score *= 1.3
        
        return min(1.0, base_score)
    
    def _generate_mitigation_actions(self, category: str, severity: str) -> List[str]:
        """Generate mitigation actions for threat."""
        actions = []
        
        if severity in ['critical', 'high']:
            actions.append("Block IP address immediately")
            actions.append("Alert security team")
        
        if category in ['sql_injection', 'command_injection']:
            actions.extend([
                "Review database access logs",
                "Check for data exfiltration",
                "Update input validation"
            ])
        elif category == 'xss':
            actions.extend([
                "Review output encoding",
                "Check for stored XSS payloads",
                "Update CSP headers"
            ])
        elif category == 'path_traversal':
            actions.extend([
                "Review file access permissions",
                "Check for unauthorized file access",
                "Update path validation"
            ])
        
        actions.append("Log incident for investigation")
        
        return actions
    
    def initialize_behavioral_detection(self):
        """Initialize machine learning models for behavioral anomaly detection."""
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available, behavioral detection disabled")
            return
        
        self.anomaly_detector = IsolationForest(
            contamination=0.1,  # Expect 10% anomalies
            random_state=42
        )
        self.scaler = StandardScaler()
        self.behavioral_features = []
        self.model_trained = False
        
        logger.info("Behavioral anomaly detection initialized")
    
    def extract_behavioral_features(self, request_data: Dict[str, Any]) -> List[float]:
        """Extract behavioral features from request data."""
        features = []
        
        # Request timing features
        current_time = time.time()
        features.extend([
            current_time % 86400,  # Time of day
            (current_time // 86400) % 7,  # Day of week
        ])
        
        # Request characteristics
        path = request_data.get('path', '')
        features.extend([
            len(path),  # Path length
            path.count('/'),  # Path depth
            len(request_data.get('params', {})),  # Number of parameters
            len(str(request_data.get('body', ''))),  # Body length
        ])
        
        # IP-based features (simplified)
        source_ip = request_data.get('source_ip', '0.0.0.0')
        ip_parts = source_ip.split('.')
        if len(ip_parts) == 4:
            try:
                features.extend([int(part) for part in ip_parts])
            except ValueError:
                features.extend([0, 0, 0, 0])
        else:
            features.extend([0, 0, 0, 0])
        
        # User agent features
        user_agent = request_data.get('headers', {}).get('user-agent', '')
        features.extend([
            len(user_agent),  # User agent length
            user_agent.count('Mozilla'),  # Browser indicator
            user_agent.count('bot'),  # Bot indicator
        ])
        
        return features
    
    def update_behavioral_model(self, request_data_batch: List[Dict[str, Any]]):
        """Update behavioral anomaly detection model with new data."""
        if not SKLEARN_AVAILABLE:
            return
        
        if not hasattr(self, 'anomaly_detector'):
            self.initialize_behavioral_detection()
        
        # Extract features from batch
        feature_batch = []
        for request_data in request_data_batch:
            features = self.extract_behavioral_features(request_data)
            feature_batch.append(features)
        
        if len(feature_batch) > 0:
            # Add to historical features
            self.behavioral_features.extend(feature_batch)
            
            # Keep only recent features (last 10000)
            if len(self.behavioral_features) > 10000:
                self.behavioral_features = self.behavioral_features[-10000:]
            
            # Retrain model if we have enough data
            if len(self.behavioral_features) >= 100:
                try:
                    features_array = np.array(self.behavioral_features)
                    scaled_features = self.scaler.fit_transform(features_array)
                    self.anomaly_detector.fit(scaled_features)
                    self.model_trained = True
                    logger.debug(f"Behavioral model updated with {len(self.behavioral_features)} samples")
                except Exception as e:
                    logger.error(f"Failed to update behavioral model: {e}")
    
    def detect_behavioral_anomaly(self, request_data: Dict[str, Any]) -> float:
        """Detect behavioral anomalies using machine learning."""
        if not SKLEARN_AVAILABLE or not hasattr(self, 'anomaly_detector') or not self.model_trained:
            return 0.0
        
        try:
            features = self.extract_behavioral_features(request_data)
            features_array = np.array([features])
            scaled_features = self.scaler.transform(features_array)
            
            # Get anomaly score (-1 for anomaly, 1 for normal)
            anomaly_score = self.anomaly_detector.decision_function(scaled_features)[0]
            
            # Convert to 0-1 scale (higher = more anomalous)
            normalized_score = max(0, (1 - anomaly_score) / 2)
            
            return min(1.0, normalized_score)
            
        except Exception as e:
            logger.error(f"Behavioral anomaly detection failed: {e}")
            return 0.0
    
    def analyze_request_enhanced(self, request_data: Dict[str, Any]) -> List[SecurityThreat]:
        """Enhanced request analysis with behavioral detection."""
        threats = self.analyze_request(request_data)  # Use existing analysis
        
        # Add behavioral anomaly detection
        behavioral_score = self.detect_behavioral_anomaly(request_data)
        
        if behavioral_score > 0.7:  # High anomaly threshold
            threat = self._create_threat(
                'behavioral_anomaly',
                f"Behavioral anomaly detected (score: {behavioral_score:.2f})",
                request_data.get('source_ip', 'unknown'),
                request_data.get('headers', {}).get('user-agent', ''),
                request_data.get('path', ''),
                f"Anomaly score: {behavioral_score:.2f}"
            )
            threats.append(threat)
        
        return threats
    
    def get_threat_intelligence(self, threat_type: str) -> Dict[str, Any]:
        """Get threat intelligence for specific threat type."""
        intel = {
            'sql_injection': {
                'description': 'SQL injection attacks attempt to execute malicious SQL commands',
                'indicators': ['union select', 'drop table', 'or 1=1'],
                'mitigation': 'Use parameterized queries and input validation',
                'severity_factors': ['payload_complexity', 'target_sensitivity']
            },
            'xss': {
                'description': 'Cross-site scripting attacks inject malicious scripts',
                'indicators': ['<script>', 'javascript:', 'onload='],
                'mitigation': 'Implement output encoding and CSP headers',
                'severity_factors': ['script_complexity', 'execution_context']
            },
            'behavioral_anomaly': {
                'description': 'Unusual behavioral patterns detected by ML models',
                'indicators': ['unusual_timing', 'abnormal_patterns', 'statistical_deviation'],
                'mitigation': 'Investigate user behavior and implement additional monitoring',
                'severity_factors': ['anomaly_score', 'pattern_deviation']
            }
        }
        
        return intel.get(threat_type, {
            'description': 'Unknown threat type',
            'indicators': [],
            'mitigation': 'General security monitoring recommended',
            'severity_factors': []
        })
    
    def classify_threat_with_ml(self, threat: SecurityThreat) -> SecurityThreat:
        """Use ML techniques to enhance threat classification."""
        # Get threat intelligence
        intel = self.get_threat_intelligence(threat.category)
        
        # Enhance threat with intelligence
        threat.description += f" | Intelligence: {intel['description']}"
        
        # Adjust risk score based on intelligence
        severity_factors = intel.get('severity_factors', [])
        if severity_factors:
            # Increase risk for threats with multiple severity factors
            risk_multiplier = 1 + (len(severity_factors) * 0.1)
            threat.risk_score = min(1.0, threat.risk_score * risk_multiplier)
        
        return threat


class ComplianceValidator:
    """Automated compliance checking and validation."""
    
    def __init__(self):
        self.regulations = {
            'GDPR': self._gdpr_checks,
            'HIPAA': self._hipaa_checks,
            'SOC2': self._soc2_checks,
            'PCI_DSS': self._pci_dss_checks
        }
    
    def validate_compliance(self, regulation: str, system_config: Dict[str, Any]) -> List[ComplianceCheck]:
        """Validate compliance for specific regulation."""
        if regulation not in self.regulations:
            raise ValueError(f"Unsupported regulation: {regulation}")
        
        return self.regulations[regulation](system_config)
    
    def _gdpr_checks(self, config: Dict[str, Any]) -> List[ComplianceCheck]:
        """GDPR compliance checks."""
        checks = []
        
        # Data encryption at rest
        checks.append(ComplianceCheck(
            regulation="GDPR",
            requirement="Article 32 - Data Encryption",
            status="compliant" if config.get('encryption_at_rest', False) else "non_compliant",
            details="Personal data must be encrypted when stored",
            evidence=["Encryption configuration verified"] if config.get('encryption_at_rest') else [],
            remediation_steps=[] if config.get('encryption_at_rest') else ["Enable database encryption", "Implement key management"],
            last_checked=datetime.utcnow().isoformat()
        ))
        
        # Right to be forgotten
        checks.append(ComplianceCheck(
            regulation="GDPR",
            requirement="Article 17 - Right to Erasure",
            status="compliant" if config.get('data_deletion_api', False) else "warning",
            details="Users must be able to request deletion of their data",
            evidence=["Data deletion API available"] if config.get('data_deletion_api') else [],
            remediation_steps=[] if config.get('data_deletion_api') else ["Implement data deletion API", "Create data retention policy"],
            last_checked=datetime.utcnow().isoformat()
        ))
        
        # Data processing logs
        checks.append(ComplianceCheck(
            regulation="GDPR",
            requirement="Article 30 - Records of Processing",
            status="compliant" if config.get('audit_logging', False) else "non_compliant",
            details="All data processing activities must be logged",
            evidence=["Audit logging enabled"] if config.get('audit_logging') else [],
            remediation_steps=[] if config.get('audit_logging') else ["Enable comprehensive audit logging", "Set up log retention"],
            last_checked=datetime.utcnow().isoformat()
        ))
        
        # Data minimization
        checks.append(ComplianceCheck(
            regulation="GDPR",
            requirement="Article 5 - Data Minimization",
            status="warning",
            details="Only necessary data should be collected and processed",
            evidence=["Data collection review needed"],
            remediation_steps=["Review data collection practices", "Remove unnecessary data fields", "Implement data retention limits"],
            last_checked=datetime.utcnow().isoformat()
        ))
        
        return checks
    
    def _hipaa_checks(self, config: Dict[str, Any]) -> List[ComplianceCheck]:
        """HIPAA compliance checks."""
        checks = []
        
        # Access controls
        checks.append(ComplianceCheck(
            regulation="HIPAA",
            requirement="164.312(a)(1) - Access Control",
            status="compliant" if config.get('access_controls', False) else "non_compliant",
            details="Unique user identification, emergency access, automatic logoff, encryption/decryption",
            evidence=["Access control system verified"] if config.get('access_controls') else [],
            remediation_steps=[] if config.get('access_controls') else ["Implement role-based access control", "Set up automatic logoff"],
            last_checked=datetime.utcnow().isoformat()
        ))
        
        # Audit controls
        checks.append(ComplianceCheck(
            regulation="HIPAA",
            requirement="164.312(b) - Audit Controls",
            status="compliant" if config.get('audit_logging', False) else "non_compliant",
            details="Hardware, software, and procedural mechanisms that record and examine access to PHI",
            evidence=["Audit logging system active"] if config.get('audit_logging') else [],
            remediation_steps=[] if config.get('audit_logging') else ["Enable comprehensive audit logging", "Set up audit log monitoring"],
            last_checked=datetime.utcnow().isoformat()
        ))
        
        # Data integrity
        checks.append(ComplianceCheck(
            regulation="HIPAA",
            requirement="164.312(c)(1) - Integrity",
            status="compliant" if config.get('data_integrity_controls', False) else "warning",
            details="PHI must not be improperly altered or destroyed",
            evidence=["Data integrity mechanisms in place"] if config.get('data_integrity_controls') else [],
            remediation_steps=[] if config.get('data_integrity_controls') else ["Implement data checksums", "Set up backup verification"],
            last_checked=datetime.utcnow().isoformat()
        ))
        
        return checks
    
    def _soc2_checks(self, config: Dict[str, Any]) -> List[ComplianceCheck]:
        """SOC 2 compliance checks."""
        checks = []
        
        # Security principle
        checks.append(ComplianceCheck(
            regulation="SOC2",
            requirement="Security - Access Controls",
            status="compliant" if config.get('access_controls', False) else "non_compliant",
            details="Logical and physical access controls to protect against unauthorized access",
            evidence=["Access control system implemented"] if config.get('access_controls') else [],
            remediation_steps=[] if config.get('access_controls') else ["Implement multi-factor authentication", "Set up access reviews"],
            last_checked=datetime.utcnow().isoformat()
        ))
        
        # Availability principle
        checks.append(ComplianceCheck(
            regulation="SOC2",
            requirement="Availability - System Monitoring",
            status="compliant" if config.get('system_monitoring', False) else "warning",
            details="System availability monitoring and incident response procedures",
            evidence=["Monitoring system active"] if config.get('system_monitoring') else [],
            remediation_steps=[] if config.get('system_monitoring') else ["Set up system monitoring", "Create incident response plan"],
            last_checked=datetime.utcnow().isoformat()
        ))
        
        return checks
    
    def _pci_dss_checks(self, config: Dict[str, Any]) -> List[ComplianceCheck]:
        """PCI DSS compliance checks."""
        checks = []
        
        # Network security
        checks.append(ComplianceCheck(
            regulation="PCI_DSS",
            requirement="Requirement 1 - Firewall Configuration",
            status="compliant" if config.get('firewall_enabled', False) else "non_compliant",
            details="Install and maintain a firewall configuration to protect cardholder data",
            evidence=["Firewall configuration verified"] if config.get('firewall_enabled') else [],
            remediation_steps=[] if config.get('firewall_enabled') else ["Configure firewall rules", "Regular firewall review"],
            last_checked=datetime.utcnow().isoformat()
        ))
        
        # Encryption
        checks.append(ComplianceCheck(
            regulation="PCI_DSS",
            requirement="Requirement 4 - Encrypt Data Transmission",
            status="compliant" if config.get('encryption_in_transit', False) else "non_compliant",
            details="Encrypt transmission of cardholder data across open, public networks",
            evidence=["TLS encryption enabled"] if config.get('encryption_in_transit') else [],
            remediation_steps=[] if config.get('encryption_in_transit') else ["Enable TLS 1.3", "Implement proper certificate management"],
            last_checked=datetime.utcnow().isoformat()
        ))
        
        return checks


class VulnerabilityScanner:
    """Automated vulnerability scanning and assessment."""
    
    def __init__(self):
        self.known_vulnerabilities = self._load_vulnerability_database()
    
    def _load_vulnerability_database(self) -> Dict[str, Any]:
        """Load vulnerability database (simplified)."""
        return {
            'weak_passwords': {
                'severity': 'high',
                'description': 'Weak password policy detected',
                'mitigation': 'Implement strong password requirements'
            },
            'outdated_dependencies': {
                'severity': 'medium',
                'description': 'Outdated software dependencies detected',
                'mitigation': 'Update dependencies to latest versions'
            },
            'unencrypted_data': {
                'severity': 'critical',
                'description': 'Unencrypted sensitive data detected',
                'mitigation': 'Implement data encryption'
            },
            'excessive_permissions': {
                'severity': 'medium',
                'description': 'Excessive user permissions detected',
                'mitigation': 'Review and reduce user permissions'
            }
        }
    
    def scan_system(self, system_config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive vulnerability scan."""
        vulnerabilities = []
        
        # Password policy check
        password_policy = system_config.get('password_policy', {})
        if (password_policy.get('min_length', 0) < 8 or 
            not password_policy.get('require_special_chars', False)):
            vulnerabilities.append({
                'type': 'weak_passwords',
                'severity': 'high',
                'details': 'Password policy does not meet security standards',
                'location': 'Authentication system'
            })
        
        # Dependency check
        dependencies = system_config.get('dependencies', [])
        if self._check_outdated_dependencies(dependencies):
            vulnerabilities.append({
                'type': 'outdated_dependencies',
                'severity': 'medium',
                'details': 'Some dependencies are outdated and may contain vulnerabilities',
                'location': 'Application dependencies'
            })
        
        # Encryption check
        if not system_config.get('encryption_at_rest', False):
            vulnerabilities.append({
                'type': 'unencrypted_data',
                'severity': 'critical',
                'details': 'Sensitive data is not encrypted at rest',
                'location': 'Database/File system'
            })
        
        # Permission check
        user_permissions = system_config.get('user_permissions', {})
        if self._check_excessive_permissions(user_permissions):
            vulnerabilities.append({
                'type': 'excessive_permissions',
                'severity': 'medium',
                'details': 'Some users have excessive permissions',
                'location': 'Access control system'
            })
        
        # Calculate overall risk score
        risk_score = self._calculate_vulnerability_risk_score(vulnerabilities)
        
        return {
            'scan_timestamp': datetime.utcnow().isoformat(),
            'vulnerabilities_found': len(vulnerabilities),
            'vulnerabilities': vulnerabilities,
            'risk_score': risk_score,
            'recommendations': self._generate_vulnerability_recommendations(vulnerabilities)
        }
    
    def _check_outdated_dependencies(self, dependencies: List[str]) -> bool:
        """Check for outdated dependencies."""
        # Simplified check - in practice would check against CVE database
        return len(dependencies) > 0 and hash(str(dependencies)) % 3 == 0
    
    def _check_excessive_permissions(self, permissions: Dict[str, List[str]]) -> bool:
        """Check for excessive user permissions."""
        # Simplified check - look for admin permissions
        for user, perms in permissions.items():
            if 'admin' in perms or 'root' in perms:
                return True
        return False
    
    def _calculate_vulnerability_risk_score(self, vulnerabilities: List[Dict]) -> float:
        """Calculate overall risk score based on vulnerabilities."""
        if not vulnerabilities:
            return 0.0
        
        severity_weights = {
            'critical': 1.0,
            'high': 0.8,
            'medium': 0.5,
            'low': 0.2
        }
        
        total_weight = sum(severity_weights.get(vuln['severity'], 0.2) for vuln in vulnerabilities)
        return min(1.0, total_weight / len(vulnerabilities))
    
    def _generate_vulnerability_recommendations(self, vulnerabilities: List[Dict]) -> List[str]:
        """Generate recommendations based on vulnerabilities."""
        recommendations = []
        
        if any(v['type'] == 'unencrypted_data' for v in vulnerabilities):
            recommendations.append("Implement comprehensive data encryption strategy")
        
        if any(v['type'] == 'weak_passwords' for v in vulnerabilities):
            recommendations.append("Strengthen password policy and implement MFA")
        
        if any(v['type'] == 'outdated_dependencies' for v in vulnerabilities):
            recommendations.append("Establish regular dependency update schedule")
        
        if any(v['type'] == 'excessive_permissions' for v in vulnerabilities):
            recommendations.append("Conduct access review and implement least privilege principle")
        
        recommendations.append("Schedule regular security assessments")
        recommendations.append("Implement continuous security monitoring")
        
        return recommendations


class RealTimeSecurityMonitor:
    """Real-time security monitoring with streaming threat detection and alerting."""
    
    def __init__(self, alert_threshold: float = 0.7):
        self.alert_threshold = alert_threshold
        self.active_incidents = {}
        self.alert_subscribers = []
        self.correlation_rules = self._load_correlation_rules()
        self.event_queue = deque(maxlen=10000)
        self.running = False
        self._lock = threading.Lock()
    
    def start_monitoring(self):
        """Start real-time monitoring."""
        self.running = True
        monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        monitor_thread.start()
        logger.info("Real-time security monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.running = False
        logger.info("Real-time security monitoring stopped")
    
    def add_alert_subscriber(self, callback: Callable[[Dict[str, Any]], None]):
        """Add alert subscriber callback."""
        self.alert_subscribers.append(callback)
    
    def process_security_event(self, event: Dict[str, Any]):
        """Process incoming security event."""
        with self._lock:
            self.event_queue.append({
                **event,
                'timestamp': datetime.utcnow(),
                'processed': False
            })
    
    def _monitoring_loop(self):
        """Main monitoring loop for real-time threat detection."""
        while self.running:
            try:
                # Process queued events
                events_to_process = []
                with self._lock:
                    for event in self.event_queue:
                        if not event.get('processed', False):
                            events_to_process.append(event)
                            event['processed'] = True
                
                for event in events_to_process:
                    self._analyze_event(event)
                
                # Check for incident correlation
                self._correlate_incidents()
                
                time.sleep(1)  # Process events every second
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
    
    def _analyze_event(self, event: Dict[str, Any]):
        """Analyze individual security event."""
        risk_score = event.get('risk_score', 0.0)
        event_type = event.get('event_type', 'unknown')
        
        # Generate alerts for high-risk events
        if risk_score >= self.alert_threshold:
            alert = {
                'alert_id': secrets.token_hex(8),
                'timestamp': datetime.utcnow(),
                'event_type': event_type,
                'risk_score': risk_score,
                'source_event': event,
                'severity': self._determine_alert_severity(risk_score),
                'recommended_actions': self._get_recommended_actions(event_type)
            }
            
            self._send_alert(alert)
            metrics.record_error('security_alert', 'realtime_monitor')
    
    def _correlate_incidents(self):
        """Correlate multiple events to identify potential incidents."""
        # Group recent events by source IP
        recent_events = [e for e in self.event_queue if 
                        datetime.utcnow() - e['timestamp'] < timedelta(minutes=5)]
        
        events_by_ip = defaultdict(list)
        for event in recent_events:
            source_ip = event.get('source_ip')
            if source_ip:
                events_by_ip[source_ip].append(event)
        
        # Look for correlation patterns
        for source_ip, ip_events in events_by_ip.items():
            if len(ip_events) >= 3:  # Multiple events from same IP
                self._create_correlated_incident(source_ip, ip_events)
    
    def _create_correlated_incident(self, source_ip: str, events: List[Dict[str, Any]]):
        """Create incident from correlated events."""
        if source_ip not in self.active_incidents:
            incident = SecurityIncident(
                incident_id=f"INC_{secrets.token_hex(8)}",
                severity=IncidentSeverity.HIGH,
                category="correlated_attack",
                title=f"Multiple security events from {source_ip}",
                description=f"Detected {len(events)} correlated security events from IP {source_ip}",
                affected_assets=[source_ip],
                detection_timestamp=datetime.utcnow(),
                status="open",
                evidence=[{'events': events}],
                timeline=[{
                    'timestamp': datetime.utcnow(),
                    'action': 'incident_created',
                    'details': f'{len(events)} events correlated'
                }]
            )
            
            self.active_incidents[source_ip] = incident
            logger.warning(f"Created correlated incident for IP {source_ip}")
    
    def _load_correlation_rules(self) -> List[Dict[str, Any]]:
        """Load event correlation rules."""
        return [
            {
                'name': 'multiple_failed_auth',
                'pattern': 'authentication_failure',
                'threshold': 5,
                'window': 300,  # 5 minutes
                'action': 'create_incident'
            },
            {
                'name': 'sql_injection_sequence',
                'pattern': 'sql_injection',
                'threshold': 2,
                'window': 60,   # 1 minute
                'action': 'escalate_alert'
            }
        ]
    
    def _determine_alert_severity(self, risk_score: float) -> str:
        """Determine alert severity based on risk score."""
        if risk_score >= 0.9:
            return "critical"
        elif risk_score >= 0.7:
            return "high"
        elif risk_score >= 0.5:
            return "medium"
        else:
            return "low"
    
    def _get_recommended_actions(self, event_type: str) -> List[str]:
        """Get recommended actions for event type."""
        actions = {
            'sql_injection': [
                'Block source IP immediately',
                'Review database query logs',
                'Verify input validation'
            ],
            'authentication_failure': [
                'Monitor for brute force patterns',
                'Check account lockout policies',
                'Review access logs'
            ],
            'behavioral_anomaly': [
                'Investigate user behavior',
                'Review recent activity patterns',
                'Consider additional authentication'
            ]
        }
        
        return actions.get(event_type, ['Investigate security event', 'Review security logs'])
    
    def _send_alert(self, alert: Dict[str, Any]):
        """Send alert to subscribers."""
        for callback in self.alert_subscribers:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")


class AdvancedDataProtection:
    """Advanced data protection with field-level encryption and privacy-preserving techniques."""
    
    def __init__(self, encryption_provider: AdvancedEncryption = None):
        self.encryption = encryption_provider or AdvancedEncryption()
        self.tokenization_map = {}
        self.data_classification = {}
        self.privacy_policies = {}
        self._lock = threading.Lock()
    
    def classify_data(self, data: Dict[str, Any], classification_rules: Dict[str, str] = None) -> Dict[str, str]:
        """Classify data fields based on sensitivity."""
        if not classification_rules:
            classification_rules = self._get_default_classification_rules()
        
        classifications = {}
        
        for field_name, field_value in data.items():
            field_class = "public"  # Default classification
            
            # Check classification rules
            for rule, classification in classification_rules.items():
                if rule.lower() in field_name.lower():
                    field_class = classification
                    break
            
            # Additional heuristics
            if isinstance(field_value, str):
                if self._looks_like_email(field_value):
                    field_class = "pii"
                elif self._looks_like_credit_card(field_value):
                    field_class = "sensitive"
                elif self._looks_like_ssn(field_value):
                    field_class = "confidential"
            
            classifications[field_name] = field_class
        
        return classifications
    
    def apply_data_protection(self, data: Dict[str, Any], 
                            protection_policy: Dict[str, str] = None) -> Dict[str, Any]:
        """Apply data protection based on classification and policy."""
        if not protection_policy:
            protection_policy = self._get_default_protection_policy()
        
        # Classify data first
        classifications = self.classify_data(data)
        protected_data = data.copy()
        
        for field_name, classification in classifications.items():
            if field_name in protected_data:
                protection_method = protection_policy.get(classification, "none")
                
                if protection_method == "encrypt":
                    protected_data[field_name] = self.encryption.encrypt_data(str(protected_data[field_name]))
                    protected_data[f"{field_name}_protected"] = "encrypted"
                    
                elif protection_method == "tokenize":
                    token, encrypted_value = self.encryption.tokenize_sensitive_data(str(protected_data[field_name]))
                    with self._lock:
                        self.tokenization_map[token] = encrypted_value
                    protected_data[field_name] = token
                    protected_data[f"{field_name}_protected"] = "tokenized"
                    
                elif protection_method == "hash":
                    hashed_value, salt = self.encryption.hash_sensitive_data(str(protected_data[field_name]))
                    protected_data[field_name] = hashed_value
                    protected_data[f"{field_name}_salt"] = salt
                    protected_data[f"{field_name}_protected"] = "hashed"
                    
                elif protection_method == "mask":
                    protected_data[field_name] = self._mask_value(str(protected_data[field_name]))
                    protected_data[f"{field_name}_protected"] = "masked"
        
        return protected_data
    
    def remove_data_protection(self, protected_data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove data protection to restore original values."""
        restored_data = protected_data.copy()
        
        # Find protected fields
        protected_fields = [k.replace('_protected', '') for k in protected_data.keys() 
                           if k.endswith('_protected')]
        
        for field_name in protected_fields:
            if field_name in restored_data:
                protection_method = protected_data.get(f"{field_name}_protected")
                
                if protection_method == "encrypted":
                    restored_data[field_name] = self.encryption.decrypt_data(restored_data[field_name])
                    
                elif protection_method == "tokenized":
                    token = restored_data[field_name]
                    with self._lock:
                        if token in self.tokenization_map:
                            encrypted_value = self.tokenization_map[token]
                            restored_data[field_name] = self.encryption.decrypt_data(encrypted_value)
                
                # Remove protection metadata
                restored_data.pop(f"{field_name}_protected", None)
                restored_data.pop(f"{field_name}_salt", None)
        
        return restored_data
    
    def apply_differential_privacy(self, data: List[Dict[str, Any]], 
                                 epsilon: float = 1.0, fields: List[str] = None) -> List[Dict[str, Any]]:
        """Apply differential privacy to protect individual privacy in datasets."""
        if not fields:
            return data
        
        private_data = []
        
        for record in data:
            private_record = record.copy()
            
            for field in fields:
                if field in private_record and isinstance(private_record[field], (int, float)):
                    # Add Laplace noise for differential privacy
                    noise = np.random.laplace(0, 1/epsilon)
                    private_record[field] = private_record[field] + noise
            
            private_data.append(private_record)
        
        return private_data
    
    def federated_learning_aggregation(self, model_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate model updates from federated learning while preserving privacy."""
        if not model_updates:
            return {}
        
        # Simple federated averaging (in practice would be more sophisticated)
        aggregated = {}
        
        # Get all parameter names
        all_params = set()
        for update in model_updates:
            all_params.update(update.get('parameters', {}).keys())
        
        # Average parameters
        for param_name in all_params:
            param_values = [
                update['parameters'].get(param_name, 0) 
                for update in model_updates 
                if 'parameters' in update and param_name in update['parameters']
            ]
            
            if param_values:
                aggregated[param_name] = sum(param_values) / len(param_values)
        
        return {
            'aggregated_parameters': aggregated,
            'num_contributors': len(model_updates),
            'aggregation_timestamp': datetime.utcnow().isoformat()
        }
    
    def _get_default_classification_rules(self) -> Dict[str, str]:
        """Get default data classification rules."""
        return {
            'password': 'confidential',
            'ssn': 'confidential',
            'social_security': 'confidential',
            'credit_card': 'sensitive',
            'card_number': 'sensitive',
            'email': 'pii',
            'phone': 'pii',
            'address': 'pii',
            'name': 'pii',
            'ip_address': 'internal',
            'user_id': 'internal'
        }
    
    def _get_default_protection_policy(self) -> Dict[str, str]:
        """Get default data protection policy."""
        return {
            'confidential': 'encrypt',
            'sensitive': 'tokenize',
            'pii': 'hash',
            'internal': 'mask',
            'public': 'none'
        }
    
    def _looks_like_email(self, value: str) -> bool:
        """Check if value looks like an email address."""
        return '@' in value and '.' in value.split('@')[-1]
    
    def _looks_like_credit_card(self, value: str) -> bool:
        """Check if value looks like a credit card number."""
        cleaned = ''.join(c for c in value if c.isdigit())
        return len(cleaned) in [13, 14, 15, 16] and cleaned.isdigit()
    
    def _looks_like_ssn(self, value: str) -> bool:
        """Check if value looks like a social security number."""
        cleaned = ''.join(c for c in value if c.isdigit())
        return len(cleaned) == 9 and cleaned.isdigit()
    
    def _mask_value(self, value: str) -> str:
        """Mask sensitive value."""
        if len(value) <= 4:
            return '*' * len(value)
        return value[:2] + '*' * (len(value) - 4) + value[-2:]


class SecurityPolicyEngine:
    """Dynamic security policy engine with rule-based enforcement."""
    
    def __init__(self):
        self.policies = {}
        self.policy_cache = {}
        self.policy_history = []
        self.enforcement_stats = defaultdict(int)
        self._lock = threading.Lock()
    
    def create_policy(self, policy: SecurityPolicy) -> bool:
        """Create new security policy."""
        try:
            with self._lock:
                self.policies[policy.policy_id] = policy
                self.policy_history.append({
                    'action': 'created',
                    'policy_id': policy.policy_id,
                    'timestamp': datetime.utcnow(),
                    'details': {'name': policy.name, 'category': policy.category}
                })
            
            logger.info(f"Created security policy: {policy.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create policy: {e}")
            return False
    
    def update_policy(self, policy_id: str, updates: Dict[str, Any]) -> bool:
        """Update existing security policy."""
        try:
            with self._lock:
                if policy_id not in self.policies:
                    return False
                
                policy = self.policies[policy_id]
                
                # Update policy fields
                for field, value in updates.items():
                    if hasattr(policy, field):
                        setattr(policy, field, value)
                
                policy.last_updated = datetime.utcnow()
                policy.version = self._increment_version(policy.version)
                
                # Clear cache for this policy
                self.policy_cache.pop(policy_id, None)
                
                self.policy_history.append({
                    'action': 'updated',
                    'policy_id': policy_id,
                    'timestamp': datetime.utcnow(),
                    'details': updates
                })
            
            logger.info(f"Updated security policy: {policy_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update policy: {e}")
            return False
    
    def enforce_policy(self, policy_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enforce security policy against given context."""
        try:
            policy = self.get_policy(policy_id)
            if not policy or not policy.is_active:
                return {'allowed': True, 'reason': 'No active policy'}
            
            # Check if policy applies to context
            if not self._policy_applies(policy, context):
                return {'allowed': True, 'reason': 'Policy not applicable'}
            
            # Evaluate policy rules
            enforcement_result = self._evaluate_policy_rules(policy, context)
            
            # Record enforcement statistics
            with self._lock:
                self.enforcement_stats[f"{policy_id}_evaluations"] += 1
                if not enforcement_result['allowed']:
                    self.enforcement_stats[f"{policy_id}_violations"] += 1
            
            return enforcement_result
            
        except Exception as e:
            logger.error(f"Policy enforcement failed: {e}")
            return {'allowed': False, 'reason': f'Enforcement error: {e}'}
    
    def get_policy(self, policy_id: str) -> Optional[SecurityPolicy]:
        """Get security policy by ID."""
        return self.policies.get(policy_id)
    
    def list_policies(self, category: str = None, active_only: bool = True) -> List[SecurityPolicy]:
        """List security policies."""
        policies = list(self.policies.values())
        
        if category:
            policies = [p for p in policies if p.category == category]
        
        if active_only:
            policies = [p for p in policies if p.is_active]
        
        return policies
    
    def get_policy_compliance_report(self) -> Dict[str, Any]:
        """Generate policy compliance report."""
        report = {
            'total_policies': len(self.policies),
            'active_policies': len([p for p in self.policies.values() if p.is_active]),
            'categories': defaultdict(int),
            'enforcement_stats': dict(self.enforcement_stats),
            'recent_violations': [],
            'compliance_score': 0.0
        }
        
        # Category breakdown
        for policy in self.policies.values():
            report['categories'][policy.category] += 1
        
        # Recent violations from history
        recent_violations = [
            event for event in self.policy_history[-100:]  # Last 100 events
            if 'violation' in event.get('action', '').lower()
        ]
        report['recent_violations'] = recent_violations[:10]  # Top 10
        
        # Calculate compliance score
        total_evaluations = sum(
            count for key, count in self.enforcement_stats.items()
            if key.endswith('_evaluations')
        )
        total_violations = sum(
            count for key, count in self.enforcement_stats.items()
            if key.endswith('_violations')
        )
        
        if total_evaluations > 0:
            report['compliance_score'] = 1.0 - (total_violations / total_evaluations)
        
        return report
    
    def _policy_applies(self, policy: SecurityPolicy, context: Dict[str, Any]) -> bool:
        """Check if policy applies to given context."""
        # Check applicable_to field
        if policy.applicable_to:
            context_type = context.get('type', 'unknown')
            if context_type not in policy.applicable_to:
                return False
        
        return True
    
    def _evaluate_policy_rules(self, policy: SecurityPolicy, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate policy rules against context."""
        result = {'allowed': True, 'violations': [], 'warnings': []}
        
        for rule in policy.rules:
            rule_result = self._evaluate_single_rule(rule, context)
            
            if not rule_result['passed']:
                violation = {
                    'rule': rule.get('name', 'unnamed'),
                    'reason': rule_result['reason'],
                    'severity': rule.get('severity', 'medium')
                }
                
                if policy.enforcement_level == 'blocking':
                    result['allowed'] = False
                    result['violations'].append(violation)
                elif policy.enforcement_level == 'warning':
                    result['warnings'].append(violation)
        
        return result
    
    def _evaluate_single_rule(self, rule: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate single policy rule."""
        rule_type = rule.get('type', 'unknown')
        
        if rule_type == 'time_based':
            return self._evaluate_time_rule(rule, context)
        elif rule_type == 'ip_based':
            return self._evaluate_ip_rule(rule, context)
        elif rule_type == 'user_based':
            return self._evaluate_user_rule(rule, context)
        elif rule_type == 'resource_based':
            return self._evaluate_resource_rule(rule, context)
        else:
            return {'passed': True, 'reason': 'Unknown rule type'}
    
    def _evaluate_time_rule(self, rule: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate time-based rule."""
        current_time = datetime.utcnow().time()
        allowed_start = datetime.strptime(rule.get('allowed_start', '00:00'), '%H:%M').time()
        allowed_end = datetime.strptime(rule.get('allowed_end', '23:59'), '%H:%M').time()
        
        if allowed_start <= current_time <= allowed_end:
            return {'passed': True, 'reason': 'Within allowed time window'}
        else:
            return {'passed': False, 'reason': f'Outside allowed time window: {allowed_start}-{allowed_end}'}
    
    def _evaluate_ip_rule(self, rule: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate IP-based rule."""
        source_ip = context.get('source_ip', '')
        allowed_ips = rule.get('allowed_ips', [])
        blocked_ips = rule.get('blocked_ips', [])
        
        if blocked_ips and source_ip in blocked_ips:
            return {'passed': False, 'reason': f'IP {source_ip} is blocked'}
        
        if allowed_ips and source_ip not in allowed_ips:
            return {'passed': False, 'reason': f'IP {source_ip} not in allowed list'}
        
        return {'passed': True, 'reason': 'IP validation passed'}
    
    def _evaluate_user_rule(self, rule: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate user-based rule."""
        user = context.get('user', '')
        allowed_users = rule.get('allowed_users', [])
        required_roles = rule.get('required_roles', [])
        
        if allowed_users and user not in allowed_users:
            return {'passed': False, 'reason': f'User {user} not in allowed list'}
        
        user_roles = context.get('user_roles', [])
        if required_roles and not any(role in user_roles for role in required_roles):
            return {'passed': False, 'reason': f'User lacks required roles: {required_roles}'}
        
        return {'passed': True, 'reason': 'User validation passed'}
    
    def _evaluate_resource_rule(self, rule: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate resource-based rule."""
        resource = context.get('resource', '')
        allowed_resources = rule.get('allowed_resources', [])
        
        if allowed_resources and not any(res in resource for res in allowed_resources):
            return {'passed': False, 'reason': f'Resource {resource} not allowed'}
        
        return {'passed': True, 'reason': 'Resource validation passed'}
    
    def _increment_version(self, current_version: str) -> str:
        """Increment policy version."""
        try:
            major, minor, patch = map(int, current_version.split('.'))
            return f"{major}.{minor}.{patch + 1}"
        except:
            return "1.0.1"


class AdvancedSecurityOrchestrator:
    """Main orchestrator for advanced security operations."""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_security_config(config_path)
        self.encryption = AdvancedEncryption()
        self.threat_detector = ThreatDetectionSystem()
        self.compliance_validator = ComplianceValidator()
        self.vulnerability_scanner = VulnerabilityScanner()
        self.zero_trust_auth = ZeroTrustAuthenticator()
        self.realtime_monitor = RealTimeSecurityMonitor()
        self.data_protection = AdvancedDataProtection(self.encryption)
        self.policy_engine = SecurityPolicyEngine()
        self.audit_logger = SecureAuditLogger()
        self.incident_response = IncidentResponseAutomation(self.audit_logger)
        self.vulnerability_assessment = AutomatedVulnerabilityAssessment()
        self.security_events = []
        
        # Initialize behavioral detection
        self.threat_detector.initialize_behavioral_detection()
        
        # Start real-time monitoring if enabled
        if self.config.get('realtime_monitoring_enabled', True):
            self.realtime_monitor.start_monitoring()
            self.realtime_monitor.add_alert_subscriber(self._handle_security_alert)
        
        # Log system initialization
        self.audit_logger.log_event(
            event_type="system_initialized",
            user_id="system",
            source_ip="127.0.0.1",
            resource="advanced_security_orchestrator",
            action="initialize",
            result="success",
            details={"components": ["threat_detection", "zero_trust_auth", "realtime_monitor", "data_protection", "policy_engine", "audit_logger", "incident_response", "vulnerability_assessment"]}
        )
    
    def _handle_security_alert(self, alert: Dict[str, Any]):
        """Handle security alerts from real-time monitor."""
        logger.warning(f"Security Alert: {alert['event_type']} - Severity: {alert['severity']} - Risk Score: {alert['risk_score']}")
        
        # Record in security events
        self.security_events.append({
            'event_type': 'security_alert',
            'timestamp': alert['timestamp'].isoformat() if isinstance(alert['timestamp'], datetime) else alert['timestamp'],
            'alert_id': alert['alert_id'],
            'severity': alert['severity'],
            'risk_score': alert['risk_score'],
            'recommended_actions': alert['recommended_actions']
        })
        
        # Record metrics
        metrics.record_error('security_alert_generated', 'advanced_security_orchestrator')
    
    def authenticate_user(self, username: str, password: str, source_ip: str, 
                         user_agent: str, additional_factors: Dict[str, Any] = None) -> AuthenticationAttempt:
        """Perform zero-trust authentication."""
        auth_result = self.zero_trust_auth.authenticate(
            username, password, source_ip, user_agent, additional_factors
        )
        
        # Process authentication event through real-time monitor
        auth_event = {
            'event_type': 'authentication',
            'source_ip': source_ip,
            'username': username,
            'status': auth_result.status.value,
            'risk_score': auth_result.risk_score
        }
        self.realtime_monitor.process_security_event(auth_event)
        
        return auth_result
    
    def apply_data_protection_policy(self, data: Dict[str, Any], 
                                   policy_name: str = None) -> Dict[str, Any]:
        """Apply data protection policies to sensitive data."""
        return self.data_protection.apply_data_protection(data)
    
    def enforce_security_policy(self, policy_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enforce security policy for given context."""
        return self.policy_engine.enforce_policy(policy_id, context)
    
    def create_security_incident(self, title: str, description: str, severity: str, 
                               affected_assets: List[str]) -> str:
        """Create security incident with automated response."""
        severity_enum = IncidentSeverity(severity.lower())
        incident_id = self.incident_response.create_incident(
            title, description, severity_enum, affected_assets
        )
        
        # Log incident creation
        self.audit_logger.log_event(
            event_type="security_incident_created",
            user_id="orchestrator",
            source_ip="127.0.0.1",
            resource="security_orchestrator",
            action="create_incident",
            result="success",
            details={"incident_id": incident_id, "title": title, "severity": severity}
        )
        
        return incident_id
    
    def scan_project_vulnerabilities(self, project_path: str) -> Dict[str, Any]:
        """Scan project for vulnerabilities and generate remediation plan."""
        vulnerabilities = self.vulnerability_assessment.scan_dependencies(project_path)
        remediation_plan = self.vulnerability_assessment.generate_remediation_plan(vulnerabilities)
        
        # Log vulnerability scan
        self.audit_logger.log_event(
            event_type="vulnerability_scan",
            user_id="orchestrator",
            source_ip="127.0.0.1",
            resource="security_orchestrator",
            action="scan_vulnerabilities",
            result="success",
            details={
                "project_path": project_path, 
                "vulnerabilities_found": len(vulnerabilities),
                "critical_count": remediation_plan['critical_count'],
                "high_count": remediation_plan['high_count']
            }
        )
        
        return {
            "vulnerabilities": [asdict(v) for v in vulnerabilities],
            "remediation_plan": remediation_plan
        }
    
    def verify_audit_integrity(self) -> Dict[str, Any]:
        """Verify integrity of audit logs."""
        return self.audit_logger.verify_log_integrity()
    
    def get_comprehensive_security_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive security dashboard data."""
        dashboard_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "threat_detection": {
                "total_threats": len(self.threat_detector.threat_history),
                "recent_threats": len([t for t in self.threat_detector.threat_history 
                                     if (datetime.utcnow() - datetime.fromisoformat(t.timestamp.replace('Z', '+00:00'))).days < 1]),
                "behavioral_model_trained": getattr(self.threat_detector, 'model_trained', False)
            },
            "authentication": {
                "blocked_ips": len(self.zero_trust_auth.blocked_ips),
                "active_sessions": len(self.zero_trust_auth.session_store),
                "failed_attempts": sum(len(attempts) for attempts in self.zero_trust_auth.failed_attempts.values())
            },
            "incidents": self.incident_response.get_incident_metrics(),
            "policies": {
                "total_policies": len(self.policy_engine.policies),
                "active_policies": len([p for p in self.policy_engine.policies.values() if p.is_active]),
                "enforcement_stats": dict(self.policy_engine.enforcement_stats)
            },
            "audit_logs": {
                "total_entries": len(self.audit_logger.log_entries),
                "integrity_status": self.audit_logger.verify_log_integrity()['valid']
            },
            "real_time_monitoring": {
                "active_incidents": len(self.realtime_monitor.active_incidents),
                "events_queued": len(self.realtime_monitor.event_queue),
                "monitoring_active": self.realtime_monitor.running
            },
            "vulnerability_assessment": {
                "scans_performed": len(self.vulnerability_assessment.scan_history),
                "cve_cache_size": len(self.vulnerability_assessment.cve_cache)
            },
            "system_metrics": self.get_security_metrics()
        }
        
        return dashboard_data
        
    def _load_security_config(self, config_path: str) -> Dict[str, Any]:
        """Load security configuration."""
        default_config = {
            'threat_detection_enabled': True,
            'compliance_checks': ['GDPR', 'SOC2'],
            'vulnerability_scan_interval': 86400,  # Daily
            'encryption_at_rest': True,
            'encryption_in_transit': True,
            'audit_logging': True,
            'access_controls': True,
            'password_policy': {
                'min_length': 12,
                'require_special_chars': True,
                'require_numbers': True,
                'require_uppercase': True
            },
            'session_timeout': 3600,  # 1 hour
            'max_failed_attempts': 3,
            'security_headers_enabled': True
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def perform_security_audit(self) -> SecurityAuditResult:
        """Perform comprehensive security audit."""
        logger.info("Starting comprehensive security audit...")
        audit_start = time.time()
        
        audit_id = hashlib.sha256(f"audit_{time.time()}".encode()).hexdigest()[:16]
        
        # Threat detection summary
        recent_threats = [
            t for t in self.threat_detector.threat_history
            if (datetime.utcnow() - datetime.fromisoformat(t.timestamp.replace('Z', '+00:00'))).days < 1
        ]
        
        # Compliance validation
        compliance_results = []
        for regulation in self.config.get('compliance_checks', []):
            try:
                compliance_checks = self.compliance_validator.validate_compliance(regulation, self.config)
                compliance_results.extend(compliance_checks)
            except Exception as e:
                logger.error(f"Compliance check failed for {regulation}: {e}")
        
        # Vulnerability scanning
        vulnerability_scan = self.vulnerability_scanner.scan_system(self.config)
        
        # Access control audit
        access_control_audit = self._audit_access_controls()
        
        # Data protection audit
        data_protection_audit = self._audit_data_protection()
        
        # Calculate overall security score
        overall_score = self._calculate_security_score(
            recent_threats, compliance_results, vulnerability_scan
        )
        
        # Generate recommendations
        recommendations = self._generate_security_recommendations(
            recent_threats, compliance_results, vulnerability_scan
        )
        
        # Create audit result
        audit_result = SecurityAuditResult(
            audit_id=audit_id,
            audit_timestamp=datetime.utcnow().isoformat(),
            overall_score=overall_score,
            threats_detected=recent_threats,
            compliance_status=compliance_results,
            vulnerability_scan=vulnerability_scan,
            access_control_audit=access_control_audit,
            data_protection_audit=data_protection_audit,
            recommendations=recommendations,
            next_audit_date=(datetime.utcnow() + timedelta(days=7)).isoformat()
        )
        
        # Log audit completion
        audit_duration = time.time() - audit_start
        logger.info(f"Security audit completed in {audit_duration:.2f}s. Overall score: {overall_score:.2f}")
        
        # Record security event
        self.security_events.append({
            'event_type': 'security_audit',
            'timestamp': datetime.utcnow().isoformat(),
            'details': {
                'audit_id': audit_id,
                'overall_score': overall_score,
                'threats_count': len(recent_threats),
                'vulnerabilities_count': vulnerability_scan['vulnerabilities_found']
            }
        })
        
        return audit_result
    
    def _audit_access_controls(self) -> Dict[str, Any]:
        """Audit access control systems."""
        return {
            'mfa_enabled': self.config.get('mfa_enabled', False),
            'session_timeout': self.config.get('session_timeout', 3600),
            'password_policy_compliant': self._validate_password_policy(),
            'role_based_access': self.config.get('role_based_access', False),
            'access_review_date': self.config.get('last_access_review', 'Never'),
            'privileged_accounts': self.config.get('privileged_accounts_count', 0)
        }
    
    def _audit_data_protection(self) -> Dict[str, Any]:
        """Audit data protection measures."""
        return {
            'encryption_at_rest': self.config.get('encryption_at_rest', False),
            'encryption_in_transit': self.config.get('encryption_in_transit', False),
            'key_management': self.config.get('key_management_system', False),
            'data_classification': self.config.get('data_classification_implemented', False),
            'backup_encryption': self.config.get('backup_encryption', False),
            'data_loss_prevention': self.config.get('dlp_enabled', False)
        }
    
    def _validate_password_policy(self) -> bool:
        """Validate password policy compliance."""
        policy = self.config.get('password_policy', {})
        return (
            policy.get('min_length', 0) >= 12 and
            policy.get('require_special_chars', False) and
            policy.get('require_numbers', False) and
            policy.get('require_uppercase', False)
        )
    
    def _calculate_security_score(self, threats: List[SecurityThreat],
                                compliance_results: List[ComplianceCheck],
                                vulnerability_scan: Dict[str, Any]) -> float:
        """Calculate overall security score."""
        
        # Base score
        score = 1.0
        
        # Deduct for active threats
        critical_threats = [t for t in threats if t.severity == 'critical']
        high_threats = [t for t in threats if t.severity == 'high']
        
        score -= len(critical_threats) * 0.2
        score -= len(high_threats) * 0.1
        
        # Deduct for compliance issues
        non_compliant_checks = [c for c in compliance_results if c.status == 'non_compliant']
        score -= len(non_compliant_checks) * 0.1
        
        # Deduct for vulnerabilities
        score -= vulnerability_scan.get('risk_score', 0) * 0.3
        
        return max(0.0, score)
    
    def _generate_security_recommendations(self, threats: List[SecurityThreat],
                                         compliance_results: List[ComplianceCheck],
                                         vulnerability_scan: Dict[str, Any]) -> List[str]:
        """Generate security recommendations."""
        recommendations = []
        
        # Threat-based recommendations
        if any(t.severity == 'critical' for t in threats):
            recommendations.append("URGENT: Address critical security threats immediately")
        
        # Compliance recommendations
        non_compliant = [c for c in compliance_results if c.status == 'non_compliant']
        if non_compliant:
            recommendations.append(f"Address {len(non_compliant)} compliance violations")
        
        # Vulnerability recommendations
        recommendations.extend(vulnerability_scan.get('recommendations', []))
        
        # General recommendations
        if not self.config.get('mfa_enabled', False):
            recommendations.append("Enable multi-factor authentication")
        
        if self.config.get('session_timeout', 3600) > 3600:
            recommendations.append("Reduce session timeout to maximum 1 hour")
        
        recommendations.append("Conduct regular security training for staff")
        recommendations.append("Implement security incident response plan")
        
        return recommendations
    
    def monitor_request(self, request_data: Dict[str, Any]) -> List[SecurityThreat]:
        """Monitor and analyze incoming request for threats with enhanced detection."""
        if not self.config.get('threat_detection_enabled', True):
            return []
        
        # Use enhanced threat detection with behavioral analysis
        threats = self.threat_detector.analyze_request_enhanced(request_data)
        
        # Process threats through real-time monitor
        for threat in threats:
            threat_event = {
                'event_type': 'threat_detected',
                'source_ip': threat.source_ip,
                'threat_category': threat.category,
                'risk_score': threat.risk_score
            }
            self.realtime_monitor.process_security_event(threat_event)
        
        # Log critical threats
        for threat in threats:
            if threat.severity in ['critical', 'high']:
                logger.warning(f"Security threat detected: {threat.description} from {threat.source_ip}")
                
                # Enhanced threat with ML classification
                enhanced_threat = self.threat_detector.classify_threat_with_ml(threat)
                
                # Record security event
                self.security_events.append({
                    'event_type': 'threat_detected',
                    'timestamp': datetime.utcnow().isoformat(),
                    'threat_id': enhanced_threat.threat_id,
                    'severity': enhanced_threat.severity,
                    'source_ip': enhanced_threat.source_ip,
                    'category': enhanced_threat.category,
                    'risk_score': enhanced_threat.risk_score,
                    'mitigation_actions': enhanced_threat.mitigation_actions
                })
        
        # Update behavioral model with request data
        self.threat_detector.update_behavioral_model([request_data])
        
        return threats
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get comprehensive security metrics."""
        recent_events = [
            e for e in self.security_events
            if (datetime.utcnow() - datetime.fromisoformat(e['timestamp'])).days < 7
        ]
        
        threat_counts = {}
        for event in recent_events:
            if event['event_type'] == 'threat_detected':
                severity = event.get('severity', 'unknown')
                threat_counts[severity] = threat_counts.get(severity, 0) + 1
        
        return {
            'events_last_7_days': len(recent_events),
            'threat_counts': threat_counts,
            'total_threats_detected': len(self.threat_detector.threat_history),
            'config_status': {
                'encryption_enabled': self.config.get('encryption_at_rest', False),
                'audit_logging': self.config.get('audit_logging', False),
                'access_controls': self.config.get('access_controls', False)
            },
            'last_audit': self.security_events[-1] if self.security_events else None
        }


class SecureAuditLogger:
    """Tamper-proof audit logging with integrity verification."""
    
    def __init__(self, log_file_path: str = None, encryption_provider: AdvancedEncryption = None):
        self.log_file_path = log_file_path or os.path.join(os.getcwd(), "secure_audit.log")
        self.encryption = encryption_provider or AdvancedEncryption()
        self.log_entries = deque(maxlen=10000)
        self.current_hash = None
        self._lock = threading.Lock()
        self._initialize_log_chain()
    
    def _initialize_log_chain(self):
        """Initialize the audit log chain."""
        genesis_entry = AuditLogEntry(
            log_id="GENESIS_LOG",
            timestamp=datetime.utcnow(),
            event_type="system_initialized",
            user_id=None,
            source_ip="127.0.0.1",
            resource="audit_system",
            action="initialize",
            result="success",
            details={"message": "Secure audit logging initialized"},
            integrity_hash=hashlib.sha256("GENESIS".encode()).hexdigest(),
            previous_hash=None
        )
        
        self.current_hash = genesis_entry.integrity_hash
        self.log_entries.append(genesis_entry)
    
    def log_event(self, event_type: str, user_id: Optional[str], source_ip: str,
                  resource: str, action: str, result: str, details: Dict[str, Any] = None) -> str:
        """Log a secure audit event with tamper protection."""
        if details is None:
            details = {}
        
        with self._lock:
            log_id = f"LOG_{secrets.token_hex(8)}_{int(time.time())}"
            
            # Create log entry with previous hash for chain integrity
            entry = AuditLogEntry(
                log_id=log_id,
                timestamp=datetime.utcnow(),
                event_type=event_type,
                user_id=user_id,
                source_ip=source_ip,
                resource=resource,
                action=action,
                result=result,
                details=details,
                integrity_hash="",  # Will be calculated
                previous_hash=self.current_hash
            )
            
            # Calculate integrity hash
            entry_data = f"{entry.log_id}{entry.timestamp}{entry.event_type}{entry.user_id}{entry.source_ip}{entry.resource}{entry.action}{entry.result}{json.dumps(entry.details, sort_keys=True)}{entry.previous_hash}"
            entry.integrity_hash = hashlib.sha256(entry_data.encode()).hexdigest()
            
            # Update current hash
            self.current_hash = entry.integrity_hash
            
            # Store entry
            self.log_entries.append(entry)
            
            # Write to persistent storage
            self._write_to_file(entry)
            
            return log_id
    
    def verify_log_integrity(self) -> Dict[str, Any]:
        """Verify the integrity of the entire audit log chain."""
        verification_result = {
            'valid': True,
            'total_entries': len(self.log_entries),
            'corrupted_entries': [],
            'verification_timestamp': datetime.utcnow().isoformat()
        }
        
        previous_hash = None
        
        for i, entry in enumerate(self.log_entries):
            # Verify previous hash chain
            if entry.previous_hash != previous_hash:
                verification_result['valid'] = False
                verification_result['corrupted_entries'].append({
                    'index': i,
                    'log_id': entry.log_id,
                    'error': 'Hash chain broken'
                })
            
            # Verify entry hash
            expected_data = f"{entry.log_id}{entry.timestamp}{entry.event_type}{entry.user_id}{entry.source_ip}{entry.resource}{entry.action}{entry.result}{json.dumps(entry.details, sort_keys=True)}{entry.previous_hash}"
            expected_hash = hashlib.sha256(expected_data.encode()).hexdigest()
            
            if entry.integrity_hash != expected_hash:
                verification_result['valid'] = False
                verification_result['corrupted_entries'].append({
                    'index': i,
                    'log_id': entry.log_id,
                    'error': 'Entry hash mismatch'
                })
            
            previous_hash = entry.integrity_hash
        
        return verification_result
    
    def get_audit_trail(self, start_time: datetime = None, end_time: datetime = None,
                       event_type: str = None, user_id: str = None) -> List[AuditLogEntry]:
        """Get filtered audit trail."""
        filtered_entries = list(self.log_entries)
        
        if start_time:
            filtered_entries = [e for e in filtered_entries if e.timestamp >= start_time]
        
        if end_time:
            filtered_entries = [e for e in filtered_entries if e.timestamp <= end_time]
        
        if event_type:
            filtered_entries = [e for e in filtered_entries if e.event_type == event_type]
        
        if user_id:
            filtered_entries = [e for e in filtered_entries if e.user_id == user_id]
        
        return filtered_entries
    
    def _write_to_file(self, entry: AuditLogEntry):
        """Write audit entry to persistent storage."""
        try:
            encrypted_entry = self.encryption.encrypt_data(json.dumps(asdict(entry), default=str))
            
            with open(self.log_file_path, 'a') as f:
                f.write(f"{encrypted_entry}\n")
                
        except Exception as e:
            logger.error(f"Failed to write audit log to file: {e}")


class IncidentResponseAutomation:
    """Automated incident response system with SOAR capabilities."""
    
    def __init__(self, audit_logger: SecureAuditLogger = None):
        self.incidents = {}
        self.response_playbooks = {}
        self.automation_rules = []
        self.containment_actions = {}
        self.audit_logger = audit_logger or SecureAuditLogger()
        self._lock = threading.Lock()
        self._load_default_playbooks()
    
    def create_incident(self, title: str, description: str, severity: IncidentSeverity,
                       affected_assets: List[str], evidence: List[Dict[str, Any]] = None) -> str:
        """Create new security incident."""
        incident_id = f"INC_{datetime.utcnow().strftime('%Y%m%d')}_{secrets.token_hex(4)}"
        
        incident = SecurityIncident(
            incident_id=incident_id,
            severity=severity,
            category="security_incident",
            title=title,
            description=description,
            affected_assets=affected_assets,
            detection_timestamp=datetime.utcnow(),
            status="open",
            evidence=evidence or [],
            timeline=[{
                'timestamp': datetime.utcnow(),
                'action': 'incident_created',
                'details': f'Created by automated system: {title}'
            }]
        )
        
        with self._lock:
            self.incidents[incident_id] = incident
        
        # Log incident creation
        self.audit_logger.log_event(
            event_type="incident_created",
            user_id="system_automation",
            source_ip="127.0.0.1",
            resource="incident_response_system",
            action="create_incident",
            result="success",
            details={"incident_id": incident_id, "severity": severity.value, "title": title}
        )
        
        # Trigger automated response
        self._trigger_automated_response(incident)
        
        logger.warning(f"Security incident created: {incident_id} - {title}")
        return incident_id
    
    def update_incident_status(self, incident_id: str, status: str, 
                             assigned_to: str = None, notes: str = None) -> bool:
        """Update incident status and assignment."""
        with self._lock:
            if incident_id not in self.incidents:
                return False
            
            incident = self.incidents[incident_id]
            old_status = incident.status
            incident.status = status
            
            if assigned_to:
                incident.assigned_to = assigned_to
            
            # Add to timeline
            timeline_entry = {
                'timestamp': datetime.utcnow(),
                'action': 'status_updated',
                'details': f'Status changed from {old_status} to {status}'
            }
            
            if notes:
                timeline_entry['notes'] = notes
            
            incident.timeline.append(timeline_entry)
        
        # Log status update
        self.audit_logger.log_event(
            event_type="incident_updated",
            user_id=assigned_to or "system_automation",
            source_ip="127.0.0.1",
            resource="incident_response_system",
            action="update_status",
            result="success",
            details={"incident_id": incident_id, "old_status": old_status, "new_status": status}
        )
        
        return True
    
    def add_containment_action(self, incident_id: str, action: str, 
                             automated: bool = False, result: str = None) -> bool:
        """Add containment action to incident."""
        with self._lock:
            if incident_id not in self.incidents:
                return False
            
            incident = self.incidents[incident_id]
            
            containment_entry = {
                'timestamp': datetime.utcnow(),
                'action': action,
                'automated': automated,
                'result': result or 'pending',
                'executed_by': 'system_automation' if automated else 'manual'
            }
            
            incident.containment_actions.append(action)
            incident.timeline.append({
                'timestamp': datetime.utcnow(),
                'action': 'containment_action_added',
                'details': f'Added containment action: {action}'
            })
        
        # Log containment action
        self.audit_logger.log_event(
            event_type="containment_action",
            user_id="system_automation" if automated else "manual_operator",
            source_ip="127.0.0.1",
            resource="incident_response_system",
            action="add_containment",
            result="success",
            details={"incident_id": incident_id, "action": action, "automated": automated}
        )
        
        return True
    
    def execute_containment_playbook(self, incident_id: str, playbook_name: str) -> Dict[str, Any]:
        """Execute automated containment playbook."""
        if playbook_name not in self.response_playbooks:
            return {"success": False, "reason": "Playbook not found"}
        
        playbook = self.response_playbooks[playbook_name]
        results = {"success": True, "actions_executed": [], "failures": []}
        
        for action in playbook.get('actions', []):
            try:
                result = self._execute_containment_action(incident_id, action)
                results["actions_executed"].append({
                    "action": action,
                    "result": result
                })
                
                # Add to incident
                self.add_containment_action(incident_id, action['name'], automated=True, result="success")
                
            except Exception as e:
                error_msg = f"Failed to execute {action.get('name', 'unknown')}: {str(e)}"
                results["failures"].append(error_msg)
                results["success"] = False
                logger.error(error_msg)
        
        return results
    
    def get_incident_metrics(self) -> Dict[str, Any]:
        """Get incident response metrics."""
        incidents_by_status = defaultdict(int)
        incidents_by_severity = defaultdict(int)
        avg_response_times = {}
        
        for incident in self.incidents.values():
            incidents_by_status[incident.status] += 1
            incidents_by_severity[incident.severity.value] += 1
        
        return {
            'total_incidents': len(self.incidents),
            'incidents_by_status': dict(incidents_by_status),
            'incidents_by_severity': dict(incidents_by_severity),
            'active_incidents': len([i for i in self.incidents.values() if i.status in ['open', 'investigating']]),
            'average_response_time': avg_response_times,
            'playbooks_available': len(self.response_playbooks)
        }
    
    def _load_default_playbooks(self):
        """Load default incident response playbooks."""
        self.response_playbooks = {
            'sql_injection_response': {
                'name': 'SQL Injection Response',
                'description': 'Automated response to SQL injection attacks',
                'actions': [
                    {'name': 'block_source_ip', 'type': 'network', 'priority': 'high'},
                    {'name': 'review_database_logs', 'type': 'investigation', 'priority': 'medium'},
                    {'name': 'validate_input_filters', 'type': 'verification', 'priority': 'medium'}
                ]
            },
            'brute_force_response': {
                'name': 'Brute Force Attack Response',
                'description': 'Automated response to brute force authentication attacks',
                'actions': [
                    {'name': 'rate_limit_source', 'type': 'network', 'priority': 'high'},
                    {'name': 'lock_targeted_accounts', 'type': 'access_control', 'priority': 'high'},
                    {'name': 'notify_account_owners', 'type': 'notification', 'priority': 'medium'}
                ]
            },
            'data_exfiltration_response': {
                'name': 'Data Exfiltration Response',
                'description': 'Automated response to potential data exfiltration',
                'actions': [
                    {'name': 'block_network_egress', 'type': 'network', 'priority': 'critical'},
                    {'name': 'isolate_affected_systems', 'type': 'containment', 'priority': 'critical'},
                    {'name': 'preserve_forensic_evidence', 'type': 'investigation', 'priority': 'high'}
                ]
            }
        }
    
    def _trigger_automated_response(self, incident: SecurityIncident):
        """Trigger automated response based on incident type."""
        incident_category = incident.category.lower()
        
        # Map incident categories to playbooks
        playbook_mapping = {
            'sql_injection': 'sql_injection_response',
            'brute_force': 'brute_force_response',
            'data_exfiltration': 'data_exfiltration_response'
        }
        
        if incident_category in playbook_mapping:
            playbook_name = playbook_mapping[incident_category]
            logger.info(f"Triggering automated playbook: {playbook_name} for incident {incident.incident_id}")
            self.execute_containment_playbook(incident.incident_id, playbook_name)
    
    def _execute_containment_action(self, incident_id: str, action: Dict[str, Any]) -> str:
        """Execute individual containment action."""
        action_name = action.get('name', 'unknown')
        action_type = action.get('type', 'generic')
        
        # Simulate containment actions (in practice would integrate with security tools)
        if action_type == 'network':
            return f"Network action '{action_name}' executed successfully"
        elif action_type == 'access_control':
            return f"Access control action '{action_name}' executed successfully"
        elif action_type == 'investigation':
            return f"Investigation action '{action_name}' initiated"
        elif action_type == 'notification':
            return f"Notification '{action_name}' sent successfully"
        elif action_type == 'containment':
            return f"Containment action '{action_name}' executed successfully"
        else:
            return f"Generic action '{action_name}' executed"


class AutomatedVulnerabilityAssessment:
    """Enhanced vulnerability assessment with CVE integration and automated scanning."""
    
    def __init__(self):
        self.vulnerability_database = {}
        self.cve_cache = {}
        self.scan_history = []
        self.dependency_scanner = DependencyScanner()
        self._lock = threading.Lock()
    
    def scan_dependencies(self, project_path: str) -> List[VulnerabilityAssessment]:
        """Scan project dependencies for known vulnerabilities."""
        vulnerabilities = []
        
        try:
            # Scan different dependency files
            dependency_files = self._find_dependency_files(project_path)
            
            for dep_file in dependency_files:
                file_vulns = self._scan_dependency_file(dep_file)
                vulnerabilities.extend(file_vulns)
            
            # Record scan in history
            self.scan_history.append({
                'timestamp': datetime.utcnow(),
                'project_path': project_path,
                'vulnerabilities_found': len(vulnerabilities),
                'scan_type': 'dependency_scan'
            })
            
        except Exception as e:
            logger.error(f"Dependency scan failed: {e}")
        
        return vulnerabilities
    
    def get_cve_details(self, cve_id: str) -> Optional[Dict[str, Any]]:
        """Get CVE details from cache or external source."""
        if cve_id in self.cve_cache:
            return self.cve_cache[cve_id]
        
        # Simulate CVE lookup (in practice would use NIST NVD API)
        cve_details = self._mock_cve_lookup(cve_id)
        
        if cve_details:
            with self._lock:
                self.cve_cache[cve_id] = cve_details
        
        return cve_details
    
    def calculate_vulnerability_priority(self, vuln: VulnerabilityAssessment) -> int:
        """Calculate vulnerability remediation priority (1=highest, 5=lowest)."""
        priority = 3  # Default medium priority
        
        # CVSS score impact
        if vuln.cvss_score >= 9.0:
            priority = 1  # Critical
        elif vuln.cvss_score >= 7.0:
            priority = 2  # High
        elif vuln.cvss_score >= 4.0:
            priority = 3  # Medium
        else:
            priority = 4  # Low
        
        # Exploitability impact
        if vuln.exploitable and priority > 1:
            priority = max(1, priority - 1)
        
        # Component criticality (simplified)
        critical_components = ['authentication', 'encryption', 'database', 'api']
        if any(comp in vuln.component.lower() for comp in critical_components):
            priority = max(1, priority - 1)
        
        return min(5, max(1, priority))
    
    def generate_remediation_plan(self, vulnerabilities: List[VulnerabilityAssessment]) -> Dict[str, Any]:
        """Generate comprehensive remediation plan."""
        plan = {
            'total_vulnerabilities': len(vulnerabilities),
            'critical_count': 0,
            'high_count': 0,
            'medium_count': 0,
            'low_count': 0,
            'immediate_actions': [],
            'short_term_actions': [],
            'long_term_actions': [],
            'estimated_effort': 0,
            'risk_reduction': 0.0
        }
        
        for vuln in vulnerabilities:
            priority = self.calculate_vulnerability_priority(vuln)
            
            # Count by severity
            if vuln.cvss_score >= 9.0:
                plan['critical_count'] += 1
            elif vuln.cvss_score >= 7.0:
                plan['high_count'] += 1
            elif vuln.cvss_score >= 4.0:
                plan['medium_count'] += 1
            else:
                plan['low_count'] += 1
            
            # Categorize by remediation timeline
            remediation_item = {
                'vulnerability_id': vuln.vulnerability_id,
                'component': vuln.component,
                'severity': vuln.severity,
                'remediation': vuln.remediation,
                'deadline': vuln.remediation_deadline.isoformat(),
                'priority': priority
            }
            
            if priority == 1:  # Critical - immediate
                plan['immediate_actions'].append(remediation_item)
            elif priority == 2:  # High - within 1 week
                plan['short_term_actions'].append(remediation_item)
            else:  # Medium/Low - within 1 month
                plan['long_term_actions'].append(remediation_item)
            
            # Estimate effort (simplified)
            effort_mapping = {'critical': 8, 'high': 4, 'medium': 2, 'low': 1}
            plan['estimated_effort'] += effort_mapping.get(vuln.severity, 1)
        
        # Calculate risk reduction (simplified)
        total_risk = sum(vuln.cvss_score for vuln in vulnerabilities)
        if total_risk > 0:
            plan['risk_reduction'] = min(1.0, total_risk / (len(vulnerabilities) * 10))
        
        return plan
    
    def _find_dependency_files(self, project_path: str) -> List[str]:
        """Find dependency files in project."""
        dependency_files = []
        
        common_files = [
            'package.json',          # Node.js
            'requirements.txt',      # Python
            'Pipfile',              # Python (pipenv)
            'composer.json',        # PHP
            'pom.xml',              # Java (Maven)
            'build.gradle',         # Java (Gradle)
            'Gemfile',              # Ruby
            'go.mod',               # Go
        ]
        
        for file_name in common_files:
            file_path = os.path.join(project_path, file_name)
            if os.path.exists(file_path):
                dependency_files.append(file_path)
        
        return dependency_files
    
    def _scan_dependency_file(self, file_path: str) -> List[VulnerabilityAssessment]:
        """Scan individual dependency file."""
        vulnerabilities = []
        file_name = os.path.basename(file_path)
        
        try:
            # Parse different file types
            if file_name == 'package.json':
                dependencies = self._parse_package_json(file_path)
            elif file_name in ['requirements.txt', 'Pipfile']:
                dependencies = self._parse_python_requirements(file_path)
            else:
                dependencies = []  # Simplified - would implement other parsers
            
            # Check each dependency for vulnerabilities
            for dep_name, dep_version in dependencies.items():
                vulns = self._check_dependency_vulnerability(dep_name, dep_version, file_name)
                vulnerabilities.extend(vulns)
                
        except Exception as e:
            logger.error(f"Failed to scan {file_path}: {e}")
        
        return vulnerabilities
    
    def _parse_package_json(self, file_path: str) -> Dict[str, str]:
        """Parse package.json file."""
        try:
            with open(file_path, 'r') as f:
                package_data = json.load(f)
            
            dependencies = {}
            for dep_type in ['dependencies', 'devDependencies']:
                deps = package_data.get(dep_type, {})
                dependencies.update(deps)
            
            return dependencies
        except Exception:
            return {}
    
    def _parse_python_requirements(self, file_path: str) -> Dict[str, str]:
        """Parse Python requirements file."""
        try:
            dependencies = {}
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if '==' in line:
                            name, version = line.split('==')
                            dependencies[name.strip()] = version.strip()
            return dependencies
        except Exception:
            return {}
    
    def _check_dependency_vulnerability(self, name: str, version: str, 
                                      source_file: str) -> List[VulnerabilityAssessment]:
        """Check dependency for known vulnerabilities."""
        vulnerabilities = []
        
        # Simulate vulnerability check (would use real vulnerability databases)
        # For demo, create mock vulnerabilities for certain patterns
        if 'old' in name.lower() or '1.0' in version:
            vuln = VulnerabilityAssessment(
                vulnerability_id=f"VULN_{secrets.token_hex(4)}",
                cve_id=f"CVE-2023-{secrets.randbelow(10000):04d}",
                severity="high" if 'critical' in name.lower() else "medium",
                component=f"{name}@{version}",
                version=version,
                description=f"Security vulnerability in {name} version {version}",
                impact="Potential security compromise",
                remediation=f"Update {name} to latest version",
                cvss_score=7.5 if 'critical' in name.lower() else 5.5,
                exploitable=True,
                detection_date=datetime.utcnow(),
                remediation_deadline=datetime.utcnow() + timedelta(days=30)
            )
            vulnerabilities.append(vuln)
        
        return vulnerabilities
    
    def _mock_cve_lookup(self, cve_id: str) -> Dict[str, Any]:
        """Mock CVE lookup (would use real NIST NVD API)."""
        return {
            'cve_id': cve_id,
            'published_date': datetime.utcnow().isoformat(),
            'last_modified': datetime.utcnow().isoformat(),
            'cvss_v3_score': round(np.random.uniform(1.0, 10.0), 1),
            'description': f"Mock CVE entry for {cve_id}",
            'references': [f"https://cve.mitre.org/cgi-bin/cvename.cgi?name={cve_id}"]
        }


class DependencyScanner:
    """Helper class for dependency scanning."""
    
    def __init__(self):
        self.supported_ecosystems = ['npm', 'pip', 'composer', 'maven', 'gradle']
    
    def scan_ecosystem(self, ecosystem: str, manifest_path: str) -> List[Dict[str, Any]]:
        """Scan specific ecosystem for vulnerabilities."""
        if ecosystem not in self.supported_ecosystems:
            return []
        
        # Simplified scanning logic
        return []


def create_security_report(orchestrator: AdvancedSecurityOrchestrator, 
                          output_dir: str = "security_reports") -> str:
    """Create comprehensive security report."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Perform security audit
    audit_result = orchestrator.perform_security_audit()
    
    # Generate report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_dir, f"security_report_{timestamp}.json")
    
    report_data = {
        'report_metadata': {
            'generated_at': datetime.utcnow().isoformat(),
            'report_type': 'comprehensive_security_audit',
            'version': '1.0.0'
        },
        'audit_result': asdict(audit_result),
        'security_metrics': orchestrator.get_security_metrics(),
        'configuration': {
            'threat_detection_enabled': orchestrator.config.get('threat_detection_enabled'),
            'compliance_checks': orchestrator.config.get('compliance_checks'),
            'encryption_settings': {
                'at_rest': orchestrator.config.get('encryption_at_rest'),
                'in_transit': orchestrator.config.get('encryption_in_transit')
            }
        }
    }
    
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2, default=str)
    
    logger.info(f"Security report generated: {report_file}")
    return report_file


if __name__ == "__main__":
    print("Advanced Security and Compliance Framework")
    print("Provides enterprise-grade security monitoring and compliance validation.")