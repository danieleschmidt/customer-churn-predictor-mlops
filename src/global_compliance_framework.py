"""
Global Compliance Framework for Customer Churn Prediction.

This module provides comprehensive regulatory compliance capabilities including
GDPR, CCPA, PDPA compliance, data sovereignty, audit trails, and multi-language
support for global enterprise deployments.
"""

import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import uuid
import logging
from abc import ABC, abstractmethod

# Internationalization
try:
    import babel
    from babel.dates import format_datetime
    from babel.numbers import format_currency
    BABEL_AVAILABLE = True
except ImportError:
    BABEL_AVAILABLE = False

# Encryption and security
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    import secrets
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

# Local imports
from .logging_config import get_logger

logger = get_logger(__name__)


class ComplianceRegulation(Enum):
    """Supported compliance regulations."""
    GDPR = "gdpr"  # General Data Protection Regulation (EU)
    CCPA = "ccpa"  # California Consumer Privacy Act (US)
    PDPA = "pdpa"  # Personal Data Protection Act (Singapore/Thailand)
    PIPEDA = "pipeda"  # Personal Information Protection and Electronic Documents Act (Canada)
    LGPD = "lgpd"  # Lei Geral de Proteção de Dados (Brazil)
    PRIVACY_ACT = "privacy_act"  # Privacy Act (Australia)
    DPA = "dpa"  # Data Protection Act (UK)


class DataClassification(Enum):
    """Data classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PERSONAL = "personal"
    SENSITIVE_PERSONAL = "sensitive_personal"


class DataProcessingLawfulBasis(Enum):
    """GDPR lawful basis for processing."""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"


@dataclass
class DataSubject:
    """Represents a data subject (individual whose data is processed)."""
    subject_id: str
    email: Optional[str] = None
    jurisdiction: Optional[str] = None
    consent_status: Dict[str, bool] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class DataProcessingRecord:
    """Record of data processing activities."""
    record_id: str
    data_subject_id: str
    processing_purpose: str
    lawful_basis: DataProcessingLawfulBasis
    data_categories: List[str]
    retention_period: timedelta
    third_party_sharing: bool
    cross_border_transfer: bool
    automated_decision_making: bool
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditLogEntry:
    """Audit log entry for compliance tracking."""
    entry_id: str
    timestamp: datetime
    user_id: Optional[str]
    action: str
    resource: str
    data_subject_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    result: str  # success, failure, unauthorized
    details: Dict[str, Any] = field(default_factory=dict)


class DataRetentionPolicy:
    """Data retention policy management."""
    
    def __init__(self):
        self.policies: Dict[str, timedelta] = {
            'customer_data': timedelta(days=2555),  # 7 years
            'transaction_data': timedelta(days=2555),
            'marketing_data': timedelta(days=1095),  # 3 years
            'analytics_data': timedelta(days=730),  # 2 years
            'ml_model_data': timedelta(days=1095),
            'audit_logs': timedelta(days=2555),
            'consent_records': timedelta(days=2555)
        }
        
        # Regulation-specific overrides
        self.regulation_overrides = {
            ComplianceRegulation.GDPR: {
                'marketing_data': timedelta(days=730),
                'analytics_data': timedelta(days=365)
            }
        }
    
    def get_retention_period(self, data_type: str, regulation: ComplianceRegulation) -> timedelta:
        """Get retention period for data type under specific regulation."""
        # Check for regulation-specific override
        if regulation in self.regulation_overrides:
            overrides = self.regulation_overrides[regulation]
            if data_type in overrides:
                return overrides[data_type]
        
        # Return default policy
        return self.policies.get(data_type, timedelta(days=1095))  # Default 3 years
    
    def should_delete_data(self, data_type: str, created_date: datetime, regulation: ComplianceRegulation) -> bool:
        """Check if data should be deleted based on retention policy."""
        retention_period = self.get_retention_period(data_type, regulation)
        expiry_date = created_date + retention_period
        return datetime.now() > expiry_date


class ConsentManager:
    """Manages user consent for data processing."""
    
    def __init__(self):
        self.consents: Dict[str, Dict[str, Any]] = {}
        self.consent_history: List[Dict[str, Any]] = []
    
    def record_consent(
        self, 
        subject_id: str, 
        purpose: str, 
        granted: bool,
        consent_method: str = "explicit",
        jurisdiction: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Record consent given by data subject."""
        consent_id = str(uuid.uuid4())
        
        consent_record = {
            'consent_id': consent_id,
            'subject_id': subject_id,
            'purpose': purpose,
            'granted': granted,
            'method': consent_method,
            'jurisdiction': jurisdiction,
            'timestamp': datetime.now(),
            'metadata': metadata or {}
        }
        
        # Update current consent status
        if subject_id not in self.consents:
            self.consents[subject_id] = {}
        
        self.consents[subject_id][purpose] = {
            'granted': granted,
            'consent_id': consent_id,
            'timestamp': consent_record['timestamp'],
            'method': consent_method
        }
        
        # Add to history
        self.consent_history.append(consent_record)
        
        logger.info(f"Consent recorded: {subject_id} - {purpose} - {'Granted' if granted else 'Denied'}")
        
        return consent_id
    
    def withdraw_consent(self, subject_id: str, purpose: str) -> bool:
        """Withdraw consent for specific purpose."""
        if subject_id in self.consents and purpose in self.consents[subject_id]:
            # Record withdrawal
            withdrawal_id = self.record_consent(
                subject_id, purpose, granted=False, 
                consent_method="withdrawal"
            )
            
            logger.info(f"Consent withdrawn: {subject_id} - {purpose}")
            return True
        
        return False
    
    def check_consent(self, subject_id: str, purpose: str) -> bool:
        """Check if subject has given consent for purpose."""
        if subject_id in self.consents and purpose in self.consents[subject_id]:
            return self.consents[subject_id][purpose]['granted']
        return False
    
    def get_consent_status(self, subject_id: str) -> Dict[str, Any]:
        """Get all consent status for a subject."""
        return self.consents.get(subject_id, {})


class DataSubjectRightsManager:
    """Manages data subject rights under various regulations."""
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        self.requests: Dict[str, Dict[str, Any]] = {}
        self.fulfillment_log: List[Dict[str, Any]] = []
        
        # Initialize encryption
        if CRYPTO_AVAILABLE and encryption_key:
            self.cipher = Fernet(encryption_key)
        else:
            self.cipher = None
    
    def submit_access_request(self, subject_id: str, email: str, data_types: List[str]) -> str:
        """Submit data access request (GDPR Article 15)."""
        request_id = str(uuid.uuid4())
        
        request = {
            'request_id': request_id,
            'type': 'access',
            'subject_id': subject_id,
            'email': email,
            'data_types': data_types,
            'status': 'pending',
            'submitted_at': datetime.now(),
            'due_date': datetime.now() + timedelta(days=30)  # GDPR requirement
        }
        
        self.requests[request_id] = request
        logger.info(f"Data access request submitted: {request_id}")
        
        return request_id
    
    def submit_rectification_request(
        self, 
        subject_id: str, 
        email: str, 
        corrections: Dict[str, Any]
    ) -> str:
        """Submit data rectification request (GDPR Article 16)."""
        request_id = str(uuid.uuid4())
        
        request = {
            'request_id': request_id,
            'type': 'rectification',
            'subject_id': subject_id,
            'email': email,
            'corrections': corrections,
            'status': 'pending',
            'submitted_at': datetime.now(),
            'due_date': datetime.now() + timedelta(days=30)
        }
        
        self.requests[request_id] = request
        logger.info(f"Data rectification request submitted: {request_id}")
        
        return request_id
    
    def submit_erasure_request(self, subject_id: str, email: str, reason: str) -> str:
        """Submit data erasure request (GDPR Article 17 - Right to be forgotten)."""
        request_id = str(uuid.uuid4())
        
        request = {
            'request_id': request_id,
            'type': 'erasure',
            'subject_id': subject_id,
            'email': email,
            'reason': reason,
            'status': 'pending',
            'submitted_at': datetime.now(),
            'due_date': datetime.now() + timedelta(days=30)
        }
        
        self.requests[request_id] = request
        logger.info(f"Data erasure request submitted: {request_id}")
        
        return request_id
    
    def submit_portability_request(
        self, 
        subject_id: str, 
        email: str, 
        format_preference: str = "json"
    ) -> str:
        """Submit data portability request (GDPR Article 20)."""
        request_id = str(uuid.uuid4())
        
        request = {
            'request_id': request_id,
            'type': 'portability',
            'subject_id': subject_id,
            'email': email,
            'format_preference': format_preference,
            'status': 'pending',
            'submitted_at': datetime.now(),
            'due_date': datetime.now() + timedelta(days=30)
        }
        
        self.requests[request_id] = request
        logger.info(f"Data portability request submitted: {request_id}")
        
        return request_id
    
    def process_request(self, request_id: str, fulfillment_data: Dict[str, Any]) -> bool:
        """Process and fulfill a data subject rights request."""
        if request_id not in self.requests:
            return False
        
        request = self.requests[request_id]
        request['status'] = 'completed'
        request['completed_at'] = datetime.now()
        request['fulfillment_data'] = fulfillment_data
        
        # Log fulfillment
        fulfillment_record = {
            'request_id': request_id,
            'request_type': request['type'],
            'subject_id': request['subject_id'],
            'fulfilled_at': datetime.now(),
            'fulfillment_method': fulfillment_data.get('method', 'automated')
        }
        
        self.fulfillment_log.append(fulfillment_record)
        
        logger.info(f"Data subject request fulfilled: {request_id}")
        return True
    
    def get_pending_requests(self) -> List[Dict[str, Any]]:
        """Get all pending requests."""
        return [req for req in self.requests.values() if req['status'] == 'pending']
    
    def get_overdue_requests(self) -> List[Dict[str, Any]]:
        """Get all overdue requests."""
        now = datetime.now()
        return [
            req for req in self.requests.values() 
            if req['status'] == 'pending' and req['due_date'] < now
        ]


class InternationalizationManager:
    """Manages multi-language support and localization."""
    
    def __init__(self):
        self.supported_locales = {
            'en': 'English',
            'es': 'Español',
            'fr': 'Français', 
            'de': 'Deutsch',
            'ja': '日本語',
            'zh': '中文',
            'pt': 'Português',
            'it': 'Italiano',
            'ru': 'Русский',
            'ar': 'العربية'
        }
        
        # Privacy notice templates in multiple languages
        self.privacy_notices = {
            'en': {
                'title': 'Privacy Notice for ML Model',
                'collection': 'We collect your data to improve our churn prediction model.',
                'purpose': 'Your data is used for machine learning model training and prediction.',
                'retention': 'We retain your data for {days} days as required by law.',
                'rights': 'You have the right to access, rectify, and delete your personal data.',
                'contact': 'Contact us at privacy@company.com for data-related inquiries.'
            },
            'es': {
                'title': 'Aviso de Privacidad para Modelo ML',
                'collection': 'Recopilamos sus datos para mejorar nuestro modelo de predicción de abandono.',
                'purpose': 'Sus datos se utilizan para entrenar y predecir con nuestro modelo de aprendizaje automático.',
                'retention': 'Conservamos sus datos durante {days} días según lo exige la ley.',
                'rights': 'Tiene derecho a acceder, rectificar y eliminar sus datos personales.',
                'contact': 'Contáctenos en privacy@company.com para consultas relacionadas con datos.'
            },
            'fr': {
                'title': 'Avis de Confidentialité pour Modèle ML',
                'collection': 'Nous collectons vos données pour améliorer notre modèle de prédiction d\'attrition.',
                'purpose': 'Vos données sont utilisées pour entraîner et prédire avec notre modèle d\'apprentissage automatique.',
                'retention': 'Nous conservons vos données pendant {days} jours comme l\'exige la loi.',
                'rights': 'Vous avez le droit d\'accéder, rectifier et supprimer vos données personnelles.',
                'contact': 'Contactez-nous à privacy@company.com pour les demandes liées aux données.'
            },
            'de': {
                'title': 'Datenschutzhinweis für ML-Modell',
                'collection': 'Wir sammeln Ihre Daten, um unser Kündigungsprognosemodell zu verbessern.',
                'purpose': 'Ihre Daten werden zum Trainieren und Vorhersagen mit unserem maschinellen Lernmodell verwendet.',
                'retention': 'Wir bewahren Ihre Daten {days} Tage lang auf, wie gesetzlich vorgeschrieben.',
                'rights': 'Sie haben das Recht auf Zugang, Berichtigung und Löschung Ihrer personenbezogenen Daten.',
                'contact': 'Kontaktieren Sie uns unter privacy@company.com für datenbezogene Anfragen.'
            }
        }
        
        # Compliance terms dictionary
        self.compliance_terms = {
            'en': {
                'data_controller': 'Data Controller',
                'data_processor': 'Data Processor',
                'lawful_basis': 'Lawful Basis',
                'consent': 'Consent',
                'legitimate_interest': 'Legitimate Interest',
                'data_subject': 'Data Subject',
                'personal_data': 'Personal Data',
                'processing': 'Processing',
                'retention_period': 'Retention Period',
                'right_to_access': 'Right to Access',
                'right_to_rectification': 'Right to Rectification',
                'right_to_erasure': 'Right to Erasure',
                'right_to_portability': 'Right to Data Portability'
            },
            'es': {
                'data_controller': 'Responsable del Tratamiento',
                'data_processor': 'Encargado del Tratamiento',
                'lawful_basis': 'Base Jurídica',
                'consent': 'Consentimiento',
                'legitimate_interest': 'Interés Legítimo',
                'data_subject': 'Interesado',
                'personal_data': 'Datos Personales',
                'processing': 'Tratamiento',
                'retention_period': 'Período de Retención',
                'right_to_access': 'Derecho de Acceso',
                'right_to_rectification': 'Derecho de Rectificación',
                'right_to_erasure': 'Derecho de Supresión',
                'right_to_portability': 'Derecho a la Portabilidad'
            }
        }
    
    def get_privacy_notice(self, locale: str, retention_days: int) -> Dict[str, str]:
        """Get localized privacy notice."""
        if locale not in self.privacy_notices:
            locale = 'en'  # Fallback to English
        
        notice = self.privacy_notices[locale].copy()
        notice['retention'] = notice['retention'].format(days=retention_days)
        
        return notice
    
    def get_compliance_term(self, term: str, locale: str) -> str:
        """Get localized compliance term."""
        if locale not in self.compliance_terms:
            locale = 'en'
        
        terms = self.compliance_terms[locale]
        return terms.get(term, term)
    
    def format_date_for_locale(self, date: datetime, locale: str) -> str:
        """Format date according to locale."""
        if BABEL_AVAILABLE:
            try:
                return format_datetime(date, locale=locale)
            except:
                pass
        
        # Fallback formatting
        if locale.startswith('en'):
            return date.strftime('%m/%d/%Y %H:%M')
        elif locale.startswith('de') or locale.startswith('fr'):
            return date.strftime('%d.%m.%Y %H:%M')
        else:
            return date.strftime('%Y-%m-%d %H:%M')


class AuditTrailManager:
    """Manages comprehensive audit trails for compliance."""
    
    def __init__(self, log_file_path: Optional[str] = None):
        self.audit_logs: List[AuditLogEntry] = []
        self.log_file_path = log_file_path or "audit_trail.log"
        
        # Set up audit logger
        self.audit_logger = logging.getLogger('compliance_audit')
        self.audit_logger.setLevel(logging.INFO)
        
        if not self.audit_logger.handlers:
            handler = logging.FileHandler(self.log_file_path)
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.audit_logger.addHandler(handler)
    
    def log_data_access(
        self, 
        user_id: str, 
        data_subject_id: str, 
        data_types: List[str],
        purpose: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        result: str = "success"
    ):
        """Log data access event."""
        entry = AuditLogEntry(
            entry_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            user_id=user_id,
            action="data_access",
            resource="personal_data",
            data_subject_id=data_subject_id,
            ip_address=ip_address,
            user_agent=user_agent,
            result=result,
            details={
                'data_types': data_types,
                'purpose': purpose
            }
        )
        
        self._store_audit_entry(entry)
    
    def log_model_training(
        self,
        user_id: str,
        model_id: str,
        data_subjects_count: int,
        features_used: List[str],
        purpose: str = "ml_model_training",
        result: str = "success"
    ):
        """Log ML model training event."""
        entry = AuditLogEntry(
            entry_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            user_id=user_id,
            action="model_training",
            resource=model_id,
            data_subject_id=None,
            ip_address=None,
            user_agent=None,
            result=result,
            details={
                'data_subjects_count': data_subjects_count,
                'features_used': features_used,
                'purpose': purpose
            }
        )
        
        self._store_audit_entry(entry)
    
    def log_prediction_request(
        self,
        user_id: str,
        model_id: str,
        data_subject_id: Optional[str],
        prediction_result: Any,
        ip_address: Optional[str] = None,
        result: str = "success"
    ):
        """Log ML prediction request."""
        entry = AuditLogEntry(
            entry_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            user_id=user_id,
            action="prediction_request",
            resource=model_id,
            data_subject_id=data_subject_id,
            ip_address=ip_address,
            user_agent=None,
            result=result,
            details={
                'prediction_result': str(prediction_result)
            }
        )
        
        self._store_audit_entry(entry)
    
    def log_consent_change(
        self,
        subject_id: str,
        purpose: str,
        old_status: bool,
        new_status: bool,
        method: str
    ):
        """Log consent status change."""
        entry = AuditLogEntry(
            entry_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            user_id=None,
            action="consent_change",
            resource="consent_record",
            data_subject_id=subject_id,
            ip_address=None,
            user_agent=None,
            result="success",
            details={
                'purpose': purpose,
                'old_status': old_status,
                'new_status': new_status,
                'method': method
            }
        )
        
        self._store_audit_entry(entry)
    
    def log_data_deletion(
        self,
        user_id: str,
        data_subject_id: str,
        data_types: List[str],
        reason: str,
        result: str = "success"
    ):
        """Log data deletion event."""
        entry = AuditLogEntry(
            entry_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            user_id=user_id,
            action="data_deletion",
            resource="personal_data",
            data_subject_id=data_subject_id,
            ip_address=None,
            user_agent=None,
            result=result,
            details={
                'data_types': data_types,
                'reason': reason
            }
        )
        
        self._store_audit_entry(entry)
    
    def _store_audit_entry(self, entry: AuditLogEntry):
        """Store audit entry in memory and log file."""
        self.audit_logs.append(entry)
        
        # Log to file
        log_message = (
            f"ID:{entry.entry_id} | "
            f"Action:{entry.action} | "
            f"User:{entry.user_id} | "
            f"Resource:{entry.resource} | "
            f"Subject:{entry.data_subject_id} | "
            f"Result:{entry.result} | "
            f"Details:{json.dumps(entry.details)}"
        )
        
        self.audit_logger.info(log_message)
        
        # Keep only recent entries in memory
        if len(self.audit_logs) > 10000:
            self.audit_logs = self.audit_logs[-5000:]
    
    def get_audit_trail_for_subject(self, subject_id: str) -> List[AuditLogEntry]:
        """Get all audit entries for a specific data subject."""
        return [
            entry for entry in self.audit_logs 
            if entry.data_subject_id == subject_id
        ]
    
    def generate_compliance_report(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate compliance report for given time period."""
        relevant_entries = [
            entry for entry in self.audit_logs
            if start_date <= entry.timestamp <= end_date
        ]
        
        # Aggregate statistics
        action_counts = {}
        user_activity = {}
        success_rate = 0
        
        for entry in relevant_entries:
            # Count actions
            action_counts[entry.action] = action_counts.get(entry.action, 0) + 1
            
            # Count user activity
            if entry.user_id:
                user_activity[entry.user_id] = user_activity.get(entry.user_id, 0) + 1
            
            # Calculate success rate
            if entry.result == "success":
                success_rate += 1
        
        success_rate = (success_rate / len(relevant_entries)) * 100 if relevant_entries else 0
        
        return {
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'total_entries': len(relevant_entries),
            'action_breakdown': action_counts,
            'user_activity': user_activity,
            'success_rate_percent': success_rate,
            'unique_users': len(user_activity),
            'unique_data_subjects': len(set(
                entry.data_subject_id for entry in relevant_entries 
                if entry.data_subject_id
            ))
        }


class GlobalComplianceFramework:
    """
    Comprehensive global compliance framework for ML systems.
    
    Features:
    - Multi-regulation compliance (GDPR, CCPA, PDPA, etc.)
    - Data subject rights management
    - Consent management with audit trails
    - Automated data retention and deletion
    - Comprehensive audit logging
    - Multi-language privacy notices
    - Cross-border data transfer controls
    """
    
    def __init__(
        self,
        primary_regulation: ComplianceRegulation = ComplianceRegulation.GDPR,
        supported_locales: Optional[List[str]] = None,
        encryption_key: Optional[bytes] = None
    ):
        self.primary_regulation = primary_regulation
        self.supported_locales = supported_locales or ['en', 'es', 'fr', 'de']
        
        # Initialize components
        self.retention_policy = DataRetentionPolicy()
        self.consent_manager = ConsentManager()
        self.rights_manager = DataSubjectRightsManager(encryption_key)
        self.i18n_manager = InternationalizationManager()
        self.audit_manager = AuditTrailManager()
        
        # Data processing records
        self.processing_records: Dict[str, DataProcessingRecord] = {}
        self.data_subjects: Dict[str, DataSubject] = {}
        
        logger.info(f"Global Compliance Framework initialized for {primary_regulation.value}")
    
    def register_data_subject(
        self, 
        subject_id: str, 
        email: Optional[str] = None,
        jurisdiction: Optional[str] = None
    ) -> DataSubject:
        """Register a new data subject."""
        subject = DataSubject(
            subject_id=subject_id,
            email=email,
            jurisdiction=jurisdiction
        )
        
        self.data_subjects[subject_id] = subject
        
        # Log registration
        self.audit_manager.log_data_access(
            user_id="system",
            data_subject_id=subject_id,
            data_types=["registration"],
            purpose="data_subject_registration"
        )
        
        return subject
    
    def create_processing_record(
        self,
        subject_id: str,
        processing_purpose: str,
        lawful_basis: DataProcessingLawfulBasis,
        data_categories: List[str],
        retention_days: int,
        third_party_sharing: bool = False,
        cross_border_transfer: bool = False,
        automated_decision_making: bool = True
    ) -> str:
        """Create a data processing record."""
        record_id = str(uuid.uuid4())
        
        record = DataProcessingRecord(
            record_id=record_id,
            data_subject_id=subject_id,
            processing_purpose=processing_purpose,
            lawful_basis=lawful_basis,
            data_categories=data_categories,
            retention_period=timedelta(days=retention_days),
            third_party_sharing=third_party_sharing,
            cross_border_transfer=cross_border_transfer,
            automated_decision_making=automated_decision_making
        )
        
        self.processing_records[record_id] = record
        
        # Log processing record creation
        self.audit_manager.log_data_access(
            user_id="system",
            data_subject_id=subject_id,
            data_types=data_categories,
            purpose=processing_purpose
        )
        
        return record_id
    
    def check_ml_compliance(
        self,
        model_id: str,
        training_data_subjects: List[str],
        features_used: List[str],
        purpose: str = "churn_prediction"
    ) -> Dict[str, Any]:
        """Check ML model compliance status."""
        compliance_issues = []
        compliance_score = 100
        
        # Check consent for all data subjects
        consent_issues = 0
        for subject_id in training_data_subjects:
            if not self.consent_manager.check_consent(subject_id, purpose):
                consent_issues += 1
                compliance_issues.append(f"Missing consent for subject {subject_id}")
        
        if consent_issues > 0:
            compliance_score -= min(50, consent_issues * 5)
        
        # Check data retention
        retention_issues = 0
        for subject_id in training_data_subjects:
            if subject_id in self.data_subjects:
                subject = self.data_subjects[subject_id]
                if self.retention_policy.should_delete_data(
                    "ml_model_data", 
                    subject.created_at, 
                    self.primary_regulation
                ):
                    retention_issues += 1
                    compliance_issues.append(f"Data retention exceeded for subject {subject_id}")
        
        if retention_issues > 0:
            compliance_score -= min(30, retention_issues * 3)
        
        # Check for sensitive features
        sensitive_features = ['age', 'gender', 'ethnicity', 'health_status']
        sensitive_used = [f for f in features_used if f.lower() in sensitive_features]
        
        if sensitive_used:
            compliance_issues.append(f"Sensitive features used: {sensitive_used}")
            compliance_score -= 20
        
        # Overall compliance status
        if compliance_score >= 90:
            status = "compliant"
        elif compliance_score >= 70:
            status = "warning"
        else:
            status = "non_compliant"
        
        result = {
            'model_id': model_id,
            'compliance_status': status,
            'compliance_score': compliance_score,
            'primary_regulation': self.primary_regulation.value,
            'issues_found': compliance_issues,
            'data_subjects_count': len(training_data_subjects),
            'consent_coverage': (len(training_data_subjects) - consent_issues) / len(training_data_subjects) * 100,
            'features_used': features_used,
            'sensitive_features': sensitive_used,
            'check_timestamp': datetime.now().isoformat()
        }
        
        # Log compliance check
        self.audit_manager.log_model_training(
            user_id="compliance_system",
            model_id=model_id,
            data_subjects_count=len(training_data_subjects),
            features_used=features_used,
            result=status
        )
        
        return result
    
    def handle_data_subject_request(
        self,
        request_type: str,
        subject_id: str,
        email: str,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Handle data subject rights requests."""
        additional_data = additional_data or {}
        
        if request_type == "access":
            return self.rights_manager.submit_access_request(
                subject_id, email, additional_data.get('data_types', [])
            )
        elif request_type == "rectification":
            return self.rights_manager.submit_rectification_request(
                subject_id, email, additional_data.get('corrections', {})
            )
        elif request_type == "erasure":
            return self.rights_manager.submit_erasure_request(
                subject_id, email, additional_data.get('reason', 'user_request')
            )
        elif request_type == "portability":
            return self.rights_manager.submit_portability_request(
                subject_id, email, additional_data.get('format', 'json')
            )
        else:
            raise ValueError(f"Unsupported request type: {request_type}")
    
    def generate_privacy_notice(
        self, 
        locale: str = 'en',
        data_types: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """Generate localized privacy notice."""
        # Get retention period for ML data
        retention_period = self.retention_policy.get_retention_period(
            "ml_model_data", self.primary_regulation
        )
        
        return self.i18n_manager.get_privacy_notice(locale, retention_period.days)
    
    def get_compliance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive compliance dashboard data."""
        now = datetime.now()
        last_30_days = now - timedelta(days=30)
        
        # Generate audit report
        audit_report = self.audit_manager.generate_compliance_report(last_30_days, now)
        
        # Pending requests
        pending_requests = self.rights_manager.get_pending_requests()
        overdue_requests = self.rights_manager.get_overdue_requests()
        
        # Consent statistics
        total_subjects = len(self.data_subjects)
        consent_stats = {
            'total_subjects': total_subjects,
            'with_consent': 0,
            'consent_purposes': {}
        }
        
        for subject_id in self.data_subjects:
            subject_consents = self.consent_manager.get_consent_status(subject_id)
            if any(consent['granted'] for consent in subject_consents.values()):
                consent_stats['with_consent'] += 1
            
            for purpose in subject_consents:
                if purpose not in consent_stats['consent_purposes']:
                    consent_stats['consent_purposes'][purpose] = {'granted': 0, 'denied': 0}
                
                if subject_consents[purpose]['granted']:
                    consent_stats['consent_purposes'][purpose]['granted'] += 1
                else:
                    consent_stats['consent_purposes'][purpose]['denied'] += 1
        
        return {
            'overview': {
                'primary_regulation': self.primary_regulation.value,
                'supported_locales': self.supported_locales,
                'total_data_subjects': total_subjects,
                'total_processing_records': len(self.processing_records)
            },
            'data_subject_requests': {
                'pending_count': len(pending_requests),
                'overdue_count': len(overdue_requests),
                'pending_requests': pending_requests[:10],  # Latest 10
                'overdue_requests': overdue_requests
            },
            'consent_management': consent_stats,
            'audit_trail': audit_report,
            'retention_compliance': {
                'policies_count': len(self.retention_policy.policies),
                'regulation_overrides': len(self.retention_policy.regulation_overrides)
            },
            'generated_at': now.isoformat()
        }


# Factory functions

def create_gdpr_compliance_framework(
    supported_locales: Optional[List[str]] = None
) -> GlobalComplianceFramework:
    """Create GDPR-compliant framework."""
    return GlobalComplianceFramework(
        primary_regulation=ComplianceRegulation.GDPR,
        supported_locales=supported_locales or ['en', 'de', 'fr', 'es', 'it']
    )


def create_ccpa_compliance_framework(
    supported_locales: Optional[List[str]] = None
) -> GlobalComplianceFramework:
    """Create CCPA-compliant framework."""
    return GlobalComplianceFramework(
        primary_regulation=ComplianceRegulation.CCPA,
        supported_locales=supported_locales or ['en', 'es']
    )


def create_multi_regulation_framework(
    regulations: List[ComplianceRegulation],
    supported_locales: Optional[List[str]] = None
) -> Dict[str, GlobalComplianceFramework]:
    """Create multiple compliance frameworks."""
    frameworks = {}
    
    for regulation in regulations:
        frameworks[regulation.value] = GlobalComplianceFramework(
            primary_regulation=regulation,
            supported_locales=supported_locales
        )
    
    return frameworks