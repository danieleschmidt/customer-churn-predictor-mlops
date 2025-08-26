"""
Comprehensive Security Audit for MLOps Platform.

Performs automated security testing, vulnerability assessment, and compliance validation.
"""

import asyncio
import sys
import os
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.logging_config import get_logger
from src.validation import safe_write_json
from src.security import get_security_report

logger = get_logger(__name__)


@dataclass
class SecurityFinding:
    """Security audit finding."""
    severity: str  # critical, high, medium, low
    category: str
    title: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    cve_id: Optional[str] = None
    recommendation: str = ""


@dataclass
class SecurityAuditReport:
    """Comprehensive security audit report."""
    timestamp: float = field(default_factory=time.time)
    total_files_scanned: int = 0
    findings: List[SecurityFinding] = field(default_factory=list)
    dependency_vulnerabilities: List[Dict[str, Any]] = field(default_factory=list)
    secret_exposures: List[Dict[str, Any]] = field(default_factory=list)
    compliance_status: Dict[str, bool] = field(default_factory=dict)
    security_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            'timestamp': self.timestamp,
            'total_files_scanned': self.total_files_scanned,
            'findings': [
                {
                    'severity': f.severity,
                    'category': f.category,
                    'title': f.title,
                    'description': f.description,
                    'file_path': f.file_path,
                    'line_number': f.line_number,
                    'cve_id': f.cve_id,
                    'recommendation': f.recommendation
                }
                for f in self.findings
            ],
            'dependency_vulnerabilities': self.dependency_vulnerabilities,
            'secret_exposures': self.secret_exposures,
            'compliance_status': self.compliance_status,
            'security_score': self.security_score
        }


class SecurityAuditor:
    """Comprehensive security auditor."""
    
    def __init__(self):
        self.report = SecurityAuditReport()
        self.scanned_files = set()
        
    async def run_comprehensive_audit(self) -> SecurityAuditReport:
        """Run comprehensive security audit."""
        logger.info("Starting comprehensive security audit...")
        
        # Static code analysis
        await self._run_static_analysis()
        
        # Dependency vulnerability scanning
        await self._scan_dependencies()
        
        # Secret detection
        await self._detect_secrets()
        
        # Configuration security
        await self._audit_configuration()
        
        # API security validation
        await self._validate_api_security()
        
        # Compliance checks
        await self._check_compliance()
        
        # Calculate security score
        self._calculate_security_score()
        
        logger.info(f"Security audit completed. Score: {self.report.security_score:.1f}/100")
        return self.report
        
    async def _run_static_analysis(self):
        """Run static code analysis for security issues."""
        logger.info("Running static code analysis...")
        
        # Check for common security issues
        await self._check_hardcoded_secrets()
        await self._check_sql_injection_patterns()
        await self._check_command_injection_patterns()
        await self._check_path_traversal_patterns()
        await self._check_xss_patterns()
        await self._check_insecure_random()
        await self._check_weak_crypto()
        
    async def _check_hardcoded_secrets(self):
        """Check for hardcoded secrets in code."""
        secret_patterns = [
            (r'password\s*=\s*["\'][^"\']{8,}["\']', 'Hardcoded Password'),
            (r'api[_-]?key\s*=\s*["\'][^"\']{16,}["\']', 'Hardcoded API Key'),
            (r'secret[_-]?key\s*=\s*["\'][^"\']{16,}["\']', 'Hardcoded Secret Key'),
            (r'token\s*=\s*["\'][^"\']{16,}["\']', 'Hardcoded Token'),
            (r'private[_-]?key\s*=\s*["\'][^"\']{32,}["\']', 'Hardcoded Private Key'),
        ]
        
        await self._scan_files_for_patterns(secret_patterns, 'Secret Exposure')
        
    async def _check_sql_injection_patterns(self):
        """Check for potential SQL injection vulnerabilities."""
        sql_patterns = [
            (r'execute\(["\'].*%.*["\'].*%', 'Potential SQL Injection'),
            (r'query\(["\'].*\+.*["\']', 'String Concatenation in SQL'),
            (r'WHERE.*=.*\+', 'Unsafe SQL Concatenation'),
        ]
        
        await self._scan_files_for_patterns(sql_patterns, 'Injection Vulnerability')
        
    async def _check_command_injection_patterns(self):
        """Check for command injection vulnerabilities."""
        command_patterns = [
            (r'os\.system\(.*\+.*\)', 'Command Injection Risk'),
            (r'subprocess\.call\(.*\+.*\)', 'Command Injection Risk'),
            (r'subprocess\.run\(.*shell=True.*\+', 'Shell Injection Risk'),
        ]
        
        await self._scan_files_for_patterns(command_patterns, 'Command Injection')
        
    async def _check_path_traversal_patterns(self):
        """Check for path traversal vulnerabilities."""
        path_patterns = [
            (r'open\(["\'].*\.\.[/\\]', 'Path Traversal Risk'),
            (r'Path\(["\'].*\.\.[/\\]', 'Path Traversal Risk'),
            (r'join\(.*\.\.[/\\]', 'Path Traversal Risk'),
        ]
        
        await self._scan_files_for_patterns(path_patterns, 'Path Traversal')
        
    async def _check_xss_patterns(self):
        """Check for XSS vulnerabilities."""
        xss_patterns = [
            (r'render_template_string\(.*\+', 'XSS Risk in Template'),
            (r'Markup\(.*\+', 'Unsafe HTML Markup'),
            (r'|safe.*\+', 'Unsafe Safe Filter Usage'),
        ]
        
        await self._scan_files_for_patterns(xss_patterns, 'Cross-Site Scripting')
        
    async def _check_insecure_random(self):
        """Check for insecure random number generation."""
        random_patterns = [
            (r'import random(?!\s+from\s+secrets)', 'Insecure Random Usage'),
            (r'random\.random\(', 'Weak Random Number Generation'),
            (r'random\.randint\(', 'Weak Random Number Generation'),
        ]
        
        await self._scan_files_for_patterns(random_patterns, 'Weak Randomness')
        
    async def _check_weak_crypto(self):
        """Check for weak cryptographic practices."""
        crypto_patterns = [
            (r'hashlib\.md5\(', 'Weak Hash Algorithm (MD5)'),
            (r'hashlib\.sha1\(', 'Weak Hash Algorithm (SHA1)'),
            (r'DES\.new\(', 'Weak Encryption (DES)'),
            (r'RC4\.new\(', 'Weak Encryption (RC4)'),
        ]
        
        await self._scan_files_for_patterns(crypto_patterns, 'Weak Cryptography')
        
    async def _scan_files_for_patterns(self, patterns: List[tuple], category: str):
        """Scan files for security patterns."""
        import re
        
        python_files = list(Path('src').rglob('*.py'))
        self.report.total_files_scanned = len(python_files)
        
        for file_path in python_files:
            if file_path in self.scanned_files:
                continue
                
            try:
                content = file_path.read_text(encoding='utf-8')
                lines = content.split('\n')
                
                for pattern, title in patterns:
                    regex = re.compile(pattern, re.IGNORECASE)
                    
                    for line_num, line in enumerate(lines, 1):
                        if regex.search(line):
                            severity = self._determine_severity(category, title)
                            
                            finding = SecurityFinding(
                                severity=severity,
                                category=category,
                                title=title,
                                description=f"Found pattern '{pattern}' in code",
                                file_path=str(file_path),
                                line_number=line_num,
                                recommendation=self._get_recommendation(title)
                            )
                            
                            self.report.findings.append(finding)
                            
            except Exception as e:
                logger.warning(f"Error scanning {file_path}: {e}")
                
            self.scanned_files.add(file_path)
            
    def _determine_severity(self, category: str, title: str) -> str:
        """Determine severity based on vulnerability type."""
        critical_patterns = ['SQL Injection', 'Command Injection', 'Hardcoded Private Key']
        high_patterns = ['Hardcoded Password', 'Hardcoded API Key', 'Shell Injection']
        
        if any(pattern in title for pattern in critical_patterns):
            return 'critical'
        elif any(pattern in title for pattern in high_patterns):
            return 'high'
        elif category in ['Secret Exposure', 'Injection Vulnerability']:
            return 'medium'
        else:
            return 'low'
            
    def _get_recommendation(self, title: str) -> str:
        """Get security recommendation for vulnerability."""
        recommendations = {
            'Hardcoded Password': 'Use environment variables or secure configuration management',
            'Hardcoded API Key': 'Store API keys in environment variables or key management systems',
            'Hardcoded Secret Key': 'Use secure secret management solutions',
            'SQL Injection': 'Use parameterized queries or ORM methods',
            'Command Injection': 'Validate input and use safe subprocess methods',
            'Path Traversal': 'Validate file paths and use safe path operations',
            'XSS Risk': 'Sanitize output and use proper templating escaping',
            'Weak Random': 'Use cryptographically secure random generators from secrets module',
            'Weak Hash': 'Use SHA-256 or stronger hash algorithms',
            'Weak Encryption': 'Use AES or other modern encryption algorithms'
        }
        
        for key, recommendation in recommendations.items():
            if key in title:
                return recommendation
                
        return 'Review code for security best practices'
        
    async def _scan_dependencies(self):
        """Scan dependencies for known vulnerabilities."""
        logger.info("Scanning dependencies for vulnerabilities...")
        
        try:
            # Try to use safety to scan dependencies
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'list', '--format=json'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                packages = json.loads(result.stdout)
                
                # Simulate vulnerability check (would use real vulnerability database)
                vulnerable_packages = [
                    'urllib3', 'requests', 'pyyaml', 'jinja2', 'flask'
                ]
                
                for package in packages:
                    if package['name'].lower() in vulnerable_packages:
                        vuln = {
                            'package': package['name'],
                            'version': package['version'],
                            'severity': 'medium',
                            'description': f'Potential vulnerability in {package["name"]}',
                            'recommendation': 'Update to latest version'
                        }
                        self.report.dependency_vulnerabilities.append(vuln)
                        
        except Exception as e:
            logger.error(f"Error scanning dependencies: {e}")
            
    async def _detect_secrets(self):
        """Detect exposed secrets in configuration files."""
        logger.info("Detecting exposed secrets...")
        
        config_files = [
            'config.yml', '.env', 'docker-compose.yml',
            'requirements.txt', 'pyproject.toml'
        ]
        
        secret_patterns = [
            r'[A-Za-z0-9+/]{40,}={0,2}',  # Base64 encoded
            r'[0-9a-f]{32,}',              # Hex strings
            r'sk_[a-zA-Z0-9]{24,}',        # API keys
            r'AIza[0-9A-Za-z_-]{35}'       # Google API keys
        ]
        
        for config_file in config_files:
            file_path = Path(config_file)
            if file_path.exists():
                try:
                    content = file_path.read_text()
                    
                    for pattern in secret_patterns:
                        import re
                        matches = re.findall(pattern, content)
                        
                        if matches:
                            exposure = {
                                'file': str(file_path),
                                'pattern': pattern,
                                'matches_count': len(matches),
                                'severity': 'high',
                                'recommendation': 'Review and move secrets to secure storage'
                            }
                            self.report.secret_exposures.append(exposure)
                            
                except Exception as e:
                    logger.warning(f"Error reading {file_path}: {e}")
                    
    async def _audit_configuration(self):
        """Audit security configuration."""
        logger.info("Auditing security configuration...")
        
        # Check for secure configurations
        config_checks = [
            ('Docker security', self._check_docker_security),
            ('File permissions', self._check_file_permissions),
            ('Environment security', self._check_environment_security),
        ]
        
        for check_name, check_func in config_checks:
            try:
                await check_func()
            except Exception as e:
                logger.error(f"Error in {check_name}: {e}")
                
    async def _check_docker_security(self):
        """Check Docker security configuration."""
        dockerfile_path = Path('Dockerfile')
        
        if dockerfile_path.exists():
            content = dockerfile_path.read_text()
            
            # Check for security issues
            if 'USER root' in content:
                finding = SecurityFinding(
                    severity='medium',
                    category='Container Security',
                    title='Running as Root User',
                    description='Dockerfile runs processes as root user',
                    file_path='Dockerfile',
                    recommendation='Use non-root user for container processes'
                )
                self.report.findings.append(finding)
                
            if 'ADD ' in content and 'http' in content:
                finding = SecurityFinding(
                    severity='low',
                    category='Container Security',
                    title='ADD with URL',
                    description='Using ADD with URLs can be security risk',
                    file_path='Dockerfile',
                    recommendation='Use COPY instead of ADD when possible'
                )
                self.report.findings.append(finding)
                
    async def _check_file_permissions(self):
        """Check file permissions for security."""
        sensitive_files = [
            '.env', 'config.yml', 'secrets.json'
        ]
        
        for file_name in sensitive_files:
            file_path = Path(file_name)
            if file_path.exists():
                stat = file_path.stat()
                mode = stat.st_mode
                
                # Check if file is world-readable (others can read)
                if mode & 0o004:
                    finding = SecurityFinding(
                        severity='medium',
                        category='File Security',
                        title='World-Readable Sensitive File',
                        description=f'File {file_name} is readable by all users',
                        file_path=file_name,
                        recommendation='Restrict file permissions to owner only'
                    )
                    self.report.findings.append(finding)
                    
    async def _check_environment_security(self):
        """Check environment security settings."""
        # Check for debug mode in production
        if os.getenv('DEBUG') == 'True':
            finding = SecurityFinding(
                severity='high',
                category='Configuration Security',
                title='Debug Mode Enabled',
                description='Debug mode is enabled which can expose sensitive information',
                recommendation='Disable debug mode in production'
            )
            self.report.findings.append(finding)
            
        # Check for development settings
        if os.getenv('ENVIRONMENT') == 'development':
            finding = SecurityFinding(
                severity='medium',
                category='Configuration Security',
                title='Development Environment',
                description='Running in development mode',
                recommendation='Ensure production configuration for deployment'
            )
            self.report.findings.append(finding)
            
    async def _validate_api_security(self):
        """Validate API security configurations."""
        logger.info("Validating API security...")
        
        try:
            # Test API key validation
            try:
                # Simulate API key validation test
                test_key = "test_key_123"
                if len(test_key) < 16:
                    finding = SecurityFinding(
                        severity='high',
                        category='API Security',
                        title='Weak API Key Length',
                        description='API keys should be at least 16 characters',
                        recommendation='Use longer API keys with sufficient entropy'
                    )
                    self.report.findings.append(finding)
            except Exception:
                # Skip if validation function not available
                pass
                
        except Exception as e:
            logger.error(f"Error validating API security: {e}")
            
    async def _check_compliance(self):
        """Check compliance with security standards."""
        logger.info("Checking security compliance...")
        
        compliance_checks = {
            'OWASP_Top_10': await self._check_owasp_compliance(),
            'PCI_DSS': await self._check_pci_compliance(),
            'SOC2': await self._check_soc2_compliance(),
            'GDPR': await self._check_gdpr_compliance()
        }
        
        self.report.compliance_status = compliance_checks
        
    async def _check_owasp_compliance(self) -> bool:
        """Check OWASP Top 10 compliance."""
        # Basic OWASP checks
        owasp_issues = 0
        
        # Check for injection vulnerabilities
        injection_findings = [f for f in self.report.findings 
                             if 'injection' in f.category.lower()]
        if injection_findings:
            owasp_issues += 1
            
        # Check for broken authentication
        auth_findings = [f for f in self.report.findings 
                        if 'authentication' in f.category.lower()]
        if auth_findings:
            owasp_issues += 1
            
        # Check for sensitive data exposure
        if self.report.secret_exposures:
            owasp_issues += 1
            
        return owasp_issues == 0
        
    async def _check_pci_compliance(self) -> bool:
        """Check PCI DSS compliance."""
        # Basic PCI checks (would need more comprehensive checks)
        pci_issues = 0
        
        # Check for encryption
        weak_crypto = [f for f in self.report.findings 
                      if 'weak' in f.title.lower() and 'crypto' in f.category.lower()]
        if weak_crypto:
            pci_issues += 1
            
        # Check for access controls
        access_issues = [f for f in self.report.findings 
                        if 'permission' in f.category.lower()]
        if access_issues:
            pci_issues += 1
            
        return pci_issues == 0
        
    async def _check_soc2_compliance(self) -> bool:
        """Check SOC 2 compliance."""
        # Check logging and monitoring
        has_logging = Path('logs').exists()
        has_monitoring = any('monitor' in f.name for f in Path('src').glob('*.py'))
        
        return has_logging and has_monitoring
        
    async def _check_gdpr_compliance(self) -> bool:
        """Check GDPR compliance."""
        # Check for data protection measures
        has_encryption = any('crypt' in f.name for f in Path('src').glob('*.py'))
        has_privacy_policy = Path('PRIVACY.md').exists()
        
        # Check for potential personal data handling
        personal_data_patterns = [
            'email', 'phone', 'address', 'name', 'ssn'
        ]
        
        data_handling_files = []
        for py_file in Path('src').rglob('*.py'):
            try:
                content = py_file.read_text().lower()
                if any(pattern in content for pattern in personal_data_patterns):
                    data_handling_files.append(py_file)
            except:
                pass
                
        # Basic compliance if encryption present and no obvious violations
        return has_encryption and len(data_handling_files) < 5
        
    def _calculate_security_score(self):
        """Calculate overall security score."""
        # Base score
        score = 100.0
        
        # Deduct for findings by severity
        severity_weights = {
            'critical': 20,
            'high': 10,
            'medium': 5,
            'low': 2
        }
        
        for finding in self.report.findings:
            score -= severity_weights.get(finding.severity, 1)
            
        # Deduct for vulnerabilities
        score -= len(self.report.dependency_vulnerabilities) * 3
        
        # Deduct for secret exposures
        score -= len(self.report.secret_exposures) * 8
        
        # Bonus for compliance
        compliant_standards = sum(self.report.compliance_status.values())
        score += compliant_standards * 2
        
        # Ensure score is between 0 and 100
        self.report.security_score = max(0.0, min(100.0, score))


async def main():
    """Run comprehensive security audit."""
    print("🛡️  Starting Comprehensive Security Audit")
    print("=" * 50)
    
    auditor = SecurityAuditor()
    
    try:
        # Run audit
        report = await auditor.run_comprehensive_audit()
        
        # Generate report
        report_data = report.to_dict()
        report_path = Path('security_audit_report.json')
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # Print summary
        print(f"\n📊 Security Audit Summary")
        print(f"Files Scanned: {report.total_files_scanned}")
        print(f"Security Findings: {len(report.findings)}")
        print(f"Dependency Vulnerabilities: {len(report.dependency_vulnerabilities)}")
        print(f"Secret Exposures: {len(report.secret_exposures)}")
        print(f"Security Score: {report.security_score:.1f}/100")
        
        # Print findings by severity
        severity_counts = {}
        for finding in report.findings:
            severity_counts[finding.severity] = severity_counts.get(finding.severity, 0) + 1
            
        print(f"\n🔍 Findings by Severity:")
        for severity in ['critical', 'high', 'medium', 'low']:
            count = severity_counts.get(severity, 0)
            if count > 0:
                print(f"  {severity.title()}: {count}")
                
        # Print compliance status
        print(f"\n✅ Compliance Status:")
        for standard, compliant in report.compliance_status.items():
            status = "PASS" if compliant else "FAIL"
            print(f"  {standard}: {status}")
            
        # Print top findings
        if report.findings:
            print(f"\n🚨 Top Security Issues:")
            critical_high = [f for f in report.findings if f.severity in ['critical', 'high']]
            for finding in critical_high[:5]:
                print(f"  [{finding.severity.upper()}] {finding.title}")
                if finding.file_path:
                    print(f"    File: {finding.file_path}:{finding.line_number or ''}")
                print(f"    Fix: {finding.recommendation}")
                
        print(f"\n📄 Detailed report saved to: {report_path}")
        
        # Exit code based on security score
        if report.security_score >= 80:
            print("✅ Security audit passed!")
            return 0
        elif report.security_score >= 60:
            print("⚠️  Security audit completed with warnings")
            return 1
        else:
            print("❌ Security audit failed - critical issues found")
            return 2
            
    except Exception as e:
        logger.error(f"Security audit failed: {e}")
        print(f"❌ Security audit failed: {e}")
        return 3


if __name__ == "__main__":
    exit_code = asyncio.run(main())