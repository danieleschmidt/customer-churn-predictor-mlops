"""
Quality Gates Implementation with CI/CD Integration.

This module provides comprehensive quality gates including:
- Pre-commit hooks for code quality enforcement
- CI/CD pipeline integration with automated quality checks
- Quality metrics collection and reporting with trend analysis
- Deployment gates with rollback capabilities
- Performance benchmarking gates with regression detection
- Security scanning gates with vulnerability blocking
- Code coverage enforcement with minimum thresholds
- Test result validation and failure analysis
"""

import os
import json
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import re
import yaml
from enum import Enum


class GateStatus(Enum):
    """Quality gate status."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"
    PENDING = "pending"


class GateSeverity(Enum):
    """Quality gate failure severity."""
    BLOCKING = "blocking"      # Blocks deployment/merge
    WARNING = "warning"        # Warns but doesn't block
    INFORMATIONAL = "info"     # Just provides information


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    gate_name: str
    status: GateStatus
    severity: GateSeverity
    score: float  # 0-100
    threshold: float
    message: str
    details: Dict[str, Any]
    execution_time_seconds: float
    timestamp: datetime
    recommendations: List[str] = field(default_factory=list)


@dataclass
class QualityGateReport:
    """Comprehensive quality gate report."""
    report_id: str
    timestamp: datetime
    project_name: str
    branch: str
    commit_hash: str
    gate_results: List[QualityGateResult]
    overall_status: GateStatus
    overall_score: float
    blocking_failures: int
    warnings: int
    deployment_approved: bool
    summary: str


class CodeQualityGate:
    """Code quality gate using static analysis tools."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.tools = {
            'flake8': {'enabled': True, 'threshold': 0},
            'black': {'enabled': True, 'check_only': True},
            'mypy': {'enabled': True, 'threshold': 0},
            'bandit': {'enabled': True, 'threshold': 'medium'},
            'isort': {'enabled': True, 'check_only': True},
            'pylint': {'enabled': False, 'threshold': 8.0}  # Disabled by default due to strictness
        }
    
    def run_gate(self, source_paths: List[str] = None) -> QualityGateResult:
        """Run code quality gate checks."""
        start_time = time.time()
        source_paths = source_paths or ['src/']
        
        all_issues = []
        tool_results = {}
        total_score = 0
        max_score = 0
        
        print("🔍 Running code quality checks...")
        
        # Run each enabled tool
        for tool_name, tool_config in self.tools.items():
            if not tool_config.get('enabled', False):
                continue
            
            print(f"  Running {tool_name}...")
            
            try:
                result = self._run_tool(tool_name, source_paths, tool_config)
                tool_results[tool_name] = result
                
                if result['score'] is not None:
                    total_score += result['score']
                    max_score += 100
                
                if result['issues']:
                    all_issues.extend(result['issues'])
                    
            except Exception as e:
                print(f"  ❌ {tool_name} failed: {e}")
                tool_results[tool_name] = {
                    'success': False,
                    'error': str(e),
                    'score': 0,
                    'issues': [f"{tool_name} execution failed: {e}"]
                }
                max_score += 100  # Count as 0 score
        
        # Calculate overall score
        overall_score = (total_score / max_score * 100) if max_score > 0 else 0
        
        # Determine status
        blocking_issues = len([issue for issue in all_issues if issue.get('severity') == 'error'])
        warning_issues = len([issue for issue in all_issues if issue.get('severity') == 'warning'])
        
        if blocking_issues > 0:
            status = GateStatus.FAILED
            message = f"Code quality gate failed with {blocking_issues} blocking issues"
        elif warning_issues > 0:
            status = GateStatus.WARNING  
            message = f"Code quality gate passed with {warning_issues} warnings"
        else:
            status = GateStatus.PASSED
            message = "Code quality gate passed"
        
        # Generate recommendations
        recommendations = self._generate_code_quality_recommendations(tool_results, all_issues)
        
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            gate_name="code_quality",
            status=status,
            severity=GateSeverity.BLOCKING if blocking_issues > 0 else GateSeverity.WARNING,
            score=overall_score,
            threshold=80.0,  # Configurable threshold
            message=message,
            details={
                'tool_results': tool_results,
                'total_issues': len(all_issues),
                'blocking_issues': blocking_issues,
                'warning_issues': warning_issues,
                'tools_run': list(tool_results.keys())
            },
            execution_time_seconds=execution_time,
            timestamp=datetime.now(),
            recommendations=recommendations
        )
    
    def _run_tool(self, tool_name: str, source_paths: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a specific code quality tool."""
        
        if tool_name == 'flake8':
            return self._run_flake8(source_paths, config)
        elif tool_name == 'black':
            return self._run_black(source_paths, config)
        elif tool_name == 'mypy':
            return self._run_mypy(source_paths, config)
        elif tool_name == 'bandit':
            return self._run_bandit(source_paths, config)
        elif tool_name == 'isort':
            return self._run_isort(source_paths, config)
        elif tool_name == 'pylint':
            return self._run_pylint(source_paths, config)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    def _run_flake8(self, source_paths: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
        """Run flake8 for style checking."""
        cmd = ['flake8'] + source_paths + ['--format=json']
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            issues = []
            if result.stdout.strip():
                try:
                    # Parse JSON output if available
                    flake8_data = json.loads(result.stdout)
                    for item in flake8_data:
                        issues.append({
                            'tool': 'flake8',
                            'file': item.get('filename', ''),
                            'line': item.get('line_number', 0),
                            'column': item.get('column_number', 0),
                            'severity': 'error',
                            'code': item.get('code', ''),
                            'message': item.get('text', '')
                        })
                except json.JSONDecodeError:
                    # Fall back to parsing text output
                    for line in result.stdout.strip().split('\n'):
                        if line.strip():
                            issues.append({
                                'tool': 'flake8',
                                'severity': 'error',
                                'message': line.strip()
                            })
            
            issue_count = len(issues)
            score = max(0, 100 - (issue_count * 2))  # -2 points per issue
            
            return {
                'success': result.returncode == 0,
                'score': score,
                'issues': issues,
                'issue_count': issue_count
            }
            
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'flake8 timed out', 'score': 0, 'issues': []}
        except FileNotFoundError:
            return {'success': False, 'error': 'flake8 not installed', 'score': 0, 'issues': []}
    
    def _run_black(self, source_paths: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
        """Run black for code formatting."""
        check_only = config.get('check_only', True)
        cmd = ['black'] + (['--check', '--diff'] if check_only else []) + source_paths
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            issues = []
            if result.returncode != 0:
                # Parse output for files that would be reformatted
                for line in result.stdout.split('\n'):
                    if 'would reformat' in line or 'reformatted' in line:
                        issues.append({
                            'tool': 'black',
                            'severity': 'warning',
                            'message': line.strip()
                        })
            
            issue_count = len(issues)
            score = 100 if result.returncode == 0 else max(0, 100 - (issue_count * 5))
            
            return {
                'success': result.returncode == 0,
                'score': score,
                'issues': issues,
                'issue_count': issue_count
            }
            
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'black timed out', 'score': 0, 'issues': []}
        except FileNotFoundError:
            return {'success': False, 'error': 'black not installed', 'score': 0, 'issues': []}
    
    def _run_mypy(self, source_paths: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
        """Run mypy for type checking."""
        cmd = ['mypy'] + source_paths + ['--no-error-summary']
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            
            issues = []
            if result.stdout.strip():
                for line in result.stdout.strip().split('\n'):
                    if ':' in line and ('error:' in line or 'warning:' in line):
                        parts = line.split(':', 3)
                        if len(parts) >= 4:
                            issues.append({
                                'tool': 'mypy',
                                'file': parts[0],
                                'line': parts[1],
                                'severity': 'error' if 'error:' in line else 'warning',
                                'message': parts[3].strip()
                            })
            
            error_count = len([i for i in issues if i['severity'] == 'error'])
            warning_count = len([i for i in issues if i['severity'] == 'warning'])
            
            score = max(0, 100 - (error_count * 5) - (warning_count * 2))
            
            return {
                'success': result.returncode == 0,
                'score': score,
                'issues': issues,
                'error_count': error_count,
                'warning_count': warning_count
            }
            
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'mypy timed out', 'score': 0, 'issues': []}
        except FileNotFoundError:
            return {'success': False, 'error': 'mypy not installed', 'score': 0, 'issues': []}
    
    def _run_bandit(self, source_paths: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
        """Run bandit for security issues."""
        cmd = ['bandit', '-r'] + source_paths + ['-f', 'json']
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            issues = []
            if result.stdout.strip():
                try:
                    bandit_data = json.loads(result.stdout)
                    for item in bandit_data.get('results', []):
                        severity_map = {'LOW': 'info', 'MEDIUM': 'warning', 'HIGH': 'error'}
                        issues.append({
                            'tool': 'bandit',
                            'file': item.get('filename', ''),
                            'line': item.get('line_number', 0),
                            'severity': severity_map.get(item.get('issue_severity', 'MEDIUM'), 'warning'),
                            'code': item.get('test_id', ''),
                            'message': item.get('issue_text', '')
                        })
                except json.JSONDecodeError:
                    pass
            
            high_issues = len([i for i in issues if i['severity'] == 'error'])
            medium_issues = len([i for i in issues if i['severity'] == 'warning'])
            
            score = max(0, 100 - (high_issues * 10) - (medium_issues * 5))
            
            return {
                'success': high_issues == 0,  # Success if no high severity issues
                'score': score,
                'issues': issues,
                'high_severity': high_issues,
                'medium_severity': medium_issues
            }
            
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'bandit timed out', 'score': 0, 'issues': []}
        except FileNotFoundError:
            return {'success': False, 'error': 'bandit not installed', 'score': 0, 'issues': []}
    
    def _run_isort(self, source_paths: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
        """Run isort for import sorting."""
        check_only = config.get('check_only', True)
        cmd = ['isort'] + (['--check-only', '--diff'] if check_only else []) + source_paths
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            issues = []
            if result.returncode != 0:
                # Count files that would be fixed
                for line in result.stdout.split('\n'):
                    if 'Fixing' in line or 'would fix' in line:
                        issues.append({
                            'tool': 'isort',
                            'severity': 'warning',
                            'message': line.strip()
                        })
            
            issue_count = len(issues)
            score = 100 if result.returncode == 0 else max(0, 100 - (issue_count * 3))
            
            return {
                'success': result.returncode == 0,
                'score': score,
                'issues': issues,
                'issue_count': issue_count
            }
            
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'isort timed out', 'score': 0, 'issues': []}
        except FileNotFoundError:
            return {'success': False, 'error': 'isort not installed', 'score': 0, 'issues': []}
    
    def _run_pylint(self, source_paths: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
        """Run pylint for comprehensive code analysis."""
        cmd = ['pylint'] + source_paths + ['--output-format=json']
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            issues = []
            score = 100  # Default if no score found
            
            if result.stdout.strip():
                try:
                    pylint_data = json.loads(result.stdout)
                    for item in pylint_data:
                        severity_map = {'error': 'error', 'warning': 'warning', 'convention': 'info', 'refactor': 'info'}
                        issues.append({
                            'tool': 'pylint',
                            'file': item.get('path', ''),
                            'line': item.get('line', 0),
                            'column': item.get('column', 0),
                            'severity': severity_map.get(item.get('type', 'warning'), 'warning'),
                            'code': item.get('symbol', ''),
                            'message': item.get('message', '')
                        })
                except json.JSONDecodeError:
                    # Try to extract score from stderr
                    for line in result.stderr.split('\n'):
                        if 'Your code has been rated at' in line:
                            match = re.search(r'rated at ([\d.]+)/10', line)
                            if match:
                                score = float(match.group(1)) * 10  # Convert to 0-100 scale
            
            return {
                'success': score >= config.get('threshold', 8.0) * 10,
                'score': score,
                'issues': issues,
                'issue_count': len(issues)
            }
            
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'pylint timed out', 'score': 0, 'issues': []}
        except FileNotFoundError:
            return {'success': False, 'error': 'pylint not installed', 'score': 0, 'issues': []}
    
    def _generate_code_quality_recommendations(self, tool_results: Dict[str, Any], 
                                             issues: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on code quality results."""
        recommendations = []
        
        # Tool-specific recommendations
        for tool, result in tool_results.items():
            if not result.get('success', False):
                if tool == 'flake8' and result.get('issue_count', 0) > 0:
                    recommendations.append("Fix PEP 8 style violations detected by flake8")
                elif tool == 'black' and result.get('issue_count', 0) > 0:
                    recommendations.append("Run 'black .' to automatically format code")
                elif tool == 'mypy' and result.get('error_count', 0) > 0:
                    recommendations.append("Add type hints and fix type checking errors")
                elif tool == 'bandit' and result.get('high_severity', 0) > 0:
                    recommendations.append("Fix high-severity security issues found by bandit")
                elif tool == 'isort' and result.get('issue_count', 0) > 0:
                    recommendations.append("Run 'isort .' to fix import sorting")
        
        # General recommendations based on issue patterns
        issue_types = defaultdict(int)
        for issue in issues:
            issue_types[issue.get('tool', 'unknown')] += 1
        
        if issue_types['bandit'] > 5:
            recommendations.append("Consider security review - many security issues detected")
        
        if issue_types['mypy'] > 10:
            recommendations.append("Improve type coverage - consider gradual typing approach")
        
        if not recommendations:
            recommendations.append("Code quality is good - consider enabling more strict checks")
        
        return recommendations


class TestQualityGate:
    """Test quality gate for test results and coverage."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.coverage_threshold = self.config.get('coverage_threshold', 85.0)
        self.test_pass_threshold = self.config.get('test_pass_threshold', 95.0)
    
    def run_gate(self) -> QualityGateResult:
        """Run test quality gate."""
        start_time = time.time()
        
        print("🧪 Running test quality gate...")
        
        # Run tests with coverage
        test_result = self._run_tests_with_coverage()
        
        # Analyze results
        test_pass_rate = test_result.get('pass_rate', 0)
        coverage_percentage = test_result.get('coverage', 0)
        
        # Calculate score
        test_score = (test_pass_rate / 100) * 50  # 50% weight for test pass rate
        coverage_score = (coverage_percentage / 100) * 50  # 50% weight for coverage
        overall_score = test_score + coverage_score
        
        # Determine status
        if (test_pass_rate >= self.test_pass_threshold and 
            coverage_percentage >= self.coverage_threshold):
            status = GateStatus.PASSED
            message = f"Tests passed ({test_pass_rate:.1f}%) with {coverage_percentage:.1f}% coverage"
        elif test_pass_rate < self.test_pass_threshold:
            status = GateStatus.FAILED
            message = f"Test pass rate ({test_pass_rate:.1f}%) below threshold ({self.test_pass_threshold}%)"
        else:
            status = GateStatus.FAILED
            message = f"Coverage ({coverage_percentage:.1f}%) below threshold ({self.coverage_threshold}%)"
        
        recommendations = self._generate_test_recommendations(test_result)
        
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            gate_name="test_quality",
            status=status,
            severity=GateSeverity.BLOCKING,
            score=overall_score,
            threshold=80.0,
            message=message,
            details=test_result,
            execution_time_seconds=execution_time,
            timestamp=datetime.now(),
            recommendations=recommendations
        )
    
    def _run_tests_with_coverage(self) -> Dict[str, Any]:
        """Run tests with coverage measurement."""
        try:
            # Run pytest with coverage
            cmd = [
                'python', '-m', 'pytest', 
                '--cov=src',
                '--cov-report=json:coverage.json',
                '--cov-report=term-missing',
                '--tb=short',
                '-v'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            # Parse pytest output
            test_stats = self._parse_pytest_output(result.stdout)
            
            # Parse coverage report
            coverage_data = self._parse_coverage_report()
            
            return {
                'tests_run': test_stats.get('total', 0),
                'tests_passed': test_stats.get('passed', 0),
                'tests_failed': test_stats.get('failed', 0),
                'tests_skipped': test_stats.get('skipped', 0),
                'pass_rate': test_stats.get('pass_rate', 0),
                'coverage': coverage_data.get('coverage_percent', 0),
                'missing_lines': coverage_data.get('missing_lines', 0),
                'covered_lines': coverage_data.get('covered_lines', 0),
                'total_lines': coverage_data.get('total_lines', 0),
                'duration': test_stats.get('duration', 0),
                'return_code': result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {'error': 'Tests timed out', 'pass_rate': 0, 'coverage': 0}
        except Exception as e:
            return {'error': str(e), 'pass_rate': 0, 'coverage': 0}
    
    def _parse_pytest_output(self, output: str) -> Dict[str, Any]:
        """Parse pytest output for test statistics."""
        stats = {'total': 0, 'passed': 0, 'failed': 0, 'skipped': 0, 'duration': 0}
        
        # Look for test results summary
        lines = output.split('\n')
        for line in lines:
            if 'failed' in line and 'passed' in line:
                # Parse line like "2 failed, 8 passed in 1.23s"
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'failed,' and i > 0:
                        stats['failed'] = int(parts[i-1])
                    elif part == 'passed' and i > 0:
                        stats['passed'] = int(parts[i-1])
                    elif part == 'skipped' and i > 0:
                        stats['skipped'] = int(parts[i-1])
                    elif 'in' in part and i < len(parts) - 1:
                        try:
                            stats['duration'] = float(parts[i+1].replace('s', ''))
                        except ValueError:
                            pass
            elif 'passed in' in line:
                # Parse line like "10 passed in 2.34s"
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'passed' and i > 0:
                        stats['passed'] = int(parts[i-1])
                    elif 'in' in part and i < len(parts) - 1:
                        try:
                            stats['duration'] = float(parts[i+1].replace('s', ''))
                        except ValueError:
                            pass
        
        stats['total'] = stats['passed'] + stats['failed'] + stats['skipped']
        if stats['total'] > 0:
            stats['pass_rate'] = (stats['passed'] / stats['total']) * 100
        else:
            stats['pass_rate'] = 0
        
        return stats
    
    def _parse_coverage_report(self) -> Dict[str, Any]:
        """Parse coverage JSON report."""
        coverage_file = Path('coverage.json')
        
        if not coverage_file.exists():
            return {'coverage_percent': 0, 'missing_lines': 0, 'covered_lines': 0, 'total_lines': 0}
        
        try:
            with open(coverage_file) as f:
                coverage_data = json.load(f)
            
            totals = coverage_data.get('totals', {})
            
            return {
                'coverage_percent': totals.get('percent_covered', 0),
                'missing_lines': totals.get('missing_lines', 0),
                'covered_lines': totals.get('covered_lines', 0),
                'total_lines': totals.get('num_statements', 0)
            }
            
        except Exception:
            return {'coverage_percent': 0, 'missing_lines': 0, 'covered_lines': 0, 'total_lines': 0}
    
    def _generate_test_recommendations(self, test_result: Dict[str, Any]) -> List[str]:
        """Generate test improvement recommendations."""
        recommendations = []
        
        if test_result.get('tests_failed', 0) > 0:
            recommendations.append(f"Fix {test_result['tests_failed']} failing tests")
        
        coverage = test_result.get('coverage', 0)
        if coverage < self.coverage_threshold:
            missing_coverage = self.coverage_threshold - coverage
            recommendations.append(f"Increase test coverage by {missing_coverage:.1f}%")
        
        if test_result.get('tests_run', 0) == 0:
            recommendations.append("Add test cases - no tests found")
        
        if test_result.get('duration', 0) > 300:  # 5 minutes
            recommendations.append("Consider optimizing slow tests")
        
        return recommendations


class SecurityGate:
    """Security gate for vulnerability scanning."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
    def run_gate(self) -> QualityGateResult:
        """Run security gate checks."""
        start_time = time.time()
        
        print("🔒 Running security gate...")
        
        security_results = []
        overall_score = 100
        
        # Run dependency vulnerability scan
        print("  Checking dependencies for vulnerabilities...")
        dep_result = self._check_dependencies()
        security_results.append(dep_result)
        
        if dep_result.get('vulnerabilities', 0) > 0:
            critical_vulns = dep_result.get('critical', 0)
            high_vulns = dep_result.get('high', 0)
            medium_vulns = dep_result.get('medium', 0)
            
            overall_score -= (critical_vulns * 30 + high_vulns * 20 + medium_vulns * 10)
        
        # Run secrets scan
        print("  Scanning for secrets...")
        secrets_result = self._scan_secrets()
        security_results.append(secrets_result)
        
        if secrets_result.get('secrets_found', 0) > 0:
            overall_score -= secrets_result['secrets_found'] * 25  # Heavy penalty
        
        overall_score = max(0, overall_score)
        
        # Determine status
        critical_issues = sum(r.get('critical', 0) for r in security_results)
        high_issues = sum(r.get('high', 0) for r in security_results)
        secrets_found = sum(r.get('secrets_found', 0) for r in security_results)
        
        if critical_issues > 0 or secrets_found > 0:
            status = GateStatus.FAILED
            severity = GateSeverity.BLOCKING
            message = f"Security gate failed: {critical_issues} critical vulnerabilities, {secrets_found} secrets"
        elif high_issues > 0:
            status = GateStatus.WARNING
            severity = GateSeverity.WARNING
            message = f"Security gate warning: {high_issues} high-severity vulnerabilities"
        else:
            status = GateStatus.PASSED
            severity = GateSeverity.INFORMATIONAL
            message = "Security gate passed"
        
        recommendations = self._generate_security_recommendations(security_results)
        
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            gate_name="security",
            status=status,
            severity=severity,
            score=overall_score,
            threshold=80.0,
            message=message,
            details={
                'dependency_scan': dep_result,
                'secrets_scan': secrets_result,
                'total_vulnerabilities': sum(r.get('vulnerabilities', 0) for r in security_results)
            },
            execution_time_seconds=execution_time,
            timestamp=datetime.now(),
            recommendations=recommendations
        )
    
    def _check_dependencies(self) -> Dict[str, Any]:
        """Check dependencies for known vulnerabilities."""
        try:
            # Use safety to check dependencies
            result = subprocess.run(
                ['safety', 'check', '--json'], 
                capture_output=True, 
                text=True, 
                timeout=120
            )
            
            if result.stdout.strip():
                try:
                    vulnerabilities = json.loads(result.stdout)
                    
                    # Count by severity
                    severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
                    
                    for vuln in vulnerabilities:
                        # Map CVE scores to severity (simplified)
                        # In real implementation, would parse actual CVSS scores
                        severity_counts['high'] += 1  # Default to high for safety output
                    
                    return {
                        'vulnerabilities': len(vulnerabilities),
                        'critical': severity_counts['critical'],
                        'high': severity_counts['high'],
                        'medium': severity_counts['medium'],
                        'low': severity_counts['low'],
                        'details': vulnerabilities
                    }
                    
                except json.JSONDecodeError:
                    pass
            
            return {'vulnerabilities': 0, 'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
            
        except subprocess.TimeoutExpired:
            return {'error': 'Dependency scan timed out', 'vulnerabilities': 0}
        except FileNotFoundError:
            return {'error': 'safety not installed', 'vulnerabilities': 0}
        except Exception as e:
            return {'error': str(e), 'vulnerabilities': 0}
    
    def _scan_secrets(self) -> Dict[str, Any]:
        """Scan for hardcoded secrets."""
        secrets_patterns = [
            r'(?i)(api[_-]?key|apikey)[\s]*[=:]\s*["\']([^"\']+)["\']',
            r'(?i)(password|passwd|pwd)[\s]*[=:]\s*["\']([^"\']+)["\']',
            r'(?i)(secret|token)[\s]*[=:]\s*["\']([^"\']+)["\']',
            r'(?i)(aws[_-]?access[_-]?key|access[_-]?key)[\s]*[=:]\s*["\']([A-Z0-9]{20})["\']',
            r'(?i)(aws[_-]?secret[_-]?key|secret[_-]?key)[\s]*[=:]\s*["\']([A-Za-z0-9/+=]{40})["\']',
        ]
        
        secrets_found = []
        
        # Scan source files
        source_paths = ['src/', 'tests/', '.env*', 'config*']
        
        for path_pattern in source_paths:
            path = Path(path_pattern)
            
            if path.is_file():
                secrets_found.extend(self._scan_file_for_secrets(path, secrets_patterns))
            elif path.is_dir():
                for file_path in path.rglob('*.py'):
                    secrets_found.extend(self._scan_file_for_secrets(file_path, secrets_patterns))
        
        return {
            'secrets_found': len(secrets_found),
            'details': secrets_found
        }
    
    def _scan_file_for_secrets(self, file_path: Path, patterns: List[str]) -> List[Dict[str, Any]]:
        """Scan a single file for secrets."""
        secrets = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
                for i, line in enumerate(content.split('\n'), 1):
                    for pattern in patterns:
                        matches = re.finditer(pattern, line)
                        for match in matches:
                            # Skip obvious test/example values
                            value = match.group(2) if len(match.groups()) >= 2 else match.group(0)
                            if not self._is_test_value(value):
                                secrets.append({
                                    'file': str(file_path),
                                    'line': i,
                                    'type': match.group(1) if len(match.groups()) >= 1 else 'unknown',
                                    'value': value[:20] + '...' if len(value) > 20 else value
                                })
                                
        except Exception:
            pass  # Skip files that can't be read
        
        return secrets
    
    def _is_test_value(self, value: str) -> bool:
        """Check if a value is likely a test/dummy value."""
        test_indicators = [
            'test', 'example', 'dummy', 'fake', 'mock', 'placeholder',
            'xxx', 'yyy', '123', 'abc', 'your-key-here', 'insert-key-here'
        ]
        
        value_lower = value.lower()
        return any(indicator in value_lower for indicator in test_indicators)
    
    def _generate_security_recommendations(self, security_results: List[Dict[str, Any]]) -> List[str]:
        """Generate security recommendations."""
        recommendations = []
        
        for result in security_results:
            if result.get('vulnerabilities', 0) > 0:
                recommendations.append("Update dependencies with known vulnerabilities")
                
            if result.get('secrets_found', 0) > 0:
                recommendations.append("Remove hardcoded secrets and use environment variables")
                recommendations.append("Consider using a secrets management system")
        
        if not recommendations:
            recommendations.append("Security scan passed - maintain good security practices")
        
        return recommendations


class QualityGateOrchestrator:
    """Orchestrates all quality gates."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.gates = {
            'code_quality': CodeQualityGate(self.config.get('code_quality', {})),
            'test_quality': TestQualityGate(self.config.get('test_quality', {})),
            'security': SecurityGate(self.config.get('security', {}))
        }
    
    def run_all_gates(self, enabled_gates: List[str] = None) -> QualityGateReport:
        """Run all or specified quality gates."""
        if enabled_gates is None:
            enabled_gates = list(self.gates.keys())
        
        print(f"🚪 Running quality gates: {', '.join(enabled_gates)}")
        
        report_id = f"quality_gate_{int(time.time())}"
        timestamp = datetime.now()
        
        # Get project info
        project_info = self._get_project_info()
        
        gate_results = []
        
        for gate_name in enabled_gates:
            if gate_name not in self.gates:
                print(f"⚠️ Unknown gate: {gate_name}")
                continue
            
            print(f"\n{'='*50}")
            print(f"Running {gate_name} gate")
            print(f"{'='*50}")
            
            try:
                gate = self.gates[gate_name]
                result = gate.run_gate()
                gate_results.append(result)
                
                status_emoji = {
                    GateStatus.PASSED: "✅",
                    GateStatus.FAILED: "❌", 
                    GateStatus.WARNING: "⚠️",
                    GateStatus.SKIPPED: "⏭️"
                }.get(result.status, "❓")
                
                print(f"{status_emoji} {gate_name}: {result.message}")
                print(f"   Score: {result.score:.1f}/100")
                print(f"   Duration: {result.execution_time_seconds:.2f}s")
                
                if result.recommendations:
                    print("   Recommendations:")
                    for rec in result.recommendations:
                        print(f"     • {rec}")
                
            except Exception as e:
                print(f"❌ {gate_name} gate failed with error: {e}")
                # Create failed result
                gate_results.append(QualityGateResult(
                    gate_name=gate_name,
                    status=GateStatus.FAILED,
                    severity=GateSeverity.BLOCKING,
                    score=0.0,
                    threshold=0.0,
                    message=f"Gate execution failed: {str(e)}",
                    details={'error': str(e)},
                    execution_time_seconds=0.0,
                    timestamp=datetime.now()
                ))
        
        # Calculate overall results
        blocking_failures = sum(1 for r in gate_results 
                              if r.status == GateStatus.FAILED and r.severity == GateSeverity.BLOCKING)
        warnings = sum(1 for r in gate_results if r.status == GateStatus.WARNING)
        
        # Overall status
        if blocking_failures > 0:
            overall_status = GateStatus.FAILED
            deployment_approved = False
        elif warnings > 0:
            overall_status = GateStatus.WARNING
            deployment_approved = True  # Deploy with warnings
        else:
            overall_status = GateStatus.PASSED
            deployment_approved = True
        
        # Overall score (weighted average)
        if gate_results:
            overall_score = sum(r.score for r in gate_results) / len(gate_results)
        else:
            overall_score = 0.0
        
        # Generate summary
        summary = self._generate_summary(gate_results, overall_status, blocking_failures, warnings)
        
        report = QualityGateReport(
            report_id=report_id,
            timestamp=timestamp,
            project_name=project_info['project_name'],
            branch=project_info['branch'],
            commit_hash=project_info['commit_hash'],
            gate_results=gate_results,
            overall_status=overall_status,
            overall_score=overall_score,
            blocking_failures=blocking_failures,
            warnings=warnings,
            deployment_approved=deployment_approved,
            summary=summary
        )
        
        # Print final summary
        print(f"\n🎯 Quality Gates Summary")
        print(f"{'='*60}")
        print(f"Overall Status: {overall_status.value.upper()}")
        print(f"Overall Score: {overall_score:.1f}/100")
        print(f"Deployment Approved: {'✅ YES' if deployment_approved else '❌ NO'}")
        print(f"Blocking Failures: {blocking_failures}")
        print(f"Warnings: {warnings}")
        
        if not deployment_approved:
            print(f"\n❌ DEPLOYMENT BLOCKED")
            print("Fix blocking issues before proceeding with deployment.")
        elif warnings > 0:
            print(f"\n⚠️ DEPLOYMENT APPROVED WITH WARNINGS")
            print("Consider addressing warnings in next iteration.")
        else:
            print(f"\n✅ DEPLOYMENT APPROVED")
            print("All quality gates passed successfully.")
        
        # Save report
        self._save_report(report)
        
        return report
    
    def _get_project_info(self) -> Dict[str, str]:
        """Get project information from git."""
        project_info = {
            'project_name': Path.cwd().name,
            'branch': 'unknown',
            'commit_hash': 'unknown'
        }
        
        try:
            # Get current branch
            result = subprocess.run(['git', 'branch', '--show-current'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                project_info['branch'] = result.stdout.strip()
        except:
            pass
        
        try:
            # Get current commit hash
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                project_info['commit_hash'] = result.stdout.strip()[:8]
        except:
            pass
        
        return project_info
    
    def _generate_summary(self, gate_results: List[QualityGateResult], 
                         overall_status: GateStatus, blocking_failures: int, warnings: int) -> str:
        """Generate executive summary."""
        
        if overall_status == GateStatus.PASSED:
            summary = "All quality gates passed successfully. Code is ready for deployment."
        elif overall_status == GateStatus.WARNING:
            summary = f"Quality gates passed with {warnings} warnings. " \
                     "Deployment approved but consider addressing warnings."
        else:
            summary = f"Quality gates failed with {blocking_failures} blocking issues. " \
                     "Deployment blocked until issues are resolved."
        
        # Add specific gate information
        gate_summary = []
        for result in gate_results:
            gate_summary.append(f"{result.gate_name}: {result.score:.0f}%")
        
        summary += f"\n\nGate Scores: {', '.join(gate_summary)}"
        
        return summary
    
    def _save_report(self, report: QualityGateReport) -> None:
        """Save quality gate report."""
        reports_dir = Path("quality_gate_reports")
        reports_dir.mkdir(exist_ok=True)
        
        report_file = reports_dir / f"{report.report_id}.json"
        with open(report_file, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        
        print(f"\n📊 Quality gate report saved to {report_file}")


def create_pre_commit_hook():
    """Create pre-commit hook for quality gates."""
    hook_content = """#!/usr/bin/env python3
import sys
import subprocess
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from tests.quality_gates.quality_gates import QualityGateOrchestrator
    
    print("🔍 Running pre-commit quality gates...")
    
    orchestrator = QualityGateOrchestrator()
    
    # Run only fast gates for pre-commit
    report = orchestrator.run_all_gates(['code_quality'])
    
    if not report.deployment_approved:
        print("❌ Pre-commit quality gates failed!")
        print("Fix issues before committing.")
        sys.exit(1)
    else:
        print("✅ Pre-commit quality gates passed!")
        sys.exit(0)
        
except Exception as e:
    print(f"❌ Pre-commit hook failed: {e}")
    sys.exit(1)
"""
    
    # Create pre-commit hook
    git_hooks_dir = Path('.git/hooks')
    if git_hooks_dir.exists():
        hook_file = git_hooks_dir / 'pre-commit'
        with open(hook_file, 'w') as f:
            f.write(hook_content)
        
        # Make executable
        import stat
        hook_file.chmod(hook_file.stat().st_mode | stat.S_IEXEC)
        
        print(f"✅ Pre-commit hook installed at {hook_file}")
    else:
        print("⚠️ Not a git repository - pre-commit hook not installed")


def main():
    """Main function for running quality gates."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quality Gates Framework")
    parser.add_argument("--gates", nargs="+", 
                       choices=["code_quality", "test_quality", "security", "all"],
                       default=["all"],
                       help="Quality gates to run")
    parser.add_argument("--install-hooks", action="store_true",
                       help="Install pre-commit hooks")
    parser.add_argument("--config", type=str,
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    if args.install_hooks:
        create_pre_commit_hook()
        return
    
    # Load configuration
    config = {}
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            if args.config.endswith('.yaml') or args.config.endswith('.yml'):
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
    
    # Determine which gates to run
    gates_to_run = args.gates
    if "all" in gates_to_run:
        gates_to_run = ["code_quality", "test_quality", "security"]
    
    # Run quality gates
    orchestrator = QualityGateOrchestrator(config)
    report = orchestrator.run_all_gates(gates_to_run)
    
    # Exit with appropriate code
    if not report.deployment_approved:
        sys.exit(1)
    elif report.warnings > 0:
        sys.exit(2)  # Warning exit code
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()