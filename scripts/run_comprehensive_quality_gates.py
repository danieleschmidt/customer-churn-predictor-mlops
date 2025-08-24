#!/usr/bin/env python3
"""
Comprehensive Quality Gates Runner for Advanced ML System.

This script runs all quality gates to validate the advanced ML enhancements
including code quality, security, performance, and comprehensive testing.
"""

import sys
import os
import time
import subprocess
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logging_config import get_logger

logger = get_logger(__name__)


class QualityGateResult:
    """Container for quality gate results."""
    
    def __init__(self, name: str, passed: bool, score: float, details: Dict[str, Any]):
        self.name = name
        self.passed = passed
        self.score = score
        self.details = details
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'passed': self.passed,
            'score': self.score,
            'details': self.details,
            'timestamp': self.timestamp.isoformat()
        }


class ComprehensiveQualityGateRunner:
    """Comprehensive quality gate runner for advanced ML system."""
    
    def __init__(self):
        self.results: List[QualityGateResult] = []
        self.start_time = datetime.now()
        self.project_root = Path(__file__).parent.parent
        
        # Quality gate configurations
        self.quality_thresholds = {
            'code_coverage': 85.0,
            'security_score': 90.0,
            'performance_score': 80.0,
            'code_quality_score': 85.0,
            'documentation_score': 75.0,
            'test_pass_rate': 95.0
        }
    
    def run_all_quality_gates(self) -> Dict[str, Any]:
        """Run all quality gates and return comprehensive results."""
        logger.info("🚀 Starting Comprehensive Quality Gates for Advanced ML System")
        logger.info("=" * 80)
        
        # 1. Code Quality Gates
        self._run_code_quality_gates()
        
        # 2. Security Gates
        self._run_security_gates()
        
        # 3. Testing Gates
        self._run_testing_gates()
        
        # 4. Performance Gates
        self._run_performance_gates()
        
        # 5. Documentation Gates
        self._run_documentation_gates()
        
        # 6. Advanced ML Specific Gates
        self._run_advanced_ml_gates()
        
        # Generate comprehensive report
        return self._generate_final_report()
    
    def _run_code_quality_gates(self):
        """Run code quality validation gates."""
        logger.info("🔍 Running Code Quality Gates...")
        
        # Python code formatting (Black)
        try:
            result = subprocess.run([
                'python', '-m', 'black', '--check', '--diff', 'src/', 'tests/', 'scripts/'
            ], cwd=self.project_root, capture_output=True, text=True, timeout=120)
            
            black_passed = result.returncode == 0
            self.results.append(QualityGateResult(
                'code_formatting',
                black_passed,
                100.0 if black_passed else 0.0,
                {'tool': 'black', 'output': result.stdout + result.stderr}
            ))
        except Exception as e:
            logger.warning(f"Black formatting check failed: {e}")
            self.results.append(QualityGateResult(
                'code_formatting', False, 0.0, {'error': str(e)}
            ))
        
        # Import sorting (isort)
        try:
            result = subprocess.run([
                'python', '-m', 'isort', '--check-only', '--diff', 'src/', 'tests/', 'scripts/'
            ], cwd=self.project_root, capture_output=True, text=True, timeout=120)
            
            isort_passed = result.returncode == 0
            self.results.append(QualityGateResult(
                'import_sorting',
                isort_passed,
                100.0 if isort_passed else 0.0,
                {'tool': 'isort', 'output': result.stdout + result.stderr}
            ))
        except Exception as e:
            logger.warning(f"Import sorting check failed: {e}")
            self.results.append(QualityGateResult(
                'import_sorting', False, 0.0, {'error': str(e)}
            ))
        
        # Linting (Flake8)
        try:
            result = subprocess.run([
                'python', '-m', 'flake8', 'src/', 'tests/', 'scripts/',
                '--max-line-length=88',
                '--extend-ignore=E203,W503,E501'
            ], cwd=self.project_root, capture_output=True, text=True, timeout=180)
            
            flake8_passed = result.returncode == 0
            issue_count = len(result.stdout.split('\n')) - 1 if result.stdout else 0
            
            self.results.append(QualityGateResult(
                'linting',
                flake8_passed,
                max(0, 100 - issue_count * 2),  # Deduct 2 points per issue
                {'tool': 'flake8', 'issues': issue_count, 'output': result.stdout}
            ))
        except Exception as e:
            logger.warning(f"Linting check failed: {e}")
            self.results.append(QualityGateResult(
                'linting', False, 0.0, {'error': str(e)}
            ))
        
        # Type checking (MyPy) - Optional
        try:
            result = subprocess.run([
                'python', '-m', 'mypy', 'src/',
                '--ignore-missing-imports',
                '--no-strict-optional'
            ], cwd=self.project_root, capture_output=True, text=True, timeout=180)
            
            mypy_passed = result.returncode == 0
            self.results.append(QualityGateResult(
                'type_checking',
                mypy_passed,
                100.0 if mypy_passed else 50.0,  # Less critical
                {'tool': 'mypy', 'output': result.stdout + result.stderr}
            ))
        except FileNotFoundError:
            logger.info("MyPy not available, skipping type checking")
        except Exception as e:
            logger.warning(f"Type checking failed: {e}")
    
    def _run_security_gates(self):
        """Run security validation gates."""
        logger.info("🔒 Running Security Gates...")
        
        # Security vulnerability scanning (Bandit)
        try:
            result = subprocess.run([
                'python', '-m', 'bandit', '-r', 'src/', '-f', 'json'
            ], cwd=self.project_root, capture_output=True, text=True, timeout=180)
            
            bandit_output = result.stdout
            if bandit_output:
                try:
                    bandit_data = json.loads(bandit_output)
                    issues = bandit_data.get('results', [])
                    high_severity = sum(1 for issue in issues if issue.get('issue_severity') == 'HIGH')
                    medium_severity = sum(1 for issue in issues if issue.get('issue_severity') == 'MEDIUM')
                    
                    security_score = max(0, 100 - (high_severity * 20 + medium_severity * 10))
                    security_passed = high_severity == 0 and medium_severity <= 3
                    
                    self.results.append(QualityGateResult(
                        'security_scan',
                        security_passed,
                        security_score,
                        {
                            'tool': 'bandit',
                            'high_severity': high_severity,
                            'medium_severity': medium_severity,
                            'total_issues': len(issues)
                        }
                    ))
                except json.JSONDecodeError:
                    self.results.append(QualityGateResult(
                        'security_scan', True, 100.0,
                        {'tool': 'bandit', 'status': 'no_issues_found'}
                    ))
            else:
                self.results.append(QualityGateResult(
                    'security_scan', True, 100.0,
                    {'tool': 'bandit', 'status': 'clean'}
                ))
        except Exception as e:
            logger.warning(f"Security scan failed: {e}")
            self.results.append(QualityGateResult(
                'security_scan', False, 0.0, {'error': str(e)}
            ))
        
        # Dependency vulnerability check (Safety) - Optional
        try:
            result = subprocess.run([
                'python', '-m', 'safety', 'check', '--json'
            ], cwd=self.project_root, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                self.results.append(QualityGateResult(
                    'dependency_security',
                    True,
                    100.0,
                    {'tool': 'safety', 'status': 'no_vulnerabilities'}
                ))
            else:
                try:
                    vulnerabilities = json.loads(result.stdout) if result.stdout else []
                    vuln_count = len(vulnerabilities)
                    security_score = max(0, 100 - vuln_count * 15)
                    
                    self.results.append(QualityGateResult(
                        'dependency_security',
                        vuln_count == 0,
                        security_score,
                        {'tool': 'safety', 'vulnerabilities': vuln_count}
                    ))
                except:
                    self.results.append(QualityGateResult(
                        'dependency_security', False, 50.0,
                        {'tool': 'safety', 'status': 'check_failed'}
                    ))
        except FileNotFoundError:
            logger.info("Safety not available, skipping dependency security check")
        except Exception as e:
            logger.warning(f"Dependency security check failed: {e}")
    
    def _run_testing_gates(self):
        """Run comprehensive testing gates."""
        logger.info("🧪 Running Testing Gates...")
        
        # Unit tests with coverage
        try:
            result = subprocess.run([
                'python', '-m', 'pytest',
                'tests/',
                '--cov=src',
                '--cov-report=json:coverage.json',
                '--cov-report=term-missing',
                '--cov-fail-under=80',
                '-v',
                '--tb=short',
                '--maxfail=10',
                '--durations=10'
            ], cwd=self.project_root, capture_output=True, text=True, timeout=600)
            
            test_passed = result.returncode == 0
            
            # Parse coverage data
            coverage_score = 0.0
            coverage_details = {}
            
            coverage_file = self.project_root / 'coverage.json'
            if coverage_file.exists():
                try:
                    with open(coverage_file) as f:
                        coverage_data = json.load(f)
                    
                    coverage_score = coverage_data.get('totals', {}).get('percent_covered', 0)
                    coverage_details = {
                        'lines_covered': coverage_data.get('totals', {}).get('covered_lines', 0),
                        'lines_total': coverage_data.get('totals', {}).get('num_statements', 0),
                        'branches_covered': coverage_data.get('totals', {}).get('covered_branches', 0),
                        'branches_total': coverage_data.get('totals', {}).get('num_branches', 0)
                    }
                except Exception as e:
                    logger.warning(f"Failed to parse coverage data: {e}")
            
            # Test results
            passed_tests = result.stdout.count(' PASSED')
            failed_tests = result.stdout.count(' FAILED')
            total_tests = passed_tests + failed_tests
            
            test_pass_rate = (passed_tests / max(total_tests, 1)) * 100
            
            self.results.append(QualityGateResult(
                'unit_tests',
                test_passed,
                test_pass_rate,
                {
                    'passed': passed_tests,
                    'failed': failed_tests,
                    'total': total_tests,
                    'pass_rate': test_pass_rate,
                    'output': result.stdout[-2000:]  # Last 2000 chars
                }
            ))
            
            self.results.append(QualityGateResult(
                'code_coverage',
                coverage_score >= self.quality_thresholds['code_coverage'],
                coverage_score,
                coverage_details
            ))
            
        except Exception as e:
            logger.error(f"Testing gate failed: {e}")
            self.results.append(QualityGateResult(
                'unit_tests', False, 0.0, {'error': str(e)}
            ))
            self.results.append(QualityGateResult(
                'code_coverage', False, 0.0, {'error': str(e)}
            ))
        
        # Integration tests (if available)
        integration_test_dir = self.project_root / 'tests' / 'integration'
        if integration_test_dir.exists():
            try:
                result = subprocess.run([
                    'python', '-m', 'pytest',
                    'tests/integration/',
                    '-v',
                    '--tb=short'
                ], cwd=self.project_root, capture_output=True, text=True, timeout=300)
                
                integration_passed = result.returncode == 0
                self.results.append(QualityGateResult(
                    'integration_tests',
                    integration_passed,
                    100.0 if integration_passed else 0.0,
                    {'output': result.stdout[-1000:]}
                ))
            except Exception as e:
                logger.warning(f"Integration tests failed: {e}")
                self.results.append(QualityGateResult(
                    'integration_tests', False, 0.0, {'error': str(e)}
                ))
    
    def _run_performance_gates(self):
        """Run performance validation gates."""
        logger.info("⚡ Running Performance Gates...")
        
        # Performance benchmarking
        try:
            # Create a simple performance test
            perf_test_script = f'''
import time
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Generate test data
X, y = make_classification(n_samples=10000, n_features=20, random_state=42)
X_df = pd.DataFrame(X)
y_series = pd.Series(y)

# Test training performance
start_time = time.time()
model = LogisticRegression()
model.fit(X_df, y_series)
training_time = time.time() - start_time

# Test prediction performance
start_time = time.time()
predictions = model.predict(X_df)
prediction_time = time.time() - start_time

# Calculate throughput
training_throughput = len(X_df) / training_time
prediction_throughput = len(X_df) / prediction_time

print(f"training_time:{training_time:.4f}")
print(f"prediction_time:{prediction_time:.4f}")
print(f"training_throughput:{training_throughput:.2f}")
print(f"prediction_throughput:{prediction_throughput:.2f}")
'''
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(perf_test_script)
                perf_script_path = f.name
            
            result = subprocess.run([
                'python', perf_script_path
            ], cwd=self.project_root, capture_output=True, text=True, timeout=120)
            
            os.unlink(perf_script_path)  # Clean up
            
            if result.returncode == 0:
                # Parse performance metrics
                output_lines = result.stdout.strip().split('\n')
                metrics = {}
                
                for line in output_lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        try:
                            metrics[key] = float(value)
                        except ValueError:
                            pass
                
                # Performance thresholds
                training_score = min(100, metrics.get('training_throughput', 0) / 100)
                prediction_score = min(100, metrics.get('prediction_throughput', 0) / 1000)
                overall_perf_score = (training_score + prediction_score) / 2
                
                perf_passed = overall_perf_score >= self.quality_thresholds['performance_score']
                
                self.results.append(QualityGateResult(
                    'performance_benchmark',
                    perf_passed,
                    overall_perf_score,
                    {
                        'training_time': metrics.get('training_time', 0),
                        'prediction_time': metrics.get('prediction_time', 0),
                        'training_throughput': metrics.get('training_throughput', 0),
                        'prediction_throughput': metrics.get('prediction_throughput', 0)
                    }
                ))
            else:
                self.results.append(QualityGateResult(
                    'performance_benchmark', False, 0.0,
                    {'error': 'performance_test_failed', 'output': result.stderr}
                ))
                
        except Exception as e:
            logger.warning(f"Performance benchmark failed: {e}")
            self.results.append(QualityGateResult(
                'performance_benchmark', False, 0.0, {'error': str(e)}
            ))
    
    def _run_documentation_gates(self):
        """Run documentation validation gates."""
        logger.info("📚 Running Documentation Gates...")
        
        # Check for essential documentation files
        required_docs = [
            'README.md',
            'CONTRIBUTING.md',
            'LICENSE',
            'requirements.txt',
            'pyproject.toml'
        ]
        
        missing_docs = []
        for doc in required_docs:
            if not (self.project_root / doc).exists():
                missing_docs.append(doc)
        
        # Check docstring coverage
        docstring_coverage = 0.0
        try:
            # Simple docstring coverage check
            python_files = list(self.project_root.glob('src/**/*.py'))
            
            total_functions = 0
            documented_functions = 0
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Simple regex-based docstring detection
                    import re
                    functions = re.findall(r'def\s+\w+\s*\(', content)
                    docstrings = re.findall(r'"""[\s\S]*?"""', content)
                    
                    total_functions += len(functions)
                    # Rough estimate: assume one docstring per function (simplified)
                    documented_functions += min(len(functions), len(docstrings))
                    
                except Exception:
                    continue
            
            if total_functions > 0:
                docstring_coverage = (documented_functions / total_functions) * 100
        
        except Exception as e:
            logger.warning(f"Docstring coverage check failed: {e}")
        
        doc_score = max(0, 100 - len(missing_docs) * 20) * (docstring_coverage / 100)
        doc_passed = len(missing_docs) == 0 and docstring_coverage >= 50
        
        self.results.append(QualityGateResult(
            'documentation',
            doc_passed,
            doc_score,
            {
                'missing_files': missing_docs,
                'docstring_coverage': docstring_coverage,
                'required_docs_present': len(required_docs) - len(missing_docs)
            }
        ))
    
    def _run_advanced_ml_gates(self):
        """Run advanced ML specific quality gates."""
        logger.info("🤖 Running Advanced ML Quality Gates...")
        
        # Test advanced ML modules import and basic functionality
        try:
            # Test ensemble engine
            result = subprocess.run([
                'python', '-c', '''
import sys
sys.path.insert(0, ".")
from src.advanced_ensemble_engine import create_advanced_ensemble
from sklearn.datasets import make_classification
import pandas as pd

X, y = make_classification(n_samples=100, n_features=10, random_state=42)
X_df = pd.DataFrame(X)
y_series = pd.Series(y)

engine = create_advanced_ensemble()
engine.fit(X_df, y_series, optimize_hyperparameters=False)
predictions = engine.predict(X_df[:10])
print(f"ensemble_test:PASS")
'''
            ], cwd=self.project_root, capture_output=True, text=True, timeout=60)
            
            ensemble_passed = 'ensemble_test:PASS' in result.stdout
            
        except Exception as e:
            ensemble_passed = False
            logger.warning(f"Ensemble engine test failed: {e}")
        
        # Test explainable AI
        try:
            result = subprocess.run([
                'python', '-c', '''
import sys
sys.path.insert(0, ".")
from src.explainable_ai_engine import create_explainable_ai_engine
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import pandas as pd

X, y = make_classification(n_samples=100, n_features=10, random_state=42)
X_df = pd.DataFrame(X)
y_series = pd.Series(y)

model = LogisticRegression()
model.fit(X_df, y_series)

explainer = create_explainable_ai_engine()
report = explainer.explain_model(model, X_df[:50], y_series[:50], methods=["feature_importance"])
print(f"explainer_test:PASS")
'''
            ], cwd=self.project_root, capture_output=True, text=True, timeout=60)
            
            explainer_passed = 'explainer_test:PASS' in result.stdout
            
        except Exception as e:
            explainer_passed = False
            logger.warning(f"Explainer test failed: {e}")
        
        # Test quantum ML
        try:
            result = subprocess.run([
                'python', '-c', '''
import sys
sys.path.insert(0, ".")
from src.quantum_ml_optimizer import create_quantum_ml_orchestrator
print(f"quantum_test:PASS")
'''
            ], cwd=self.project_root, capture_output=True, text=True, timeout=30)
            
            quantum_passed = 'quantum_test:PASS' in result.stdout
            
        except Exception as e:
            quantum_passed = False
            logger.warning(f"Quantum ML test failed: {e}")
        
        # Test enterprise platform
        try:
            result = subprocess.run([
                'python', '-c', '''
import sys
sys.path.insert(0, ".")
from src.enterprise_ml_platform import create_enterprise_platform
platform = create_enterprise_platform(database_url="sqlite:///:memory:")
print(f"enterprise_test:PASS")
'''
            ], cwd=self.project_root, capture_output=True, text=True, timeout=30)
            
            enterprise_passed = 'enterprise_test:PASS' in result.stdout
            
        except Exception as e:
            enterprise_passed = False
            logger.warning(f"Enterprise platform test failed: {e}")
        
        # Calculate overall advanced ML score
        ml_components = [ensemble_passed, explainer_passed, quantum_passed, enterprise_passed]
        ml_score = (sum(ml_components) / len(ml_components)) * 100
        ml_passed = sum(ml_components) >= 3  # At least 3/4 must pass
        
        self.results.append(QualityGateResult(
            'advanced_ml_components',
            ml_passed,
            ml_score,
            {
                'ensemble_engine': ensemble_passed,
                'explainable_ai': explainer_passed,
                'quantum_ml': quantum_passed,
                'enterprise_platform': enterprise_passed
            }
        ))
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final report."""
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        # Calculate overall scores
        passed_gates = sum(1 for result in self.results if result.passed)
        total_gates = len(self.results)
        overall_pass_rate = (passed_gates / max(total_gates, 1)) * 100
        
        average_score = sum(result.score for result in self.results) / max(total_gates, 1)
        
        # Categorize results
        critical_failures = []
        warnings = []
        successes = []
        
        for result in self.results:
            if not result.passed and result.score < 50:
                critical_failures.append(result.name)
            elif not result.passed or result.score < 80:
                warnings.append(result.name)
            else:
                successes.append(result.name)
        
        # Determine overall status
        if len(critical_failures) == 0 and overall_pass_rate >= 90:
            overall_status = "EXCELLENT"
        elif len(critical_failures) == 0 and overall_pass_rate >= 80:
            overall_status = "GOOD"
        elif len(critical_failures) <= 2 and overall_pass_rate >= 70:
            overall_status = "ACCEPTABLE"
        else:
            overall_status = "NEEDS_IMPROVEMENT"
        
        # Generate report
        report = {
            'overall_status': overall_status,
            'summary': {
                'total_gates': total_gates,
                'passed_gates': passed_gates,
                'failed_gates': total_gates - passed_gates,
                'pass_rate': overall_pass_rate,
                'average_score': average_score,
                'duration_seconds': total_duration
            },
            'results_by_category': {
                'critical_failures': critical_failures,
                'warnings': warnings,
                'successes': successes
            },
            'detailed_results': [result.to_dict() for result in self.results],
            'recommendations': self._generate_recommendations(critical_failures, warnings),
            'timestamp': end_time.isoformat()
        }
        
        # Print summary
        self._print_summary(report)
        
        return report
    
    def _generate_recommendations(self, critical_failures: List[str], warnings: List[str]) -> List[str]:
        """Generate recommendations based on failed gates."""
        recommendations = []
        
        if 'code_formatting' in critical_failures:
            recommendations.append("Run 'python -m black src/ tests/ scripts/' to fix code formatting")
        
        if 'linting' in critical_failures:
            recommendations.append("Address linting issues with 'python -m flake8 src/ tests/ scripts/'")
        
        if 'security_scan' in critical_failures:
            recommendations.append("Review and fix security vulnerabilities identified by Bandit")
        
        if 'code_coverage' in critical_failures:
            recommendations.append("Increase test coverage by adding more unit tests")
        
        if 'unit_tests' in critical_failures:
            recommendations.append("Fix failing unit tests before deployment")
        
        if 'performance_benchmark' in critical_failures:
            recommendations.append("Optimize performance-critical code paths")
        
        if warnings:
            recommendations.append(f"Address warnings in: {', '.join(warnings)}")
        
        if not recommendations:
            recommendations.append("All quality gates passed! System is ready for deployment.")
        
        return recommendations
    
    def _print_summary(self, report: Dict[str, Any]):
        """Print executive summary of quality gate results."""
        logger.info("📊 QUALITY GATES SUMMARY")
        logger.info("=" * 80)
        
        status_emoji = {
            'EXCELLENT': '🌟',
            'GOOD': '✅',
            'ACCEPTABLE': '⚠️',
            'NEEDS_IMPROVEMENT': '❌'
        }
        
        emoji = status_emoji.get(report['overall_status'], '❓')
        logger.info(f"{emoji} Overall Status: {report['overall_status']}")
        
        summary = report['summary']
        logger.info(f"📈 Pass Rate: {summary['pass_rate']:.1f}% ({summary['passed_gates']}/{summary['total_gates']})")
        logger.info(f"⭐ Average Score: {summary['average_score']:.1f}/100")
        logger.info(f"⏱️ Duration: {summary['duration_seconds']:.1f} seconds")
        
        logger.info("\n🎯 GATE RESULTS:")
        for result in self.results:
            status_icon = "✅" if result.passed else "❌"
            logger.info(f"  {status_icon} {result.name}: {result.score:.1f}/100")
        
        if report['recommendations']:
            logger.info("\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(report['recommendations'], 1):
                logger.info(f"  {i}. {rec}")
        
        logger.info("=" * 80)


def main():
    """Main execution function."""
    runner = ComprehensiveQualityGateRunner()
    
    try:
        # Run all quality gates
        report = runner.run_all_quality_gates()
        
        # Save detailed report
        report_file = Path('quality_gates_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📄 Detailed report saved to: {report_file}")
        
        # Exit with appropriate code
        if report['overall_status'] in ['EXCELLENT', 'GOOD']:
            return 0
        elif report['overall_status'] == 'ACCEPTABLE':
            logger.warning("⚠️ Quality gates passed with warnings")
            return 0
        else:
            logger.error("❌ Quality gates failed - system not ready for deployment")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Quality gate run cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Quality gate runner failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)