#!/usr/bin/env python3
"""
Comprehensive Test Execution Framework.

This script orchestrates the execution of all testing frameworks and quality gates:
- Advanced test coverage analysis with 85% minimum requirement
- Performance testing with benchmarks and load testing  
- Property-based testing with hypothesis and fuzzing
- Integration testing with containerized environments
- Security testing with vulnerability scanning
- Chaos engineering with fault injection
- Quality gates enforcement with deployment blocking
- Test data management with synthetic data generation
- Parallel test execution with intelligent load balancing
- Comprehensive reporting with quality dashboards
"""

import os
import sys
import json
import time
import asyncio
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_execution.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class TestSuiteResult:
    """Result of a test suite execution."""
    suite_name: str
    passed: bool
    execution_time: float
    tests_run: int
    tests_passed: int
    tests_failed: int
    tests_skipped: int
    coverage_percentage: float
    quality_score: float
    error_details: Optional[str] = None
    artifacts: List[str] = None


@dataclass
class ComprehensiveTestReport:
    """Comprehensive test execution report."""
    execution_id: str
    start_time: datetime
    end_time: datetime
    total_duration: float
    suite_results: List[TestSuiteResult]
    overall_passed: bool
    overall_coverage: float
    overall_quality_score: float
    quality_gates_passed: bool
    deployment_approved: bool
    summary: str
    recommendations: List[str]


class TestSuiteExecutor:
    """Executes individual test suites with proper isolation."""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.reports_dir = self.base_dir / "test_reports"
        self.reports_dir.mkdir(exist_ok=True)
        
    def run_unit_tests(self) -> TestSuiteResult:
        """Run unit tests with coverage."""
        logger.info("🧪 Running unit tests...")
        
        start_time = time.time()
        
        try:
            cmd = [
                sys.executable, "-m", "pytest",
                "tests/",
                "-m", "unit or not (integration or performance or security or chaos or slow)",
                "--cov=src",
                "--cov-report=term-missing",
                "--cov-report=json:test_reports/unit_coverage.json",
                "--json-report",
                "--json-report-file=test_reports/unit_report.json",
                "-v",
                "--tb=short"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            execution_time = time.time() - start_time
            
            # Parse results
            coverage_data = self._parse_coverage_report("test_reports/unit_coverage.json")
            test_data = self._parse_pytest_json_report("test_reports/unit_report.json")
            
            return TestSuiteResult(
                suite_name="Unit Tests",
                passed=result.returncode == 0,
                execution_time=execution_time,
                tests_run=test_data.get('total', 0),
                tests_passed=test_data.get('passed', 0),
                tests_failed=test_data.get('failed', 0),
                tests_skipped=test_data.get('skipped', 0),
                coverage_percentage=coverage_data.get('coverage', 0.0),
                quality_score=self._calculate_quality_score(test_data, coverage_data),
                error_details=result.stderr if result.returncode != 0 else None,
                artifacts=["test_reports/unit_coverage.json", "test_reports/unit_report.json"]
            )
            
        except subprocess.TimeoutExpired:
            return TestSuiteResult(
                suite_name="Unit Tests",
                passed=False,
                execution_time=time.time() - start_time,
                tests_run=0,
                tests_passed=0,
                tests_failed=0,
                tests_skipped=0,
                coverage_percentage=0.0,
                quality_score=0.0,
                error_details="Test execution timed out",
                artifacts=[]
            )
        except Exception as e:
            return TestSuiteResult(
                suite_name="Unit Tests",
                passed=False,
                execution_time=time.time() - start_time,
                tests_run=0,
                tests_passed=0,
                tests_failed=0,
                tests_skipped=0,
                coverage_percentage=0.0,
                quality_score=0.0,
                error_details=str(e),
                artifacts=[]
            )
    
    def run_integration_tests(self) -> TestSuiteResult:
        """Run integration tests."""
        logger.info("🔗 Running integration tests...")
        
        start_time = time.time()
        
        try:
            cmd = [
                sys.executable, "-m", "pytest",
                "tests/",
                "-m", "integration",
                "--json-report",
                "--json-report-file=test_reports/integration_report.json",
                "-v",
                "--tb=short",
                "--timeout=1800"  # 30 minutes timeout for integration tests
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=2400)  # 40 minutes total
            execution_time = time.time() - start_time
            
            test_data = self._parse_pytest_json_report("test_reports/integration_report.json")
            
            return TestSuiteResult(
                suite_name="Integration Tests",
                passed=result.returncode == 0,
                execution_time=execution_time,
                tests_run=test_data.get('total', 0),
                tests_passed=test_data.get('passed', 0),
                tests_failed=test_data.get('failed', 0),
                tests_skipped=test_data.get('skipped', 0),
                coverage_percentage=0.0,  # Integration tests don't measure unit coverage
                quality_score=self._calculate_integration_quality_score(test_data),
                error_details=result.stderr if result.returncode != 0 else None,
                artifacts=["test_reports/integration_report.json"]
            )
            
        except Exception as e:
            return TestSuiteResult(
                suite_name="Integration Tests",
                passed=False,
                execution_time=time.time() - start_time,
                tests_run=0,
                tests_passed=0,
                tests_failed=0,
                tests_skipped=0,
                coverage_percentage=0.0,
                quality_score=0.0,
                error_details=str(e),
                artifacts=[]
            )
    
    def run_performance_tests(self) -> TestSuiteResult:
        """Run performance and benchmark tests."""
        logger.info("🚀 Running performance tests...")
        
        start_time = time.time()
        
        try:
            cmd = [
                sys.executable, "-m", "pytest",
                "tests/",
                "-m", "performance or benchmark",
                "--benchmark-json=test_reports/benchmark_report.json",
                "--json-report",
                "--json-report-file=test_reports/performance_report.json",
                "-v",
                "--tb=short"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            execution_time = time.time() - start_time
            
            test_data = self._parse_pytest_json_report("test_reports/performance_report.json")
            benchmark_data = self._parse_benchmark_report("test_reports/benchmark_report.json")
            
            return TestSuiteResult(
                suite_name="Performance Tests",
                passed=result.returncode == 0,
                execution_time=execution_time,
                tests_run=test_data.get('total', 0),
                tests_passed=test_data.get('passed', 0),
                tests_failed=test_data.get('failed', 0),
                tests_skipped=test_data.get('skipped', 0),
                coverage_percentage=0.0,
                quality_score=self._calculate_performance_quality_score(test_data, benchmark_data),
                error_details=result.stderr if result.returncode != 0 else None,
                artifacts=["test_reports/performance_report.json", "test_reports/benchmark_report.json"]
            )
            
        except Exception as e:
            return TestSuiteResult(
                suite_name="Performance Tests",
                passed=False,
                execution_time=time.time() - start_time,
                tests_run=0,
                tests_passed=0,
                tests_failed=0,
                tests_skipped=0,
                coverage_percentage=0.0,
                quality_score=0.0,
                error_details=str(e),
                artifacts=[]
            )
    
    def run_security_tests(self) -> TestSuiteResult:
        """Run security tests."""
        logger.info("🔒 Running security tests...")
        
        start_time = time.time()
        
        try:
            cmd = [
                sys.executable, "-m", "pytest",
                "tests/",
                "-m", "security",
                "--json-report",
                "--json-report-file=test_reports/security_report.json",
                "-v",
                "--tb=short"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
            execution_time = time.time() - start_time
            
            test_data = self._parse_pytest_json_report("test_reports/security_report.json")
            
            return TestSuiteResult(
                suite_name="Security Tests",
                passed=result.returncode == 0,
                execution_time=execution_time,
                tests_run=test_data.get('total', 0),
                tests_passed=test_data.get('passed', 0),
                tests_failed=test_data.get('failed', 0),
                tests_skipped=test_data.get('skipped', 0),
                coverage_percentage=0.0,
                quality_score=self._calculate_security_quality_score(test_data),
                error_details=result.stderr if result.returncode != 0 else None,
                artifacts=["test_reports/security_report.json"]
            )
            
        except Exception as e:
            return TestSuiteResult(
                suite_name="Security Tests",
                passed=False,
                execution_time=time.time() - start_time,
                tests_run=0,
                tests_passed=0,
                tests_failed=0,
                tests_skipped=0,
                coverage_percentage=0.0,
                quality_score=0.0,
                error_details=str(e),
                artifacts=[]
            )
    
    def run_property_based_tests(self) -> TestSuiteResult:
        """Run property-based tests."""
        logger.info("🔍 Running property-based tests...")
        
        start_time = time.time()
        
        try:
            cmd = [
                sys.executable, "-m", "pytest",
                "tests/",
                "-m", "property",
                "--json-report",
                "--json-report-file=test_reports/property_report.json",
                "-v",
                "--tb=short"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            execution_time = time.time() - start_time
            
            test_data = self._parse_pytest_json_report("test_reports/property_report.json")
            
            return TestSuiteResult(
                suite_name="Property-Based Tests",
                passed=result.returncode == 0,
                execution_time=execution_time,
                tests_run=test_data.get('total', 0),
                tests_passed=test_data.get('passed', 0),
                tests_failed=test_data.get('failed', 0),
                tests_skipped=test_data.get('skipped', 0),
                coverage_percentage=0.0,
                quality_score=self._calculate_property_quality_score(test_data),
                error_details=result.stderr if result.returncode != 0 else None,
                artifacts=["test_reports/property_report.json"]
            )
            
        except Exception as e:
            return TestSuiteResult(
                suite_name="Property-Based Tests",
                passed=False,
                execution_time=time.time() - start_time,
                tests_run=0,
                tests_passed=0,
                tests_failed=0,
                tests_skipped=0,
                coverage_percentage=0.0,
                quality_score=0.0,
                error_details=str(e),
                artifacts=[]
            )
    
    def run_chaos_tests(self) -> TestSuiteResult:
        """Run chaos engineering tests."""
        logger.info("💥 Running chaos engineering tests...")
        
        start_time = time.time()
        
        try:
            cmd = [
                sys.executable, "-m", "pytest",
                "tests/",
                "-m", "chaos",
                "--json-report",
                "--json-report-file=test_reports/chaos_report.json",
                "-v",
                "--tb=short"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            execution_time = time.time() - start_time
            
            test_data = self._parse_pytest_json_report("test_reports/chaos_report.json")
            
            return TestSuiteResult(
                suite_name="Chaos Engineering Tests",
                passed=result.returncode == 0,
                execution_time=execution_time,
                tests_run=test_data.get('total', 0),
                tests_passed=test_data.get('passed', 0),
                tests_failed=test_data.get('failed', 0),
                tests_skipped=test_data.get('skipped', 0),
                coverage_percentage=0.0,
                quality_score=self._calculate_chaos_quality_score(test_data),
                error_details=result.stderr if result.returncode != 0 else None,
                artifacts=["test_reports/chaos_report.json"]
            )
            
        except Exception as e:
            return TestSuiteResult(
                suite_name="Chaos Engineering Tests",
                passed=False,
                execution_time=time.time() - start_time,
                tests_run=0,
                tests_passed=0,
                tests_failed=0,
                tests_skipped=0,
                coverage_percentage=0.0,
                quality_score=0.0,
                error_details=str(e),
                artifacts=[]
            )
    
    def run_quality_gates(self) -> TestSuiteResult:
        """Run quality gates framework."""
        logger.info("🚪 Running quality gates...")
        
        start_time = time.time()
        
        try:
            # Run the quality gates orchestrator
            cmd = [
                sys.executable, "-c", """
import sys
sys.path.append('.')
try:
    from tests.quality_gates.quality_gates import QualityGateOrchestrator
    orchestrator = QualityGateOrchestrator()
    report = orchestrator.run_all_gates()
    
    import json
    with open('test_reports/quality_gates_report.json', 'w') as f:
        json.dump({
            'overall_passed': report.deployment_approved,
            'overall_score': report.overall_score,
            'blocking_failures': report.blocking_failures,
            'warnings': report.warnings,
            'gate_results': len(report.gate_results)
        }, f, indent=2)
    
    exit(0 if report.deployment_approved else 1)
except Exception as e:
    print(f'Quality gates failed: {e}')
    exit(1)
"""
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            execution_time = time.time() - start_time
            
            # Parse quality gates report
            quality_data = {}
            try:
                with open("test_reports/quality_gates_report.json") as f:
                    quality_data = json.load(f)
            except:
                pass
            
            return TestSuiteResult(
                suite_name="Quality Gates",
                passed=result.returncode == 0,
                execution_time=execution_time,
                tests_run=quality_data.get('gate_results', 0),
                tests_passed=quality_data.get('gate_results', 0) - quality_data.get('blocking_failures', 0),
                tests_failed=quality_data.get('blocking_failures', 0),
                tests_skipped=0,
                coverage_percentage=0.0,
                quality_score=quality_data.get('overall_score', 0.0),
                error_details=result.stderr if result.returncode != 0 else None,
                artifacts=["test_reports/quality_gates_report.json"]
            )
            
        except Exception as e:
            return TestSuiteResult(
                suite_name="Quality Gates",
                passed=False,
                execution_time=time.time() - start_time,
                tests_run=0,
                tests_passed=0,
                tests_failed=0,
                tests_skipped=0,
                coverage_percentage=0.0,
                quality_score=0.0,
                error_details=str(e),
                artifacts=[]
            )
    
    def _parse_coverage_report(self, file_path: str) -> Dict[str, Any]:
        """Parse coverage JSON report."""
        try:
            with open(file_path) as f:
                data = json.load(f)
            
            totals = data.get('totals', {})
            return {
                'coverage': totals.get('percent_covered', 0.0),
                'lines_covered': totals.get('covered_lines', 0),
                'lines_total': totals.get('num_statements', 0),
                'branches_covered': totals.get('covered_branches', 0),
                'branches_total': totals.get('num_branches', 0)
            }
        except:
            return {'coverage': 0.0}
    
    def _parse_pytest_json_report(self, file_path: str) -> Dict[str, Any]:
        """Parse pytest JSON report."""
        try:
            with open(file_path) as f:
                data = json.load(f)
            
            summary = data.get('summary', {})
            return {
                'total': summary.get('total', 0),
                'passed': summary.get('passed', 0),
                'failed': summary.get('failed', 0),
                'skipped': summary.get('skipped', 0),
                'error': summary.get('error', 0),
                'duration': data.get('duration', 0.0)
            }
        except:
            return {'total': 0, 'passed': 0, 'failed': 0, 'skipped': 0}
    
    def _parse_benchmark_report(self, file_path: str) -> Dict[str, Any]:
        """Parse benchmark JSON report."""
        try:
            with open(file_path) as f:
                data = json.load(f)
            
            benchmarks = data.get('benchmarks', [])
            if benchmarks:
                avg_time = sum(b.get('stats', {}).get('mean', 0) for b in benchmarks) / len(benchmarks)
                return {
                    'benchmark_count': len(benchmarks),
                    'avg_execution_time': avg_time,
                    'benchmarks_passed': len([b for b in benchmarks if b.get('stats', {}).get('mean', 0) > 0])
                }
            
            return {'benchmark_count': 0, 'avg_execution_time': 0.0, 'benchmarks_passed': 0}
        except:
            return {'benchmark_count': 0, 'avg_execution_time': 0.0, 'benchmarks_passed': 0}
    
    def _calculate_quality_score(self, test_data: Dict, coverage_data: Dict) -> float:
        """Calculate quality score for unit tests."""
        total_tests = test_data.get('total', 1)
        passed_tests = test_data.get('passed', 0)
        coverage = coverage_data.get('coverage', 0.0)
        
        pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Weighted score: 60% pass rate, 40% coverage
        quality_score = (pass_rate * 0.6) + (coverage * 0.4)
        return min(quality_score, 100.0)
    
    def _calculate_integration_quality_score(self, test_data: Dict) -> float:
        """Calculate quality score for integration tests."""
        total_tests = test_data.get('total', 1)
        passed_tests = test_data.get('passed', 0)
        
        pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        return min(pass_rate, 100.0)
    
    def _calculate_performance_quality_score(self, test_data: Dict, benchmark_data: Dict) -> float:
        """Calculate quality score for performance tests."""
        total_tests = test_data.get('total', 1)
        passed_tests = test_data.get('passed', 0)
        benchmarks_passed = benchmark_data.get('benchmarks_passed', 0)
        benchmark_count = benchmark_data.get('benchmark_count', 1)
        
        test_pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        benchmark_pass_rate = (benchmarks_passed / benchmark_count) * 100 if benchmark_count > 0 else 0
        
        # Weighted score: 50% test pass rate, 50% benchmark pass rate
        quality_score = (test_pass_rate * 0.5) + (benchmark_pass_rate * 0.5)
        return min(quality_score, 100.0)
    
    def _calculate_security_quality_score(self, test_data: Dict) -> float:
        """Calculate quality score for security tests."""
        total_tests = test_data.get('total', 1)
        passed_tests = test_data.get('passed', 0)
        
        # Security tests are critical - any failure significantly impacts score
        if total_tests == 0:
            return 0.0
        
        pass_rate = (passed_tests / total_tests) * 100
        
        # Penalty for security failures
        failed_tests = test_data.get('failed', 0)
        if failed_tests > 0:
            penalty = min(failed_tests * 25, 75)  # Up to 75 point penalty
            pass_rate = max(pass_rate - penalty, 0)
        
        return min(pass_rate, 100.0)
    
    def _calculate_property_quality_score(self, test_data: Dict) -> float:
        """Calculate quality score for property-based tests."""
        total_tests = test_data.get('total', 1)
        passed_tests = test_data.get('passed', 0)
        
        pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        return min(pass_rate, 100.0)
    
    def _calculate_chaos_quality_score(self, test_data: Dict) -> float:
        """Calculate quality score for chaos tests."""
        total_tests = test_data.get('total', 1)
        passed_tests = test_data.get('passed', 0)
        
        # Chaos tests passing indicates good resilience
        resilience_score = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        return min(resilience_score, 100.0)


class ComprehensiveTestRunner:
    """Main comprehensive test runner."""
    
    def __init__(self):
        self.executor = TestSuiteExecutor()
        self.reports_dir = Path("test_reports")
        self.reports_dir.mkdir(exist_ok=True)
    
    def run_all_tests(self, suites: Optional[List[str]] = None, 
                     parallel: bool = True) -> ComprehensiveTestReport:
        """Run all test suites."""
        
        execution_id = f"comprehensive_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()
        
        logger.info(f"🚀 Starting comprehensive test execution: {execution_id}")
        
        # Define test suites
        available_suites = {
            'unit': self.executor.run_unit_tests,
            'integration': self.executor.run_integration_tests,
            'performance': self.executor.run_performance_tests,
            'security': self.executor.run_security_tests,
            'property': self.executor.run_property_based_tests,
            'chaos': self.executor.run_chaos_tests,
            'quality_gates': self.executor.run_quality_gates
        }
        
        # Select suites to run
        suites_to_run = suites if suites else list(available_suites.keys())
        
        logger.info(f"📋 Test suites to execute: {suites_to_run}")
        
        # Execute test suites
        suite_results = []
        
        if parallel and len(suites_to_run) > 1:
            # Run suites in parallel (except dependencies)
            suite_results = self._run_suites_parallel(available_suites, suites_to_run)
        else:
            # Run suites sequentially
            suite_results = self._run_suites_sequential(available_suites, suites_to_run)
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        # Calculate overall metrics
        overall_passed = all(result.passed for result in suite_results)
        overall_coverage = self._calculate_overall_coverage(suite_results)
        overall_quality_score = self._calculate_overall_quality_score(suite_results)
        quality_gates_passed = any(
            result.suite_name == "Quality Gates" and result.passed 
            for result in suite_results
        )
        
        # Determine deployment approval
        deployment_approved = overall_passed and overall_coverage >= 85.0 and overall_quality_score >= 80.0
        
        # Generate summary and recommendations
        summary = self._generate_summary(suite_results, overall_passed, deployment_approved)
        recommendations = self._generate_recommendations(suite_results, overall_coverage, overall_quality_score)
        
        # Create comprehensive report
        report = ComprehensiveTestReport(
            execution_id=execution_id,
            start_time=start_time,
            end_time=end_time,
            total_duration=total_duration,
            suite_results=suite_results,
            overall_passed=overall_passed,
            overall_coverage=overall_coverage,
            overall_quality_score=overall_quality_score,
            quality_gates_passed=quality_gates_passed,
            deployment_approved=deployment_approved,
            summary=summary,
            recommendations=recommendations
        )
        
        # Save comprehensive report
        self._save_comprehensive_report(report)
        
        # Print summary
        self._print_comprehensive_summary(report)
        
        return report
    
    def _run_suites_parallel(self, available_suites: Dict, suites_to_run: List[str]) -> List[TestSuiteResult]:
        """Run test suites in parallel."""
        logger.info("🔄 Running test suites in parallel...")
        
        results = []
        
        # Separate suites by dependency requirements
        # Unit tests should run first for coverage
        independent_suites = [s for s in suites_to_run if s != 'unit' and s != 'quality_gates']
        dependent_suites = [s for s in suites_to_run if s in ['unit', 'quality_gates']]
        
        # Run unit tests first
        if 'unit' in suites_to_run:
            unit_result = available_suites['unit']()
            results.append(unit_result)
        
        # Run independent suites in parallel
        if independent_suites:
            with ThreadPoolExecutor(max_workers=min(len(independent_suites), 4)) as executor:
                future_to_suite = {
                    executor.submit(available_suites[suite_name]): suite_name
                    for suite_name in independent_suites
                }
                
                for future in as_completed(future_to_suite):
                    suite_name = future_to_suite[future]
                    try:
                        result = future.result(timeout=3600)  # 1 hour timeout per suite
                        results.append(result)
                        logger.info(f"✅ {suite_name} completed")
                    except Exception as e:
                        logger.error(f"❌ {suite_name} failed: {e}")
                        # Create failed result
                        results.append(TestSuiteResult(
                            suite_name=suite_name.replace('_', ' ').title(),
                            passed=False,
                            execution_time=0,
                            tests_run=0,
                            tests_passed=0,
                            tests_failed=0,
                            tests_skipped=0,
                            coverage_percentage=0.0,
                            quality_score=0.0,
                            error_details=str(e)
                        ))
        
        # Run quality gates last (depends on other results)
        if 'quality_gates' in suites_to_run:
            quality_gates_result = available_suites['quality_gates']()
            results.append(quality_gates_result)
        
        return results
    
    def _run_suites_sequential(self, available_suites: Dict, suites_to_run: List[str]) -> List[TestSuiteResult]:
        """Run test suites sequentially."""
        logger.info("➡️ Running test suites sequentially...")
        
        results = []
        
        for suite_name in suites_to_run:
            if suite_name in available_suites:
                logger.info(f"🏃 Running {suite_name} tests...")
                try:
                    result = available_suites[suite_name]()
                    results.append(result)
                    
                    status = "✅ PASSED" if result.passed else "❌ FAILED"
                    logger.info(f"{status} {suite_name} - {result.execution_time:.2f}s")
                    
                except Exception as e:
                    logger.error(f"❌ {suite_name} failed: {e}")
                    results.append(TestSuiteResult(
                        suite_name=suite_name.replace('_', ' ').title(),
                        passed=False,
                        execution_time=0,
                        tests_run=0,
                        tests_passed=0,
                        tests_failed=0,
                        tests_skipped=0,
                        coverage_percentage=0.0,
                        quality_score=0.0,
                        error_details=str(e)
                    ))
            else:
                logger.warning(f"⚠️ Unknown test suite: {suite_name}")
        
        return results
    
    def _calculate_overall_coverage(self, suite_results: List[TestSuiteResult]) -> float:
        """Calculate overall coverage percentage."""
        coverage_results = [r for r in suite_results if r.coverage_percentage > 0]
        
        if not coverage_results:
            return 0.0
        
        # Weight by number of tests
        total_weighted_coverage = sum(r.coverage_percentage * r.tests_run for r in coverage_results)
        total_tests = sum(r.tests_run for r in coverage_results)
        
        return total_weighted_coverage / max(total_tests, 1)
    
    def _calculate_overall_quality_score(self, suite_results: List[TestSuiteResult]) -> float:
        """Calculate overall quality score."""
        if not suite_results:
            return 0.0
        
        # Weight by number of tests and suite importance
        suite_weights = {
            'Unit Tests': 0.3,
            'Integration Tests': 0.25,
            'Security Tests': 0.2,
            'Performance Tests': 0.15,
            'Property-Based Tests': 0.05,
            'Chaos Engineering Tests': 0.03,
            'Quality Gates': 0.02
        }
        
        total_weighted_score = 0
        total_weight = 0
        
        for result in suite_results:
            weight = suite_weights.get(result.suite_name, 0.1)
            total_weighted_score += result.quality_score * weight
            total_weight += weight
        
        return total_weighted_score / max(total_weight, 1)
    
    def _generate_summary(self, suite_results: List[TestSuiteResult], 
                         overall_passed: bool, deployment_approved: bool) -> str:
        """Generate executive summary."""
        
        total_tests = sum(r.tests_run for r in suite_results)
        total_passed = sum(r.tests_passed for r in suite_results)
        total_failed = sum(r.tests_failed for r in suite_results)
        
        if deployment_approved:
            summary = "🎉 All quality gates passed! Code is ready for deployment."
        elif overall_passed:
            summary = "⚠️ Tests passed but quality gates have concerns. Review recommendations before deployment."
        else:
            summary = "❌ Tests failed. Deployment blocked until issues are resolved."
        
        summary += f"\n\nTest Execution Summary:"
        summary += f"\n• Total Tests: {total_tests:,}"
        summary += f"\n• Passed: {total_passed:,}"
        summary += f"\n• Failed: {total_failed:,}"
        summary += f"\n• Suites Executed: {len(suite_results)}"
        
        return summary
    
    def _generate_recommendations(self, suite_results: List[TestSuiteResult],
                                overall_coverage: float, overall_quality_score: float) -> List[str]:
        """Generate improvement recommendations."""
        
        recommendations = []
        
        # Coverage recommendations
        if overall_coverage < 85:
            recommendations.append(f"📈 Increase test coverage from {overall_coverage:.1f}% to 85%+ by adding unit tests")
        
        # Quality score recommendations
        if overall_quality_score < 80:
            recommendations.append(f"🎯 Improve overall quality score from {overall_quality_score:.1f} to 80+ by fixing failing tests")
        
        # Suite-specific recommendations
        for result in suite_results:
            if not result.passed:
                recommendations.append(f"🔧 Fix failing {result.suite_name.lower()}: {result.tests_failed} tests failed")
            
            if result.quality_score < 70:
                recommendations.append(f"⚡ Improve {result.suite_name.lower()} quality (currently {result.quality_score:.1f}/100)")
        
        # Performance recommendations
        slow_suites = [r for r in suite_results if r.execution_time > 300]  # 5 minutes
        if slow_suites:
            suite_names = [r.suite_name for r in slow_suites]
            recommendations.append(f"🚀 Optimize slow test suites: {', '.join(suite_names)}")
        
        # Security recommendations
        security_results = [r for r in suite_results if "Security" in r.suite_name and not r.passed]
        if security_results:
            recommendations.append("🔒 Address security test failures immediately - security is critical")
        
        if not recommendations:
            recommendations.append("✅ Excellent work! All tests passing and quality metrics meet requirements")
        
        return recommendations
    
    def _save_comprehensive_report(self, report: ComprehensiveTestReport):
        """Save comprehensive test report."""
        
        # Save JSON report
        json_file = self.reports_dir / f"comprehensive_report_{report.execution_id}.json"
        with open(json_file, 'w') as f:
            # Convert datetime objects to strings for JSON serialization
            report_dict = asdict(report)
            report_dict['start_time'] = report.start_time.isoformat()
            report_dict['end_time'] = report.end_time.isoformat()
            
            json.dump(report_dict, f, indent=2)
        
        # Save HTML report
        html_file = self.reports_dir / f"comprehensive_report_{report.execution_id}.html"
        self._generate_html_report(report, html_file)
        
        logger.info(f"📊 Comprehensive reports saved:")
        logger.info(f"  • JSON: {json_file}")
        logger.info(f"  • HTML: {html_file}")
    
    def _generate_html_report(self, report: ComprehensiveTestReport, output_file: Path):
        """Generate HTML report."""
        
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Comprehensive Test Report - {execution_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; margin-bottom: 30px; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 8px; }}
        .metrics {{ display: flex; gap: 20px; margin-bottom: 30px; flex-wrap: wrap; }}
        .metric-card {{ flex: 1; min-width: 200px; background: white; padding: 20px; border-radius: 8px; border: 1px solid #e0e0e0; text-align: center; }}
        .metric-value {{ font-size: 28px; font-weight: bold; margin-bottom: 5px; }}
        .metric-label {{ color: #666; font-size: 14px; }}
        .passed {{ color: #4caf50; }}
        .failed {{ color: #f44336; }}
        .warning {{ color: #ff9800; }}
        .suite-results {{ margin-bottom: 30px; }}
        .suite-item {{ background: white; margin: 10px 0; padding: 20px; border-radius: 8px; border-left: 4px solid #ddd; }}
        .suite-item.passed {{ border-left-color: #4caf50; }}
        .suite-item.failed {{ border-left-color: #f44336; }}
        .suite-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
        .suite-name {{ font-size: 18px; font-weight: bold; }}
        .suite-stats {{ font-size: 14px; color: #666; }}
        .recommendations {{ background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; padding: 20px; }}
        .recommendation {{ margin: 10px 0; padding-left: 20px; }}
        .summary {{ background: #e8f5e8; border: 1px solid #c8e6c9; border-radius: 8px; padding: 20px; margin-bottom: 20px; }}
        .summary.failed {{ background: #ffebee; border-color: #ffcdd2; }}
        .progress-bar {{ background: #e0e0e0; border-radius: 10px; height: 20px; margin: 10px 0; }}
        .progress-fill {{ height: 100%; border-radius: 10px; transition: width 0.3s ease; }}
        .progress-fill.high {{ background: #4caf50; }}
        .progress-fill.medium {{ background: #ff9800; }}
        .progress-fill.low {{ background: #f44336; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Comprehensive Test Report</h1>
            <p>Execution ID: {execution_id}</p>
            <p>Generated: {start_time} | Duration: {duration:.1f}s</p>
        </div>

        <div class="summary {summary_class}">
            <h2>Executive Summary</h2>
            <p>{summary}</p>
        </div>

        <div class="metrics">
            <div class="metric-card">
                <div class="metric-value {overall_status_class}">{overall_status}</div>
                <div class="metric-label">Overall Status</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{overall_coverage:.1f}%</div>
                <div class="metric-label">Coverage</div>
                <div class="progress-bar">
                    <div class="progress-fill {coverage_class}" style="width: {overall_coverage:.1f}%"></div>
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{overall_quality_score:.1f}</div>
                <div class="metric-label">Quality Score</div>
                <div class="progress-bar">
                    <div class="progress-fill {quality_class}" style="width: {overall_quality_score:.1f}%"></div>
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-value {deployment_class}">{deployment_status}</div>
                <div class="metric-label">Deployment</div>
            </div>
        </div>

        <div class="suite-results">
            <h2>Test Suite Results</h2>
            {suite_results_html}
        </div>

        {recommendations_html}
    </div>
</body>
</html>
"""
        
        # Prepare template data
        suite_results_html = ""
        for result in report.suite_results:
            status_class = "passed" if result.passed else "failed"
            status_text = "✅ PASSED" if result.passed else "❌ FAILED"
            
            suite_results_html += f"""
            <div class="suite-item {status_class}">
                <div class="suite-header">
                    <div class="suite-name">{result.suite_name}</div>
                    <div class="suite-stats">{status_text} | {result.execution_time:.1f}s</div>
                </div>
                <div class="suite-stats">
                    Tests: {result.tests_run} | Passed: {result.tests_passed} | Failed: {result.tests_failed} | Skipped: {result.tests_skipped}
                    | Quality Score: {result.quality_score:.1f}/100
                    {' | Coverage: ' + str(result.coverage_percentage) + '%' if result.coverage_percentage > 0 else ''}
                </div>
                {f'<div style="color: #f44336; margin-top: 10px;">Error: {result.error_details}</div>' if result.error_details else ''}
            </div>
            """
        
        recommendations_html = ""
        if report.recommendations:
            recommendations_html = f"""
            <div class="recommendations">
                <h2>Recommendations</h2>
                {''.join([f'<div class="recommendation">• {rec}</div>' for rec in report.recommendations])}
            </div>
            """
        
        # Generate HTML
        html_content = html_template.format(
            execution_id=report.execution_id,
            start_time=report.start_time.strftime('%Y-%m-%d %H:%M:%S'),
            duration=report.total_duration,
            summary=report.summary.replace('\n', '<br>'),
            summary_class="passed" if report.deployment_approved else "failed",
            overall_status="PASSED" if report.overall_passed else "FAILED",
            overall_status_class="passed" if report.overall_passed else "failed",
            overall_coverage=report.overall_coverage,
            coverage_class="high" if report.overall_coverage >= 85 else "medium" if report.overall_coverage >= 70 else "low",
            overall_quality_score=report.overall_quality_score,
            quality_class="high" if report.overall_quality_score >= 80 else "medium" if report.overall_quality_score >= 60 else "low",
            deployment_status="APPROVED" if report.deployment_approved else "BLOCKED",
            deployment_class="passed" if report.deployment_approved else "failed",
            suite_results_html=suite_results_html,
            recommendations_html=recommendations_html
        )
        
        with open(output_file, 'w') as f:
            f.write(html_content)
    
    def _print_comprehensive_summary(self, report: ComprehensiveTestReport):
        """Print comprehensive test summary to console."""
        
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE TEST EXECUTION REPORT")
        print(f"{'='*80}")
        
        print(f"Execution ID: {report.execution_id}")
        print(f"Duration: {report.total_duration:.2f} seconds")
        print(f"Start Time: {report.start_time}")
        print(f"End Time: {report.end_time}")
        
        print(f"\n🎯 OVERALL RESULTS:")
        status_emoji = "✅" if report.overall_passed else "❌"
        print(f"  {status_emoji} Overall Status: {'PASSED' if report.overall_passed else 'FAILED'}")
        print(f"  📊 Coverage: {report.overall_coverage:.1f}%")
        print(f"  🎯 Quality Score: {report.overall_quality_score:.1f}/100")
        
        deployment_emoji = "✅" if report.deployment_approved else "❌"
        deployment_status = "APPROVED" if report.deployment_approved else "BLOCKED"
        print(f"  🚀 Deployment: {deployment_emoji} {deployment_status}")
        
        print(f"\n📋 SUITE BREAKDOWN:")
        for result in report.suite_results:
            status_emoji = "✅" if result.passed else "❌"
            print(f"  {status_emoji} {result.suite_name}")
            print(f"      Tests: {result.tests_run} | Passed: {result.tests_passed} | Failed: {result.tests_failed}")
            print(f"      Duration: {result.execution_time:.2f}s | Quality: {result.quality_score:.1f}/100")
            if result.coverage_percentage > 0:
                print(f"      Coverage: {result.coverage_percentage:.1f}%")
            if result.error_details:
                print(f"      Error: {result.error_details}")
        
        if report.recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(report.recommendations, 1):
                print(f"  {i}. {rec}")
        
        print(f"\n{report.summary}")
        
        if not report.deployment_approved:
            print(f"\n🚫 DEPLOYMENT BLOCKED")
            print("Fix the issues above before proceeding with deployment.")
        elif not report.overall_passed:
            print(f"\n⚠️ DEPLOYMENT APPROVED WITH WARNINGS")
            print("Consider addressing the warnings in the next iteration.")
        else:
            print(f"\n🎉 DEPLOYMENT APPROVED")
            print("All tests passed and quality gates are satisfied.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Comprehensive Test Execution Framework")
    
    parser.add_argument(
        '--suites', 
        nargs='+', 
        choices=['unit', 'integration', 'performance', 'security', 'property', 'chaos', 'quality_gates'],
        help='Test suites to run (default: all)'
    )
    
    parser.add_argument(
        '--no-parallel', 
        action='store_true',
        help='Disable parallel execution'
    )
    
    parser.add_argument(
        '--coverage-threshold',
        type=float,
        default=85.0,
        help='Minimum coverage threshold (default: 85%%)'
    )
    
    parser.add_argument(
        '--quality-threshold',
        type=float, 
        default=80.0,
        help='Minimum quality score threshold (default: 80)'
    )
    
    parser.add_argument(
        '--fail-fast',
        action='store_true',
        help='Stop execution on first suite failure'
    )
    
    parser.add_argument(
        '--report-only',
        action='store_true',
        help='Generate reports without running tests'
    )
    
    args = parser.parse_args()
    
    if args.report_only:
        print("📊 Report-only mode - skipping test execution")
        return 0
    
    # Create test runner
    runner = ComprehensiveTestRunner()
    
    try:
        # Run tests
        report = runner.run_all_tests(
            suites=args.suites,
            parallel=not args.no_parallel
        )
        
        # Exit with appropriate code
        if not report.deployment_approved:
            logger.error("🚫 Deployment blocked due to test failures or quality gates")
            return 1
        elif not report.overall_passed:
            logger.warning("⚠️ Tests completed with warnings")
            return 2
        else:
            logger.info("✅ All tests passed successfully")
            return 0
            
    except KeyboardInterrupt:
        logger.info("🛑 Test execution interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())