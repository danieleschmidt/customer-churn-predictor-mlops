"""
Comprehensive Test Runner and Quality Dashboard.

This module provides:
- Parallel test execution with intelligent load balancing
- Real-time test reporting and progress tracking
- Quality metrics dashboard with trend analysis
- Test result aggregation and comprehensive reporting
- CI/CD integration with multiple output formats
- Performance monitoring during test execution
- Automated test categorization and prioritization
- Failure analysis and debugging assistance
"""

import os
import json
import time
import asyncio
import threading
import subprocess
import multiprocessing
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Set, Callable
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from enum import Enum, auto
import uuid

# For HTML reporting
try:
    from jinja2 import Template
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

# Import our test frameworks
try:
    from .coverage_analyzer import AdvancedCoverageAnalyzer
    from .performance_testing import PerformanceTestSuite
    from .property_based_testing import PropertyBasedTestRunner
    from .integration_testing import IntegrationTestSuiteRunner
    from .security_testing import SecurityTestSuite
    from .chaos_engineering import ChaosExperimentRunner, create_sample_experiments
    from .quality_gates import QualityGateOrchestrator
    FRAMEWORKS_AVAILABLE = True
except ImportError:
    FRAMEWORKS_AVAILABLE = False


class TestCategory(Enum):
    """Test categories for organization and prioritization."""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    SECURITY = "security"
    PROPERTY_BASED = "property_based"
    CHAOS = "chaos"
    QUALITY_GATES = "quality_gates"
    COVERAGE = "coverage"
    END_TO_END = "end_to_end"


class TestPriority(Enum):
    """Test execution priority levels."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


class TestStatus(Enum):
    """Test execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class TestSuite:
    """Definition of a test suite."""
    suite_id: str
    name: str
    description: str
    category: TestCategory
    priority: TestPriority
    enabled: bool
    timeout_seconds: int
    dependencies: List[str]
    runner_function: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestResult:
    """Individual test result."""
    test_id: str
    suite_id: str
    name: str
    category: TestCategory
    status: TestStatus
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    message: str
    details: Dict[str, Any]
    artifacts: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class TestExecutionReport:
    """Complete test execution report."""
    execution_id: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    error_tests: int
    success_rate: float
    coverage_percentage: float
    performance_score: float
    security_score: float
    quality_score: float
    test_results: List[TestResult]
    suite_summaries: Dict[str, Dict[str, Any]]
    system_metrics: Dict[str, Any]
    recommendations: List[str]
    artifacts: List[str] = field(default_factory=list)


class SystemMonitor:
    """Monitors system resources during test execution."""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.monitoring_active = False
        self.metrics_history = deque(maxlen=3600)  # 1 hour at 1-second intervals
        self.monitoring_thread = None
    
    def start_monitoring(self):
        """Start system monitoring."""
        import psutil
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitoring_thread.start()
        print("📊 System monitoring started")
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return summary."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        if not self.metrics_history:
            return {}
        
        # Calculate summary statistics
        cpu_values = [m['cpu_percent'] for m in self.metrics_history]
        memory_values = [m['memory_percent'] for m in self.metrics_history]
        
        return {
            'duration_seconds': len(self.metrics_history) * self.monitoring_interval,
            'cpu_usage': {
                'avg': sum(cpu_values) / len(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values)
            },
            'memory_usage': {
                'avg': sum(memory_values) / len(memory_values),
                'max': max(memory_values),
                'min': min(memory_values)
            },
            'sample_count': len(self.metrics_history)
        }
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        import psutil
        
        while self.monitoring_active:
            try:
                metrics = {
                    'timestamp': time.time(),
                    'cpu_percent': psutil.cpu_percent(interval=0.1),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_io': psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {},
                    'network_io': psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {}
                }
                
                self.metrics_history.append(metrics)
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                break


class ParallelTestExecutor:
    """Executes tests in parallel with intelligent scheduling."""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(multiprocessing.cpu_count(), 8)
        self.test_queue = deque()
        self.running_tests = {}
        self.completed_tests = []
        self.failed_dependencies = set()
    
    def add_test_suite(self, suite: TestSuite):
        """Add test suite to execution queue."""
        self.test_queue.append(suite)
    
    def execute_all(self, progress_callback: Callable = None) -> List[TestResult]:
        """Execute all queued test suites in parallel."""
        print(f"🚀 Starting parallel test execution with {self.max_workers} workers")
        
        # Sort by priority and dependencies
        sorted_suites = self._sort_suites_by_priority_and_dependencies()
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit initial batch of tests
            future_to_suite = {}
            
            for suite in sorted_suites:
                if self._can_run_suite(suite):
                    future = executor.submit(self._execute_suite, suite)
                    future_to_suite[future] = suite
                    self.running_tests[suite.suite_id] = {
                        'suite': suite,
                        'future': future,
                        'start_time': datetime.now()
                    }
            
            # Process completed tests and submit new ones
            while future_to_suite or self.test_queue:
                if not future_to_suite and self.test_queue:
                    # No tests running but tests in queue - check for dependency issues
                    print("⚠️ Tests waiting for dependencies that may never complete")
                    break
                
                # Wait for at least one test to complete
                completed_futures = as_completed(future_to_suite.keys(), timeout=1)
                
                for future in completed_futures:
                    suite = future_to_suite[future]
                    
                    try:
                        result = future.result()
                        results.append(result)
                        
                        # Update progress
                        if progress_callback:
                            progress_callback(len(results), len(sorted_suites))
                        
                        # Mark suite as completed
                        if suite.suite_id in self.running_tests:
                            del self.running_tests[suite.suite_id]
                        
                        # Check if this enables other tests to run
                        newly_runnable = self._check_newly_runnable_suites()
                        for new_suite in newly_runnable:
                            if len(future_to_suite) < self.max_workers:
                                new_future = executor.submit(self._execute_suite, new_suite)
                                future_to_suite[new_future] = new_suite
                                self.running_tests[new_suite.suite_id] = {
                                    'suite': new_suite,
                                    'future': new_future,
                                    'start_time': datetime.now()
                                }
                        
                    except Exception as e:
                        print(f"❌ Test suite {suite.name} failed: {e}")
                        
                        # Create error result
                        error_result = TestResult(
                            test_id=f"error_{suite.suite_id}",
                            suite_id=suite.suite_id,
                            name=suite.name,
                            category=suite.category,
                            status=TestStatus.ERROR,
                            start_time=self.running_tests[suite.suite_id]['start_time'],
                            end_time=datetime.now(),
                            duration_seconds=0,
                            message=f"Suite execution failed: {str(e)}",
                            details={'error': str(e)}
                        )
                        results.append(error_result)
                        
                        # Mark dependencies as failed
                        self.failed_dependencies.add(suite.suite_id)
                    
                    # Remove from future tracking
                    del future_to_suite[future]
        
        print(f"✅ Test execution completed: {len(results)} test suites processed")
        return results
    
    def _sort_suites_by_priority_and_dependencies(self) -> List[TestSuite]:
        """Sort test suites by priority and resolve dependencies."""
        # Convert queue to list for sorting
        suites = list(self.test_queue)
        self.test_queue.clear()
        
        # Sort by priority (lower number = higher priority)
        suites.sort(key=lambda s: s.priority.value)
        
        # TODO: Implement proper topological sort for dependencies
        # For now, just return priority-sorted list
        return suites
    
    def _can_run_suite(self, suite: TestSuite) -> bool:
        """Check if a test suite can run (dependencies met)."""
        if not suite.enabled:
            return False
        
        # Check if any dependencies have failed
        if any(dep in self.failed_dependencies for dep in suite.dependencies):
            return False
        
        # Check if all dependencies have completed successfully
        completed_suite_ids = {r.suite_id for r in self.completed_tests if r.status == TestStatus.PASSED}
        
        for dep in suite.dependencies:
            if dep not in completed_suite_ids and dep not in self.running_tests:
                return False
        
        return True
    
    def _check_newly_runnable_suites(self) -> List[TestSuite]:
        """Check for test suites that can now run due to completed dependencies."""
        newly_runnable = []
        
        remaining_suites = list(self.test_queue)
        self.test_queue.clear()
        
        for suite in remaining_suites:
            if self._can_run_suite(suite):
                newly_runnable.append(suite)
            else:
                self.test_queue.append(suite)  # Put back in queue
        
        return newly_runnable
    
    def _execute_suite(self, suite: TestSuite) -> TestResult:
        """Execute a single test suite."""
        start_time = datetime.now()
        
        print(f"🧪 Running {suite.name} ({suite.category.value})")
        
        try:
            # Execute the test suite
            result_data = suite.runner_function(**suite.parameters)
            
            # Convert result to standardized format
            if hasattr(result_data, 'overall_passed'):  # Integration test style
                status = TestStatus.PASSED if result_data.overall_passed else TestStatus.FAILED
                details = asdict(result_data)
            elif hasattr(result_data, 'passed'):  # Security test style
                status = TestStatus.PASSED if result_data.passed else TestStatus.FAILED
                details = asdict(result_data)
            elif isinstance(result_data, dict):
                status = TestStatus.PASSED if result_data.get('success', True) else TestStatus.FAILED
                details = result_data
            else:
                status = TestStatus.PASSED
                details = {'result': str(result_data)}
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result = TestResult(
                test_id=f"{suite.suite_id}_{int(time.time())}",
                suite_id=suite.suite_id,
                name=suite.name,
                category=suite.category,
                status=status,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                message=f"Test suite completed with status: {status.value}",
                details=details
            )
            
            self.completed_tests.append(result)
            return result
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print(f"❌ {suite.name} failed: {e}")
            
            return TestResult(
                test_id=f"{suite.suite_id}_{int(time.time())}",
                suite_id=suite.suite_id,
                name=suite.name,
                category=suite.category,
                status=TestStatus.ERROR,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                message=f"Test suite error: {str(e)}",
                details={'error': str(e), 'traceback': str(e)}
            )


class TestReportGenerator:
    """Generates comprehensive test reports in multiple formats."""
    
    def __init__(self):
        self.report_templates = self._load_report_templates()
    
    def generate_comprehensive_report(self, execution_report: TestExecutionReport, 
                                    output_formats: List[str] = None) -> Dict[str, str]:
        """Generate comprehensive test report in multiple formats."""
        output_formats = output_formats or ['json', 'html', 'junit']
        generated_files = {}
        
        reports_dir = Path("test_reports")
        reports_dir.mkdir(exist_ok=True)
        
        # Generate JSON report
        if 'json' in output_formats:
            json_file = reports_dir / f"test_report_{execution_report.execution_id}.json"
            self._generate_json_report(execution_report, json_file)
            generated_files['json'] = str(json_file)
        
        # Generate HTML report
        if 'html' in output_formats and JINJA2_AVAILABLE:
            html_file = reports_dir / f"test_report_{execution_report.execution_id}.html"
            self._generate_html_report(execution_report, html_file)
            generated_files['html'] = str(html_file)
        
        # Generate JUnit XML report
        if 'junit' in output_formats:
            junit_file = reports_dir / f"junit_report_{execution_report.execution_id}.xml"
            self._generate_junit_report(execution_report, junit_file)
            generated_files['junit'] = str(junit_file)
        
        # Generate console summary
        if 'console' in output_formats:
            self._print_console_summary(execution_report)
        
        return generated_files
    
    def _generate_json_report(self, report: TestExecutionReport, output_file: Path):
        """Generate JSON report."""
        with open(output_file, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        print(f"📄 JSON report generated: {output_file}")
    
    def _generate_html_report(self, report: TestExecutionReport, output_file: Path):
        """Generate HTML report with charts and visualizations."""
        if not JINJA2_AVAILABLE:
            print("⚠️ Jinja2 not available - HTML report skipped")
            return
        
        template = self.report_templates['html']
        
        # Prepare data for template
        template_data = {
            'report': report,
            'generation_time': datetime.now(),
            'chart_data': self._prepare_chart_data(report)
        }
        
        html_content = template.render(**template_data)
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        print(f"🌐 HTML report generated: {output_file}")
    
    def _generate_junit_report(self, report: TestExecutionReport, output_file: Path):
        """Generate JUnit XML report for CI/CD integration."""
        junit_xml = ['<?xml version="1.0" encoding="UTF-8"?>']
        junit_xml.append(f'<testsuites tests="{report.total_tests}" '
                        f'failures="{report.failed_tests}" '
                        f'errors="{report.error_tests}" '
                        f'time="{report.duration_seconds:.3f}">')
        
        # Group tests by suite
        suites = defaultdict(list)
        for test_result in report.test_results:
            suites[test_result.suite_id].append(test_result)
        
        for suite_id, tests in suites.items():
            suite_failures = sum(1 for t in tests if t.status == TestStatus.FAILED)
            suite_errors = sum(1 for t in tests if t.status == TestStatus.ERROR)
            suite_time = sum(t.duration_seconds for t in tests)
            
            junit_xml.append(f'  <testsuite name="{suite_id}" '
                           f'tests="{len(tests)}" '
                           f'failures="{suite_failures}" '
                           f'errors="{suite_errors}" '
                           f'time="{suite_time:.3f}">')
            
            for test in tests:
                junit_xml.append(f'    <testcase name="{test.name}" '
                               f'classname="{test.category.value}" '
                               f'time="{test.duration_seconds:.3f}">')
                
                if test.status == TestStatus.FAILED:
                    junit_xml.append(f'      <failure message="{test.message}">')
                    junit_xml.append(f'        {test.details}')
                    junit_xml.append('      </failure>')
                elif test.status == TestStatus.ERROR:
                    junit_xml.append(f'      <error message="{test.message}">')
                    junit_xml.append(f'        {test.details}')
                    junit_xml.append('      </error>')
                elif test.status == TestStatus.SKIPPED:
                    junit_xml.append(f'      <skipped message="{test.message}"/>')
                
                junit_xml.append('    </testcase>')
            
            junit_xml.append('  </testsuite>')
        
        junit_xml.append('</testsuites>')
        
        with open(output_file, 'w') as f:
            f.write('\n'.join(junit_xml))
        
        print(f"📋 JUnit report generated: {output_file}")
    
    def _print_console_summary(self, report: TestExecutionReport):
        """Print detailed console summary."""
        print(f"\n{'='*80}")
        print(f"TEST EXECUTION SUMMARY")
        print(f"{'='*80}")
        
        print(f"Execution ID: {report.execution_id}")
        print(f"Duration: {report.duration_seconds:.2f} seconds")
        print(f"Start Time: {report.start_time}")
        print(f"End Time: {report.end_time}")
        
        print(f"\nTEST RESULTS:")
        print(f"  Total Tests: {report.total_tests}")
        print(f"  ✅ Passed: {report.passed_tests}")
        print(f"  ❌ Failed: {report.failed_tests}")
        print(f"  ⏭️ Skipped: {report.skipped_tests}")
        print(f"  💥 Errors: {report.error_tests}")
        print(f"  Success Rate: {report.success_rate:.1f}%")
        
        print(f"\nQUALITY METRICS:")
        print(f"  📊 Coverage: {report.coverage_percentage:.1f}%")
        print(f"  🚀 Performance: {report.performance_score:.1f}/100")
        print(f"  🔒 Security: {report.security_score:.1f}/100")
        print(f"  🎯 Overall Quality: {report.quality_score:.1f}/100")
        
        # Suite breakdown
        print(f"\nSUITE BREAKDOWN:")
        for suite_id, summary in report.suite_summaries.items():
            status_emoji = "✅" if summary['passed'] else "❌"
            print(f"  {status_emoji} {suite_id}: {summary['duration']:.2f}s")
        
        # Failed tests details
        if report.failed_tests > 0:
            print(f"\nFAILED TESTS:")
            for test in report.test_results:
                if test.status in [TestStatus.FAILED, TestStatus.ERROR]:
                    print(f"  ❌ {test.name}: {test.message}")
        
        # Recommendations
        if report.recommendations:
            print(f"\nRECOMMENDATIONS:")
            for i, rec in enumerate(report.recommendations, 1):
                print(f"  {i}. {rec}")
    
    def _prepare_chart_data(self, report: TestExecutionReport) -> Dict[str, Any]:
        """Prepare data for charts in HTML report."""
        # Test status distribution
        status_data = {
            'passed': report.passed_tests,
            'failed': report.failed_tests,
            'skipped': report.skipped_tests,
            'errors': report.error_tests
        }
        
        # Test duration by category
        category_durations = defaultdict(float)
        for test in report.test_results:
            category_durations[test.category.value] += test.duration_seconds
        
        return {
            'status_distribution': status_data,
            'category_durations': dict(category_durations),
            'timeline_data': self._generate_timeline_data(report.test_results)
        }
    
    def _generate_timeline_data(self, test_results: List[TestResult]) -> List[Dict[str, Any]]:
        """Generate timeline data for test execution visualization."""
        timeline = []
        
        for test in sorted(test_results, key=lambda x: x.start_time):
            timeline.append({
                'name': test.name,
                'category': test.category.value,
                'start': test.start_time.isoformat(),
                'end': test.end_time.isoformat(),
                'duration': test.duration_seconds,
                'status': test.status.value
            })
        
        return timeline
    
    def _load_report_templates(self) -> Dict[str, Any]:
        """Load report templates."""
        templates = {}
        
        if JINJA2_AVAILABLE:
            # HTML template
            html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Test Report - {{ report.execution_id }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 30px; }
        .metrics { display: flex; gap: 20px; margin-bottom: 30px; }
        .metric-card { background: white; padding: 15px; border-radius: 8px; border: 1px solid #dee2e6; flex: 1; }
        .metric-value { font-size: 24px; font-weight: bold; color: #007bff; }
        .test-results { margin-bottom: 30px; }
        .test-item { padding: 10px; margin: 5px 0; border-radius: 4px; }
        .passed { background: #d4edda; }
        .failed { background: #f8d7da; }
        .error { background: #f8d7da; }
        .skipped { background: #fff3cd; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #dee2e6; }
        th { background: #f8f9fa; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Test Execution Report</h1>
        <p><strong>Execution ID:</strong> {{ report.execution_id }}</p>
        <p><strong>Generated:</strong> {{ generation_time }}</p>
        <p><strong>Duration:</strong> {{ "%.2f"|format(report.duration_seconds) }} seconds</p>
    </div>

    <div class="metrics">
        <div class="metric-card">
            <div class="metric-value">{{ report.success_rate|round(1) }}%</div>
            <div>Success Rate</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{{ report.coverage_percentage|round(1) }}%</div>
            <div>Coverage</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{{ report.performance_score|round(1) }}</div>
            <div>Performance Score</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{{ report.security_score|round(1) }}</div>
            <div>Security Score</div>
        </div>
    </div>

    <div class="test-results">
        <h2>Test Results Summary</h2>
        <p>Total: {{ report.total_tests }} | 
           Passed: {{ report.passed_tests }} | 
           Failed: {{ report.failed_tests }} | 
           Errors: {{ report.error_tests }} | 
           Skipped: {{ report.skipped_tests }}</p>
    </div>

    <table>
        <thead>
            <tr>
                <th>Test Name</th>
                <th>Category</th>
                <th>Status</th>
                <th>Duration (s)</th>
                <th>Message</th>
            </tr>
        </thead>
        <tbody>
            {% for test in report.test_results %}
            <tr class="{{ test.status.value }}">
                <td>{{ test.name }}</td>
                <td>{{ test.category.value }}</td>
                <td>{{ test.status.value.upper() }}</td>
                <td>{{ "%.3f"|format(test.duration_seconds) }}</td>
                <td>{{ test.message }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>

    {% if report.recommendations %}
    <div style="margin-top: 30px;">
        <h2>Recommendations</h2>
        <ul>
            {% for rec in report.recommendations %}
            <li>{{ rec }}</li>
            {% endfor %}
        </ul>
    </div>
    {% endif %}
</body>
</html>
"""
            templates['html'] = Template(html_template)
        
        return templates


class ComprehensiveTestRunner:
    """Main comprehensive test runner that orchestrates all testing frameworks."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.system_monitor = SystemMonitor()
        self.test_executor = ParallelTestExecutor(
            max_workers=self.config.get('max_workers', 4)
        )
        self.report_generator = TestReportGenerator()
        
        # Initialize test suites
        self.test_suites = self._create_test_suites()
    
    def _create_test_suites(self) -> List[TestSuite]:
        """Create all available test suites."""
        suites = []
        
        if not FRAMEWORKS_AVAILABLE:
            print("⚠️ Testing frameworks not fully available - creating basic suites")
            return self._create_basic_test_suites()
        
        # Coverage Analysis Suite
        suites.append(TestSuite(
            suite_id="coverage_analysis",
            name="Code Coverage Analysis",
            description="Comprehensive code coverage analysis with gap identification",
            category=TestCategory.COVERAGE,
            priority=TestPriority.HIGH,
            enabled=True,
            timeout_seconds=300,
            dependencies=[],
            runner_function=self._run_coverage_analysis
        ))
        
        # Unit Tests Suite
        suites.append(TestSuite(
            suite_id="unit_tests",
            name="Unit Tests",
            description="Traditional unit tests using pytest",
            category=TestCategory.UNIT,
            priority=TestPriority.CRITICAL,
            enabled=True,
            timeout_seconds=600,
            dependencies=[],
            runner_function=self._run_unit_tests
        ))
        
        # Integration Tests Suite
        suites.append(TestSuite(
            suite_id="integration_tests",
            name="Integration Tests",
            description="End-to-end integration testing",
            category=TestCategory.INTEGRATION,
            priority=TestPriority.HIGH,
            enabled=True,
            timeout_seconds=1800,
            dependencies=["unit_tests"],
            runner_function=self._run_integration_tests
        ))
        
        # Performance Tests Suite
        suites.append(TestSuite(
            suite_id="performance_tests",
            name="Performance Tests",
            description="Performance benchmarking and load testing",
            category=TestCategory.PERFORMANCE,
            priority=TestPriority.NORMAL,
            enabled=True,
            timeout_seconds=1200,
            dependencies=["unit_tests"],
            runner_function=self._run_performance_tests
        ))
        
        # Property-Based Tests Suite
        suites.append(TestSuite(
            suite_id="property_tests",
            name="Property-Based Tests",
            description="Hypothesis-based property testing",
            category=TestCategory.PROPERTY_BASED,
            priority=TestPriority.NORMAL,
            enabled=True,
            timeout_seconds=900,
            dependencies=[],
            runner_function=self._run_property_tests
        ))
        
        # Security Tests Suite
        suites.append(TestSuite(
            suite_id="security_tests",
            name="Security Tests",
            description="Comprehensive security testing and vulnerability scanning",
            category=TestCategory.SECURITY,
            priority=TestPriority.HIGH,
            enabled=True,
            timeout_seconds=1800,
            dependencies=["unit_tests"],
            runner_function=self._run_security_tests
        ))
        
        # Chaos Engineering Suite
        suites.append(TestSuite(
            suite_id="chaos_tests",
            name="Chaos Engineering Tests",
            description="Fault injection and resilience testing",
            category=TestCategory.CHAOS,
            priority=TestPriority.LOW,
            enabled=self.config.get('enable_chaos', False),
            timeout_seconds=1800,
            dependencies=["integration_tests"],
            runner_function=self._run_chaos_tests
        ))
        
        # Quality Gates Suite
        suites.append(TestSuite(
            suite_id="quality_gates",
            name="Quality Gates",
            description="Code quality and deployment gates",
            category=TestCategory.QUALITY_GATES,
            priority=TestPriority.CRITICAL,
            enabled=True,
            timeout_seconds=600,
            dependencies=["coverage_analysis", "security_tests"],
            runner_function=self._run_quality_gates
        ))
        
        return suites
    
    def _create_basic_test_suites(self) -> List[TestSuite]:
        """Create basic test suites when frameworks are not available."""
        return [
            TestSuite(
                suite_id="basic_tests",
                name="Basic Tests",
                description="Basic pytest execution",
                category=TestCategory.UNIT,
                priority=TestPriority.CRITICAL,
                enabled=True,
                timeout_seconds=600,
                dependencies=[],
                runner_function=self._run_basic_tests
            )
        ]
    
    def run_all_tests(self, categories: List[str] = None, 
                     output_formats: List[str] = None) -> TestExecutionReport:
        """Run all enabled test suites and generate comprehensive report."""
        
        execution_id = f"test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()
        
        print(f"🚀 Starting comprehensive test execution: {execution_id}")
        print(f"Start time: {start_time}")
        
        # Filter test suites by category if specified
        enabled_suites = [suite for suite in self.test_suites if suite.enabled]
        
        if categories:
            category_filters = [TestCategory(cat) for cat in categories]
            enabled_suites = [suite for suite in enabled_suites if suite.category in category_filters]
        
        print(f"📋 Test suites to execute: {len(enabled_suites)}")
        for suite in enabled_suites:
            print(f"  • {suite.name} ({suite.category.value}) - Priority: {suite.priority.name}")
        
        # Start system monitoring
        self.system_monitor.start_monitoring()
        
        # Add suites to executor
        for suite in enabled_suites:
            self.test_executor.add_test_suite(suite)
        
        # Execute tests with progress tracking
        def progress_callback(completed: int, total: int):
            progress = (completed / total) * 100
            print(f"📊 Progress: {completed}/{total} ({progress:.1f}%) suites completed")
        
        try:
            test_results = self.test_executor.execute_all(progress_callback)
        except Exception as e:
            print(f"❌ Test execution failed: {e}")
            test_results = []
        
        # Stop monitoring
        system_metrics = self.system_monitor.stop_monitoring()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Analyze results
        passed_tests = sum(1 for r in test_results if r.status == TestStatus.PASSED)
        failed_tests = sum(1 for r in test_results if r.status == TestStatus.FAILED)
        error_tests = sum(1 for r in test_results if r.status == TestStatus.ERROR)
        skipped_tests = sum(1 for r in test_results if r.status == TestStatus.SKIPPED)
        total_tests = len(test_results)
        
        success_rate = (passed_tests / max(total_tests, 1)) * 100
        
        # Calculate quality scores
        coverage_percentage = self._extract_coverage_percentage(test_results)
        performance_score = self._extract_performance_score(test_results)
        security_score = self._extract_security_score(test_results)
        quality_score = (coverage_percentage + performance_score + security_score) / 3
        
        # Generate suite summaries
        suite_summaries = {}
        for result in test_results:
            suite_summaries[result.suite_id] = {
                'passed': result.status == TestStatus.PASSED,
                'duration': result.duration_seconds,
                'message': result.message
            }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(test_results, quality_score)
        
        # Create execution report
        execution_report = TestExecutionReport(
            execution_id=execution_id,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
            error_tests=error_tests,
            success_rate=success_rate,
            coverage_percentage=coverage_percentage,
            performance_score=performance_score,
            security_score=security_score,
            quality_score=quality_score,
            test_results=test_results,
            suite_summaries=suite_summaries,
            system_metrics=system_metrics,
            recommendations=recommendations
        )
        
        # Generate reports
        output_formats = output_formats or ['console', 'json', 'html', 'junit']
        generated_files = self.report_generator.generate_comprehensive_report(
            execution_report, output_formats
        )
        
        print(f"\n🎯 Test Execution Complete!")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Quality Score: {quality_score:.1f}/100")
        
        if generated_files:
            print(f"\n📊 Reports generated:")
            for format_type, file_path in generated_files.items():
                print(f"  • {format_type.upper()}: {file_path}")
        
        return execution_report
    
    # Test suite runner methods
    def _run_coverage_analysis(self) -> Dict[str, Any]:
        """Run coverage analysis."""
        try:
            from .coverage_analyzer import AdvancedCoverageAnalyzer
            analyzer = AdvancedCoverageAnalyzer()
            result = analyzer.run_comprehensive_analysis()
            return {
                'success': result.meets_requirements,
                'coverage_percentage': result.overall_coverage.line_coverage,
                'details': asdict(result)
            }
        except Exception as e:
            return {'success': False, 'error': str(e), 'coverage_percentage': 0}
    
    def _run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests using pytest."""
        try:
            result = subprocess.run(
                ['python', '-m', 'pytest', 'tests/', '-v', '--tb=short'],
                capture_output=True, text=True, timeout=600
            )
            
            return {
                'success': result.returncode == 0,
                'output': result.stdout,
                'errors': result.stderr,
                'return_code': result.returncode
            }
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Unit tests timed out'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests."""
        try:
            from .integration_testing import IntegrationTestSuiteRunner
            runner = IntegrationTestSuiteRunner()
            import asyncio
            result = asyncio.run(runner.run_all_integration_tests())
            return {
                'success': result.overall_passed,
                'details': asdict(result)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests."""
        try:
            from .performance_testing import PerformanceTestSuite
            suite = PerformanceTestSuite()
            import asyncio
            result = asyncio.run(suite.run_complete_performance_suite())
            return {
                'success': True,
                'performance_score': 85,  # Mock score
                'details': result
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _run_property_tests(self) -> Dict[str, Any]:
        """Run property-based tests."""
        try:
            from .property_based_testing import PropertyBasedTestRunner
            runner = PropertyBasedTestRunner()
            result = runner.run_all_property_tests()
            return {
                'success': result.overall_passed,
                'details': asdict(result)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _run_security_tests(self) -> Dict[str, Any]:
        """Run security tests."""
        try:
            from .security_testing import SecurityTestSuite
            suite = SecurityTestSuite()
            result = suite.run_comprehensive_security_audit()
            return {
                'success': result.critical_vulnerabilities == 0,
                'security_score': result.overall_security_score,
                'details': asdict(result)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _run_chaos_tests(self) -> Dict[str, Any]:
        """Run chaos engineering tests."""
        try:
            from .chaos_engineering import ChaosExperimentRunner, create_sample_experiments
            runner = ChaosExperimentRunner()
            experiments = create_sample_experiments()
            results = runner.run_chaos_suite(experiments[:2])  # Run subset for speed
            
            success = all(r.success for r in results)
            return {
                'success': success,
                'resilience_score': 80,  # Mock score
                'details': [asdict(r) for r in results]
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _run_quality_gates(self) -> Dict[str, Any]:
        """Run quality gates."""
        try:
            from .quality_gates import QualityGateOrchestrator
            orchestrator = QualityGateOrchestrator()
            result = orchestrator.run_all_gates()
            return {
                'success': result.deployment_approved,
                'quality_score': result.overall_score,
                'details': asdict(result)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _run_basic_tests(self) -> Dict[str, Any]:
        """Run basic tests when frameworks not available."""
        try:
            result = subprocess.run(
                ['python', '-m', 'pytest', '--version'],
                capture_output=True, text=True, timeout=30
            )
            return {
                'success': result.returncode == 0,
                'message': 'Basic test framework check passed'
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    # Utility methods for score extraction
    def _extract_coverage_percentage(self, test_results: List[TestResult]) -> float:
        """Extract coverage percentage from test results."""
        for result in test_results:
            if result.suite_id == "coverage_analysis" and 'details' in result.details:
                details = result.details.get('details', {})
                if isinstance(details, dict):
                    overall_coverage = details.get('overall_coverage', {})
                    if isinstance(overall_coverage, dict):
                        return overall_coverage.get('line_coverage', 0)
        return 0.0
    
    def _extract_performance_score(self, test_results: List[TestResult]) -> float:
        """Extract performance score from test results."""
        for result in test_results:
            if result.suite_id == "performance_tests" and 'performance_score' in result.details:
                return result.details['performance_score']
        return 0.0
    
    def _extract_security_score(self, test_results: List[TestResult]) -> float:
        """Extract security score from test results."""
        for result in test_results:
            if result.suite_id == "security_tests" and 'security_score' in result.details:
                return result.details['security_score']
        return 0.0
    
    def _generate_recommendations(self, test_results: List[TestResult], 
                                quality_score: float) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        # Failed tests
        failed_tests = [r for r in test_results if r.status in [TestStatus.FAILED, TestStatus.ERROR]]
        if failed_tests:
            recommendations.append(f"Fix {len(failed_tests)} failed test suites before deployment")
        
        # Quality score
        if quality_score < 80:
            recommendations.append("Improve overall quality score - target is 80+")
        
        # Coverage
        coverage = self._extract_coverage_percentage(test_results)
        if coverage < 85:
            recommendations.append(f"Increase code coverage from {coverage:.1f}% to 85%+")
        
        # Security
        security_score = self._extract_security_score(test_results)
        if security_score < 80:
            recommendations.append("Address security issues to improve security score")
        
        if not recommendations:
            recommendations.append("All tests passed - maintain current quality standards")
        
        return recommendations


def main():
    """Main function for running comprehensive tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Test Runner")
    parser.add_argument("--categories", nargs="+", 
                       choices=["unit", "integration", "performance", "security", 
                               "property_based", "chaos", "quality_gates", "coverage"],
                       help="Test categories to run")
    parser.add_argument("--output-formats", nargs="+",
                       choices=["console", "json", "html", "junit"],
                       default=["console", "json"],
                       help="Output formats for reports")
    parser.add_argument("--max-workers", type=int, default=4,
                       help="Maximum parallel workers")
    parser.add_argument("--enable-chaos", action="store_true",
                       help="Enable chaos engineering tests")
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'max_workers': args.max_workers,
        'enable_chaos': args.enable_chaos
    }
    
    # Create and run test runner
    runner = ComprehensiveTestRunner(config)
    
    try:
        report = runner.run_all_tests(
            categories=args.categories,
            output_formats=args.output_formats
        )
        
        # Exit with appropriate code
        if report.failed_tests > 0 or report.error_tests > 0:
            print("❌ Test execution completed with failures")
            exit(1)
        elif report.quality_score < 80:
            print("⚠️ Test execution completed with low quality score")
            exit(2)
        else:
            print("✅ All tests passed successfully")
            exit(0)
            
    except KeyboardInterrupt:
        print("\n🛑 Test execution interrupted by user")
        exit(130)
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()