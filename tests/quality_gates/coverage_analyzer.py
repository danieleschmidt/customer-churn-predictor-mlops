"""
Advanced Test Coverage Analysis and Quality Gates.

This module provides comprehensive coverage analysis including:
- Branch and line coverage reporting with detailed metrics
- Missing test identification and automated suggestions
- Coverage requirements enforcement with customizable thresholds
- Test gap analysis and prioritized testing recommendations
- Integration with quality gates and CI/CD pipelines
"""

import os
import ast
import json
import time
import sqlite3
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Set, NamedTuple
from dataclasses import dataclass, asdict, field
from collections import defaultdict, Counter
import coverage
import pytest
from coverage import Coverage
from coverage.results import Numbers
from coverage.html import HtmlReporter
from coverage.xmlreport import XmlReporter
import xml.etree.ElementTree as ET


class CoverageLevel(NamedTuple):
    """Coverage level classification."""
    line_coverage: float
    branch_coverage: float
    function_coverage: float
    overall_score: float


@dataclass
class TestGap:
    """Represents a gap in test coverage."""
    file_path: str
    function_name: str
    start_line: int
    end_line: int
    complexity: int
    priority: str  # 'critical', 'high', 'medium', 'low'
    reason: str
    suggested_test: str


@dataclass
class CoverageRequirement:
    """Coverage requirement definition."""
    name: str
    min_line_coverage: float
    min_branch_coverage: float
    min_function_coverage: float
    applies_to: List[str]  # File patterns
    exceptions: List[str] = field(default_factory=list)


@dataclass
class CoverageAnalysisResult:
    """Complete coverage analysis result."""
    analysis_id: str
    timestamp: datetime
    overall_coverage: CoverageLevel
    file_coverage: Dict[str, CoverageLevel]
    requirements_status: Dict[str, bool]
    test_gaps: List[TestGap]
    suggestions: List[str]
    quality_score: float
    meets_requirements: bool


class CodeComplexityAnalyzer:
    """Analyzes code complexity to prioritize testing."""
    
    def __init__(self):
        self.complexity_weights = {
            'if': 1,
            'elif': 1,
            'else': 1,
            'for': 1,
            'while': 1,
            'try': 1,
            'except': 1,
            'with': 1,
            'and': 0.5,
            'or': 0.5,
            'lambda': 1
        }
    
    def calculate_cyclomatic_complexity(self, file_path: str) -> Dict[str, int]:
        """Calculate cyclomatic complexity for functions in a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
            
            complexity_map = {}
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    complexity = self._calculate_function_complexity(node)
                    complexity_map[node.name] = complexity
            
            return complexity_map
        except Exception as e:
            print(f"Error calculating complexity for {file_path}: {e}")
            return {}
    
    def _calculate_function_complexity(self, func_node: ast.FunctionDef) -> int:
        """Calculate complexity for a single function."""
        complexity = 1  # Base complexity
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.If):
                complexity += 1
            elif isinstance(node, ast.For):
                complexity += 1
            elif isinstance(node, ast.While):
                complexity += 1
            elif isinstance(node, ast.Try):
                complexity += len(node.handlers)
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            elif isinstance(node, ast.Lambda):
                complexity += 1
        
        return complexity


class TestSuggestionGenerator:
    """Generates automated test suggestions based on code analysis."""
    
    def __init__(self):
        self.test_patterns = {
            'data_validation': [
                'test_with_valid_data',
                'test_with_invalid_data',
                'test_with_edge_cases',
                'test_with_null_values',
                'test_with_boundary_values'
            ],
            'ml_model': [
                'test_model_training',
                'test_model_prediction',
                'test_model_serialization',
                'test_model_performance_regression',
                'test_model_with_different_data_sizes'
            ],
            'api_endpoint': [
                'test_valid_request',
                'test_invalid_request',
                'test_authentication',
                'test_rate_limiting',
                'test_error_handling'
            ],
            'database': [
                'test_connection',
                'test_crud_operations',
                'test_transaction_rollback',
                'test_concurrent_access',
                'test_data_integrity'
            ],
            'security': [
                'test_input_sanitization',
                'test_authentication',
                'test_authorization',
                'test_injection_attacks',
                'test_xss_prevention'
            ]
        }
    
    def generate_test_suggestions(self, file_path: str, function_name: str, 
                                code_content: str) -> List[str]:
        """Generate test suggestions for a specific function."""
        suggestions = []
        
        # Analyze function to determine category
        category = self._determine_function_category(function_name, code_content)
        
        # Get base test patterns
        base_patterns = self.test_patterns.get(category, ['test_basic_functionality'])
        
        # Generate specific test suggestions
        for pattern in base_patterns:
            test_name = f"{pattern}_{function_name}"
            suggestions.append(self._generate_test_template(test_name, function_name, category))
        
        return suggestions
    
    def _determine_function_category(self, function_name: str, code_content: str) -> str:
        """Determine the category of a function based on its content."""
        function_name_lower = function_name.lower()
        code_lower = code_content.lower()
        
        if any(keyword in function_name_lower for keyword in ['validate', 'check', 'verify']):
            return 'data_validation'
        elif any(keyword in function_name_lower for keyword in ['train', 'predict', 'model']):
            return 'ml_model'
        elif any(keyword in code_lower for keyword in ['fastapi', 'route', 'endpoint']):
            return 'api_endpoint'
        elif any(keyword in code_lower for keyword in ['database', 'sql', 'query']):
            return 'database'
        elif any(keyword in function_name_lower for keyword in ['auth', 'security', 'encrypt']):
            return 'security'
        else:
            return 'general'
    
    def _generate_test_template(self, test_name: str, function_name: str, category: str) -> str:
        """Generate a test template for the given function."""
        templates = {
            'data_validation': f'''
def {test_name}(self):
    """Test {function_name} with various data scenarios."""
    # Arrange
    test_data = self._create_test_data()
    
    # Act
    result = {function_name}(test_data)
    
    # Assert
    assert result is not None
    # Add specific assertions based on expected behavior
''',
            'ml_model': f'''
def {test_name}(self):
    """Test {function_name} ML functionality."""
    # Arrange
    X_test, y_test = self._get_test_data()
    
    # Act
    result = {function_name}(X_test)
    
    # Assert
    assert result.shape[0] == X_test.shape[0]
    # Add model-specific assertions
''',
            'api_endpoint': f'''
def {test_name}(self, api_client):
    """Test {function_name} API endpoint."""
    # Arrange
    test_payload = self._create_test_payload()
    
    # Act
    response = api_client.post('/endpoint', json=test_payload)
    
    # Assert
    assert response.status_code == 200
    # Add response validation
''',
            'general': f'''
def {test_name}(self):
    """Test {function_name} functionality."""
    # Arrange
    test_input = self._create_test_input()
    
    # Act
    result = {function_name}(test_input)
    
    # Assert
    assert result is not None
    # Add specific assertions
'''
        }
        
        return templates.get(category, templates['general'])


class AdvancedCoverageAnalyzer:
    """Advanced test coverage analyzer with quality gates."""
    
    def __init__(self, source_dir: str = "src", test_dir: str = "tests"):
        self.source_dir = Path(source_dir)
        self.test_dir = Path(test_dir)
        self.complexity_analyzer = CodeComplexityAnalyzer()
        self.suggestion_generator = TestSuggestionGenerator()
        
        # Default coverage requirements
        self.requirements = [
            CoverageRequirement(
                name="critical_modules",
                min_line_coverage=95.0,
                min_branch_coverage=90.0,
                min_function_coverage=100.0,
                applies_to=["*security*", "*auth*", "*validation*"],
                exceptions=["*test*", "*__init__*"]
            ),
            CoverageRequirement(
                name="core_modules",
                min_line_coverage=85.0,
                min_branch_coverage=80.0,
                min_function_coverage=90.0,
                applies_to=["*optimization*", "*distributed*", "*error_handling*"],
                exceptions=["*test*", "*__init__*"]
            ),
            CoverageRequirement(
                name="general_modules",
                min_line_coverage=80.0,
                min_branch_coverage=75.0,
                min_function_coverage=85.0,
                applies_to=["*"],
                exceptions=["*test*", "*__init__*", "*conftest*"]
            )
        ]
        
        # Coverage database for tracking history
        self.db_path = Path("coverage_history.db")
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize coverage history database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS coverage_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                overall_line_coverage REAL,
                overall_branch_coverage REAL,
                overall_function_coverage REAL,
                quality_score REAL,
                meets_requirements BOOLEAN,
                file_count INTEGER,
                test_count INTEGER
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS file_coverage_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id INTEGER,
                file_path TEXT,
                line_coverage REAL,
                branch_coverage REAL,
                function_coverage REAL,
                complexity INTEGER,
                FOREIGN KEY (analysis_id) REFERENCES coverage_history (id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_gaps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id INTEGER,
                file_path TEXT,
                function_name TEXT,
                complexity INTEGER,
                priority TEXT,
                reason TEXT,
                resolved BOOLEAN DEFAULT FALSE,
                FOREIGN KEY (analysis_id) REFERENCES coverage_history (id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def run_comprehensive_analysis(self, include_suggestions: bool = True) -> CoverageAnalysisResult:
        """Run comprehensive coverage analysis with quality gates."""
        analysis_id = f"analysis_{int(time.time())}"
        timestamp = datetime.now()
        
        print("🔍 Running comprehensive coverage analysis...")
        
        # Run coverage measurement
        cov = Coverage(
            source=[str(self.source_dir)],
            branch=True,
            config_file=True
        )
        
        cov.start()
        
        # Run tests to collect coverage
        print("📊 Collecting coverage data...")
        result = subprocess.run([
            "python", "-m", "pytest", str(self.test_dir),
            "--tb=short", "-v"
        ], capture_output=True, text=True, cwd=".")
        
        cov.stop()
        cov.save()
        
        # Analyze coverage data
        overall_coverage = self._calculate_overall_coverage(cov)
        file_coverage = self._calculate_file_coverage(cov)
        
        # Check requirements compliance
        requirements_status = self._check_requirements_compliance(file_coverage)
        
        # Identify test gaps
        test_gaps = self._identify_test_gaps(cov, file_coverage)
        
        # Generate suggestions if requested
        suggestions = []
        if include_suggestions:
            suggestions = self._generate_improvement_suggestions(
                overall_coverage, file_coverage, test_gaps
            )
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(overall_coverage, requirements_status)
        
        # Check if all requirements are met
        meets_requirements = all(requirements_status.values())
        
        # Create analysis result
        analysis_result = CoverageAnalysisResult(
            analysis_id=analysis_id,
            timestamp=timestamp,
            overall_coverage=overall_coverage,
            file_coverage=file_coverage,
            requirements_status=requirements_status,
            test_gaps=test_gaps,
            suggestions=suggestions,
            quality_score=quality_score,
            meets_requirements=meets_requirements
        )
        
        # Store in database
        self._store_analysis_result(analysis_result)
        
        # Generate reports
        self._generate_coverage_reports(cov, analysis_result)
        
        return analysis_result
    
    def _calculate_overall_coverage(self, cov: Coverage) -> CoverageLevel:
        """Calculate overall coverage metrics."""
        # Get coverage data
        data = cov.get_data()
        analysis = coverage.Analysis(cov, '<overall>')
        
        # Calculate line coverage
        total_lines = 0
        covered_lines = 0
        
        # Calculate branch coverage
        total_branches = 0
        covered_branches = 0
        
        # Calculate function coverage
        total_functions = 0
        covered_functions = 0
        
        for filename in data.measured_files():
            try:
                file_analysis = coverage.Analysis(cov, filename)
                
                # Line coverage
                total_lines += len(file_analysis.statements)
                covered_lines += len(file_analysis.statements) - len(file_analysis.missing)
                
                # Branch coverage (if available)
                if hasattr(file_analysis, 'branch_lines'):
                    for line in file_analysis.branch_lines():
                        branches = data.lines(filename).get(line, [])
                        total_branches += len(branches)
                        covered_branches += len([b for b in branches if b])
                
                # Function coverage (approximated)
                complexity_data = self.complexity_analyzer.calculate_cyclomatic_complexity(filename)
                total_functions += len(complexity_data)
                
                # Estimate covered functions based on line coverage
                if file_analysis.statements:
                    func_coverage_ratio = (len(file_analysis.statements) - len(file_analysis.missing)) / len(file_analysis.statements)
                    covered_functions += int(len(complexity_data) * func_coverage_ratio)
                
            except Exception as e:
                print(f"Error analyzing {filename}: {e}")
                continue
        
        # Calculate percentages
        line_coverage = (covered_lines / max(total_lines, 1)) * 100
        branch_coverage = (covered_branches / max(total_branches, 1)) * 100
        function_coverage = (covered_functions / max(total_functions, 1)) * 100
        
        # Overall score is weighted average
        overall_score = (
            line_coverage * 0.5 +
            branch_coverage * 0.3 +
            function_coverage * 0.2
        )
        
        return CoverageLevel(
            line_coverage=line_coverage,
            branch_coverage=branch_coverage,
            function_coverage=function_coverage,
            overall_score=overall_score
        )
    
    def _calculate_file_coverage(self, cov: Coverage) -> Dict[str, CoverageLevel]:
        """Calculate coverage metrics for each file."""
        file_coverage = {}
        data = cov.get_data()
        
        for filename in data.measured_files():
            try:
                # Skip test files
                if "test" in filename.lower() or filename.endswith("conftest.py"):
                    continue
                
                file_analysis = coverage.Analysis(cov, filename)
                
                # Line coverage
                total_lines = len(file_analysis.statements)
                covered_lines = total_lines - len(file_analysis.missing)
                line_coverage = (covered_lines / max(total_lines, 1)) * 100
                
                # Branch coverage (simplified)
                branch_coverage = line_coverage  # Approximation for now
                
                # Function coverage
                complexity_data = self.complexity_analyzer.calculate_cyclomatic_complexity(filename)
                total_functions = len(complexity_data)
                
                if total_functions > 0 and total_lines > 0:
                    func_coverage_ratio = covered_lines / total_lines
                    covered_functions = int(total_functions * func_coverage_ratio)
                    function_coverage = (covered_functions / total_functions) * 100
                else:
                    function_coverage = 100.0
                
                # Overall score for the file
                overall_score = (
                    line_coverage * 0.5 +
                    branch_coverage * 0.3 +
                    function_coverage * 0.2
                )
                
                file_coverage[filename] = CoverageLevel(
                    line_coverage=line_coverage,
                    branch_coverage=branch_coverage,
                    function_coverage=function_coverage,
                    overall_score=overall_score
                )
                
            except Exception as e:
                print(f"Error calculating coverage for {filename}: {e}")
                continue
        
        return file_coverage
    
    def _check_requirements_compliance(self, file_coverage: Dict[str, CoverageLevel]) -> Dict[str, bool]:
        """Check if files meet coverage requirements."""
        compliance_status = {}
        
        for requirement in self.requirements:
            compliant_files = 0
            total_applicable_files = 0
            
            for filename, coverage_data in file_coverage.items():
                # Check if requirement applies to this file
                if self._file_matches_pattern(filename, requirement.applies_to, requirement.exceptions):
                    total_applicable_files += 1
                    
                    # Check compliance
                    if (coverage_data.line_coverage >= requirement.min_line_coverage and
                        coverage_data.branch_coverage >= requirement.min_branch_coverage and
                        coverage_data.function_coverage >= requirement.min_function_coverage):
                        compliant_files += 1
            
            # Requirement is met if all applicable files are compliant
            compliance_status[requirement.name] = (
                compliant_files == total_applicable_files and total_applicable_files > 0
            )
        
        return compliance_status
    
    def _file_matches_pattern(self, filename: str, includes: List[str], excludes: List[str]) -> bool:
        """Check if a file matches the given patterns."""
        import fnmatch
        
        # Check exclusions first
        for pattern in excludes:
            if fnmatch.fnmatch(filename, pattern):
                return False
        
        # Check inclusions
        for pattern in includes:
            if fnmatch.fnmatch(filename, pattern):
                return True
        
        return False
    
    def _identify_test_gaps(self, cov: Coverage, file_coverage: Dict[str, CoverageLevel]) -> List[TestGap]:
        """Identify gaps in test coverage and prioritize them."""
        test_gaps = []
        
        for filename, coverage_data in file_coverage.items():
            try:
                file_analysis = coverage.Analysis(cov, filename)
                complexity_data = self.complexity_analyzer.calculate_cyclomatic_complexity(filename)
                
                # Get missing lines
                missing_lines = file_analysis.missing
                
                if missing_lines:
                    # Parse file to get function information
                    with open(filename, 'r', encoding='utf-8') as f:
                        tree = ast.parse(f.read())
                    
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            func_start = node.lineno
                            func_end = node.end_lineno if hasattr(node, 'end_lineno') else func_start
                            
                            # Check if function has missing coverage
                            func_missing_lines = [line for line in missing_lines 
                                                if func_start <= line <= func_end]
                            
                            if func_missing_lines:
                                complexity = complexity_data.get(node.name, 1)
                                priority = self._calculate_priority(
                                    complexity, len(func_missing_lines), coverage_data.overall_score
                                )
                                
                                test_gap = TestGap(
                                    file_path=filename,
                                    function_name=node.name,
                                    start_line=func_start,
                                    end_line=func_end,
                                    complexity=complexity,
                                    priority=priority,
                                    reason=f"Missing coverage on {len(func_missing_lines)} lines",
                                    suggested_test=f"test_{node.name}_coverage"
                                )
                                test_gaps.append(test_gap)
                
            except Exception as e:
                print(f"Error identifying gaps for {filename}: {e}")
                continue
        
        # Sort by priority
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        test_gaps.sort(key=lambda x: (priority_order[x.priority], -x.complexity))
        
        return test_gaps
    
    def _calculate_priority(self, complexity: int, missing_lines: int, coverage_score: float) -> str:
        """Calculate priority for a test gap."""
        # High complexity functions with low coverage are critical
        if complexity >= 10 and coverage_score < 50:
            return 'critical'
        elif complexity >= 5 and coverage_score < 70:
            return 'high'
        elif complexity >= 3 or coverage_score < 85:
            return 'medium'
        else:
            return 'low'
    
    def _generate_improvement_suggestions(self, overall_coverage: CoverageLevel,
                                        file_coverage: Dict[str, CoverageLevel],
                                        test_gaps: List[TestGap]) -> List[str]:
        """Generate actionable improvement suggestions."""
        suggestions = []
        
        # Overall coverage suggestions
        if overall_coverage.line_coverage < 85:
            suggestions.append(
                f"📈 Increase line coverage from {overall_coverage.line_coverage:.1f}% to 85%+ by adding tests for uncovered code paths"
            )
        
        if overall_coverage.branch_coverage < 80:
            suggestions.append(
                f"🌿 Improve branch coverage from {overall_coverage.branch_coverage:.1f}% to 80%+ by testing all conditional paths"
            )
        
        # File-specific suggestions
        low_coverage_files = [
            (filename, coverage_data) for filename, coverage_data in file_coverage.items()
            if coverage_data.overall_score < 70
        ]
        
        if low_coverage_files:
            suggestions.append(
                f"🎯 Priority files needing attention: {', '.join([Path(f[0]).name for f in low_coverage_files[:5]])}"
            )
        
        # Test gap suggestions
        critical_gaps = [gap for gap in test_gaps if gap.priority == 'critical']
        if critical_gaps:
            suggestions.append(
                f"🚨 {len(critical_gaps)} critical functions need immediate test coverage"
            )
        
        high_priority_gaps = [gap for gap in test_gaps if gap.priority == 'high']
        if high_priority_gaps:
            suggestions.append(
                f"⚠️ {len(high_priority_gaps)} high-priority functions should be tested next"
            )
        
        # Specific test suggestions for top gaps
        for gap in test_gaps[:3]:  # Top 3 gaps
            test_suggestions = self.suggestion_generator.generate_test_suggestions(
                gap.file_path, gap.function_name, ""
            )
            if test_suggestions:
                suggestions.append(
                    f"📝 For {Path(gap.file_path).name}::{gap.function_name}: {test_suggestions[0][:100]}..."
                )
        
        return suggestions
    
    def _calculate_quality_score(self, overall_coverage: CoverageLevel,
                               requirements_status: Dict[str, bool]) -> float:
        """Calculate overall quality score (0-100)."""
        # Base score from coverage metrics
        coverage_score = overall_coverage.overall_score
        
        # Requirements compliance bonus/penalty
        compliance_ratio = sum(requirements_status.values()) / max(len(requirements_status), 1)
        compliance_bonus = compliance_ratio * 10  # Up to 10 points bonus
        
        # Final score (capped at 100)
        final_score = min(coverage_score + compliance_bonus, 100.0)
        
        return final_score
    
    def _store_analysis_result(self, result: CoverageAnalysisResult) -> None:
        """Store analysis result in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Store main analysis record
        cursor.execute("""
            INSERT INTO coverage_history 
            (timestamp, overall_line_coverage, overall_branch_coverage, overall_function_coverage,
             quality_score, meets_requirements, file_count, test_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result.timestamp.isoformat(),
            result.overall_coverage.line_coverage,
            result.overall_coverage.branch_coverage,
            result.overall_coverage.function_coverage,
            result.quality_score,
            result.meets_requirements,
            len(result.file_coverage),
            len(result.test_gaps)
        ))
        
        analysis_id = cursor.lastrowid
        
        # Store file coverage data
        for filename, coverage_data in result.file_coverage.items():
            complexity = sum(self.complexity_analyzer.calculate_cyclomatic_complexity(filename).values())
            
            cursor.execute("""
                INSERT INTO file_coverage_history 
                (analysis_id, file_path, line_coverage, branch_coverage, function_coverage, complexity)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                analysis_id, filename, coverage_data.line_coverage,
                coverage_data.branch_coverage, coverage_data.function_coverage, complexity
            ))
        
        # Store test gaps
        for gap in result.test_gaps:
            cursor.execute("""
                INSERT INTO test_gaps 
                (analysis_id, file_path, function_name, complexity, priority, reason)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                analysis_id, gap.file_path, gap.function_name,
                gap.complexity, gap.priority, gap.reason
            ))
        
        conn.commit()
        conn.close()
    
    def _generate_coverage_reports(self, cov: Coverage, result: CoverageAnalysisResult) -> None:
        """Generate comprehensive coverage reports."""
        reports_dir = Path("coverage_reports")
        reports_dir.mkdir(exist_ok=True)
        
        # Generate HTML report
        html_reporter = HtmlReporter(cov, config=cov.config)
        html_reporter.report(str(reports_dir / "html"))
        
        # Generate XML report
        xml_reporter = XmlReporter(cov, config=cov.config)
        with open(reports_dir / "coverage.xml", "w") as xml_file:
            xml_reporter.report(xml_file)
        
        # Generate JSON report with analysis data
        json_report = {
            "analysis_id": result.analysis_id,
            "timestamp": result.timestamp.isoformat(),
            "overall_coverage": asdict(result.overall_coverage),
            "file_coverage": {k: asdict(v) for k, v in result.file_coverage.items()},
            "requirements_status": result.requirements_status,
            "test_gaps": [asdict(gap) for gap in result.test_gaps],
            "suggestions": result.suggestions,
            "quality_score": result.quality_score,
            "meets_requirements": result.meets_requirements
        }
        
        with open(reports_dir / "analysis_report.json", "w") as json_file:
            json.dump(json_report, json_file, indent=2)
        
        print(f"📊 Coverage reports generated in {reports_dir}/")
    
    def get_coverage_trends(self, days: int = 30) -> Dict[str, Any]:
        """Get coverage trends over time."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get historical data
        cursor.execute("""
            SELECT timestamp, overall_line_coverage, overall_branch_coverage, 
                   quality_score, meets_requirements
            FROM coverage_history 
            WHERE timestamp >= datetime('now', '-{} days')
            ORDER BY timestamp
        """.format(days))
        
        history = cursor.fetchall()
        conn.close()
        
        if not history:
            return {"message": "No historical data available"}
        
        # Calculate trends
        line_coverages = [row[1] for row in history]
        branch_coverages = [row[2] for row in history]
        quality_scores = [row[3] for row in history]
        
        return {
            "total_analyses": len(history),
            "date_range": {
                "start": history[0][0],
                "end": history[-1][0]
            },
            "line_coverage": {
                "current": line_coverages[-1],
                "trend": "improving" if len(line_coverages) > 1 and line_coverages[-1] > line_coverages[0] else "declining",
                "min": min(line_coverages),
                "max": max(line_coverages),
                "average": sum(line_coverages) / len(line_coverages)
            },
            "branch_coverage": {
                "current": branch_coverages[-1],
                "trend": "improving" if len(branch_coverages) > 1 and branch_coverages[-1] > branch_coverages[0] else "declining",
                "min": min(branch_coverages),
                "max": max(branch_coverages),
                "average": sum(branch_coverages) / len(branch_coverages)
            },
            "quality_score": {
                "current": quality_scores[-1],
                "trend": "improving" if len(quality_scores) > 1 and quality_scores[-1] > quality_scores[0] else "declining",
                "min": min(quality_scores),
                "max": max(quality_scores),
                "average": sum(quality_scores) / len(quality_scores)
            }
        }
    
    def enforce_quality_gates(self, result: CoverageAnalysisResult, 
                            fail_build: bool = True) -> Tuple[bool, List[str]]:
        """Enforce quality gates and return pass/fail status."""
        failures = []
        
        # Check overall coverage requirements
        if result.overall_coverage.line_coverage < 85:
            failures.append(f"Line coverage {result.overall_coverage.line_coverage:.1f}% below minimum 85%")
        
        if result.overall_coverage.branch_coverage < 80:
            failures.append(f"Branch coverage {result.overall_coverage.branch_coverage:.1f}% below minimum 80%")
        
        if result.quality_score < 80:
            failures.append(f"Quality score {result.quality_score:.1f} below minimum 80")
        
        # Check requirements compliance
        if not result.meets_requirements:
            failed_requirements = [req for req, passed in result.requirements_status.items() if not passed]
            failures.append(f"Failed requirements: {', '.join(failed_requirements)}")
        
        # Check for critical test gaps
        critical_gaps = [gap for gap in result.test_gaps if gap.priority == 'critical']
        if critical_gaps:
            failures.append(f"{len(critical_gaps)} critical functions without test coverage")
        
        passed = len(failures) == 0
        
        if not passed and fail_build:
            print("❌ Quality gates FAILED:")
            for failure in failures:
                print(f"  • {failure}")
        elif passed:
            print("✅ All quality gates PASSED")
        
        return passed, failures


def main():
    """Main function for running coverage analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced Coverage Analysis")
    parser.add_argument("--source", default="src", help="Source directory")
    parser.add_argument("--tests", default="tests", help="Tests directory")
    parser.add_argument("--fail-build", action="store_true", help="Fail build on quality gate failures")
    parser.add_argument("--trends", type=int, help="Show coverage trends for N days")
    
    args = parser.parse_args()
    
    analyzer = AdvancedCoverageAnalyzer(args.source, args.tests)
    
    if args.trends:
        trends = analyzer.get_coverage_trends(args.trends)
        print("📈 Coverage Trends:")
        print(json.dumps(trends, indent=2))
        return
    
    # Run analysis
    result = analyzer.run_comprehensive_analysis()
    
    # Print summary
    print("\n🎯 Coverage Analysis Summary:")
    print(f"Line Coverage: {result.overall_coverage.line_coverage:.1f}%")
    print(f"Branch Coverage: {result.overall_coverage.branch_coverage:.1f}%")
    print(f"Function Coverage: {result.overall_coverage.function_coverage:.1f}%")
    print(f"Quality Score: {result.quality_score:.1f}/100")
    print(f"Requirements Met: {'✅' if result.meets_requirements else '❌'}")
    
    print(f"\n📋 Test Gaps Identified: {len(result.test_gaps)}")
    for gap in result.test_gaps[:5]:  # Show top 5
        print(f"  • {Path(gap.file_path).name}::{gap.function_name} ({gap.priority} priority)")
    
    if result.suggestions:
        print("\n💡 Improvement Suggestions:")
        for suggestion in result.suggestions[:5]:  # Show top 5
            print(f"  • {suggestion}")
    
    # Enforce quality gates
    passed, failures = analyzer.enforce_quality_gates(result, args.fail_build)
    
    if not passed and args.fail_build:
        exit(1)


if __name__ == "__main__":
    main()