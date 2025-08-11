#!/usr/bin/env python3
"""
Quality Dashboard for Comprehensive Testing Framework.

This dashboard provides real-time monitoring and visualization of:
- Test execution progress and results
- Code coverage analysis and trends
- Performance benchmarks and regression detection
- Security vulnerability status
- Quality gate compliance
- System health and resource utilization
- Deployment approval status and recommendations
"""

import os
import json
import time
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import argparse
import threading
import logging

# Web framework for dashboard
try:
    from flask import Flask, render_template, jsonify, request, send_from_directory
    from flask_socketio import SocketIO, emit
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

# Data visualization
try:
    import plotly.graph_objs as go
    import plotly.utils
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# System monitoring
import psutil
import sqlite3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QualityMetricsCollector:
    """Collects and aggregates quality metrics from various sources."""
    
    def __init__(self):
        self.reports_dir = Path("test_reports")
        self.metrics_db = Path("quality_metrics.db")
        self._init_database()
    
    def _init_database(self):
        """Initialize metrics database."""
        conn = sqlite3.connect(self.metrics_db)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                execution_id TEXT,
                timestamp TEXT,
                duration REAL,
                overall_passed BOOLEAN,
                overall_coverage REAL,
                quality_score REAL,
                deployment_approved BOOLEAN,
                suite_results TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quality_trends (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                metric_name TEXT,
                metric_value REAL,
                metric_type TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                cpu_percent REAL,
                memory_percent REAL,
                disk_percent REAL,
                network_io_bytes REAL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def collect_latest_metrics(self) -> Dict[str, Any]:
        """Collect latest quality metrics from reports."""
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'test_execution': self._get_latest_test_execution(),
            'coverage_data': self._get_coverage_data(),
            'performance_data': self._get_performance_data(),
            'security_data': self._get_security_data(),
            'quality_gates_data': self._get_quality_gates_data(),
            'system_health': self._get_system_health(),
            'trends': self._get_trends()
        }
        
        return metrics
    
    def _get_latest_test_execution(self) -> Dict[str, Any]:
        """Get latest test execution results."""
        try:
            # Find latest comprehensive report
            report_files = list(self.reports_dir.glob("comprehensive_report_*.json"))
            
            if not report_files:
                return {'status': 'no_reports', 'message': 'No test reports found'}
            
            latest_report = max(report_files, key=os.path.getmtime)
            
            with open(latest_report) as f:
                report_data = json.load(f)
            
            # Extract key metrics
            return {
                'status': 'success',
                'execution_id': report_data.get('execution_id'),
                'overall_passed': report_data.get('overall_passed', False),
                'total_duration': report_data.get('total_duration', 0),
                'overall_coverage': report_data.get('overall_coverage', 0),
                'quality_score': report_data.get('overall_quality_score', 0),
                'deployment_approved': report_data.get('deployment_approved', False),
                'suite_count': len(report_data.get('suite_results', [])),
                'recommendations': report_data.get('recommendations', []),
                'last_updated': datetime.fromtimestamp(os.path.getmtime(latest_report)).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error collecting test execution metrics: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _get_coverage_data(self) -> Dict[str, Any]:
        """Get code coverage data."""
        try:
            coverage_files = [
                self.reports_dir / "unit_coverage.json",
                self.reports_dir / "coverage.json"
            ]
            
            for coverage_file in coverage_files:
                if coverage_file.exists():
                    with open(coverage_file) as f:
                        coverage_data = json.load(f)
                    
                    totals = coverage_data.get('totals', {})
                    return {
                        'status': 'success',
                        'line_coverage': totals.get('percent_covered', 0),
                        'branch_coverage': totals.get('percent_covered_display', '0%').replace('%', ''),
                        'total_lines': totals.get('num_statements', 0),
                        'covered_lines': totals.get('covered_lines', 0),
                        'missing_lines': totals.get('missing_lines', 0),
                        'files_covered': len(coverage_data.get('files', {}))
                    }
            
            return {'status': 'no_data', 'message': 'No coverage data found'}
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _get_performance_data(self) -> Dict[str, Any]:
        """Get performance test data."""
        try:
            perf_files = [
                self.reports_dir / "benchmark_report.json",
                self.reports_dir / "performance_report.json"
            ]
            
            for perf_file in perf_files:
                if perf_file.exists():
                    with open(perf_file) as f:
                        perf_data = json.load(f)
                    
                    if 'benchmarks' in perf_data:
                        # Benchmark data
                        benchmarks = perf_data['benchmarks']
                        if benchmarks:
                            avg_time = sum(b.get('stats', {}).get('mean', 0) for b in benchmarks) / len(benchmarks)
                            return {
                                'status': 'success',
                                'benchmark_count': len(benchmarks),
                                'avg_execution_time': avg_time,
                                'fastest_benchmark': min(b.get('stats', {}).get('mean', 0) for b in benchmarks),
                                'slowest_benchmark': max(b.get('stats', {}).get('mean', 0) for b in benchmarks)
                            }
                    else:
                        # Test report data
                        summary = perf_data.get('summary', {})
                        return {
                            'status': 'success',
                            'tests_run': summary.get('total', 0),
                            'tests_passed': summary.get('passed', 0),
                            'duration': perf_data.get('duration', 0)
                        }
            
            return {'status': 'no_data', 'message': 'No performance data found'}
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _get_security_data(self) -> Dict[str, Any]:
        """Get security test data."""
        try:
            security_file = self.reports_dir / "security_report.json"
            
            if security_file.exists():
                with open(security_file) as f:
                    security_data = json.load(f)
                
                summary = security_data.get('summary', {})
                return {
                    'status': 'success',
                    'security_tests_run': summary.get('total', 0),
                    'security_tests_passed': summary.get('passed', 0),
                    'security_failures': summary.get('failed', 0),
                    'vulnerability_count': 0,  # Would be populated by actual security scans
                    'last_scan': datetime.now().isoformat()
                }
            
            return {'status': 'no_data', 'message': 'No security data found'}
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _get_quality_gates_data(self) -> Dict[str, Any]:
        """Get quality gates data."""
        try:
            qg_file = self.reports_dir / "quality_gates_report.json"
            
            if qg_file.exists():
                with open(qg_file) as f:
                    qg_data = json.load(f)
                
                return {
                    'status': 'success',
                    'overall_passed': qg_data.get('overall_passed', False),
                    'overall_score': qg_data.get('overall_score', 0),
                    'blocking_failures': qg_data.get('blocking_failures', 0),
                    'warnings': qg_data.get('warnings', 0),
                    'gates_run': qg_data.get('gate_results', 0)
                }
            
            return {'status': 'no_data', 'message': 'No quality gates data found'}
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _get_system_health(self) -> Dict[str, Any]:
        """Get current system health metrics."""
        try:
            return {
                'status': 'success',
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0],
                'uptime': time.time() - psutil.boot_time(),
                'process_count': len(psutil.pids()),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _get_trends(self, days: int = 7) -> Dict[str, Any]:
        """Get quality trends over time."""
        try:
            conn = sqlite3.connect(self.metrics_db)
            cursor = conn.cursor()
            
            # Get recent test executions
            cursor.execute("""
                SELECT timestamp, overall_coverage, quality_score, deployment_approved
                FROM test_executions 
                WHERE timestamp >= datetime('now', '-{} days')
                ORDER BY timestamp
            """.format(days))
            
            executions = cursor.fetchall()
            conn.close()
            
            if not executions:
                return {'status': 'no_data', 'message': 'No historical data available'}
            
            # Process trend data
            timestamps = [row[0] for row in executions]
            coverages = [row[1] for row in executions]
            quality_scores = [row[2] for row in executions]
            deployment_approvals = [row[3] for row in executions]
            
            return {
                'status': 'success',
                'data_points': len(executions),
                'coverage_trend': {
                    'current': coverages[-1] if coverages else 0,
                    'trend': 'improving' if len(coverages) > 1 and coverages[-1] > coverages[0] else 'declining',
                    'data': list(zip(timestamps, coverages))
                },
                'quality_trend': {
                    'current': quality_scores[-1] if quality_scores else 0,
                    'trend': 'improving' if len(quality_scores) > 1 and quality_scores[-1] > quality_scores[0] else 'declining',
                    'data': list(zip(timestamps, quality_scores))
                },
                'deployment_success_rate': sum(deployment_approvals) / len(deployment_approvals) * 100 if deployment_approvals else 0
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def store_metrics(self, metrics: Dict[str, Any]):
        """Store metrics in database for trend analysis."""
        try:
            conn = sqlite3.connect(self.metrics_db)
            cursor = conn.cursor()
            
            # Store test execution data
            test_data = metrics.get('test_execution', {})
            if test_data.get('status') == 'success':
                cursor.execute("""
                    INSERT INTO test_executions 
                    (execution_id, timestamp, duration, overall_passed, overall_coverage, 
                     quality_score, deployment_approved, suite_results)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    test_data.get('execution_id'),
                    datetime.now().isoformat(),
                    test_data.get('total_duration', 0),
                    test_data.get('overall_passed', False),
                    test_data.get('overall_coverage', 0),
                    test_data.get('quality_score', 0),
                    test_data.get('deployment_approved', False),
                    json.dumps(test_data)
                ))
            
            # Store system metrics
            system_data = metrics.get('system_health', {})
            if system_data.get('status') == 'success':
                cursor.execute("""
                    INSERT INTO system_metrics 
                    (timestamp, cpu_percent, memory_percent, disk_percent, network_io_bytes)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    system_data.get('cpu_percent', 0),
                    system_data.get('memory_percent', 0),
                    system_data.get('disk_percent', 0),
                    0  # Network I/O placeholder
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing metrics: {e}")


class QualityDashboard:
    """Flask-based quality dashboard."""
    
    def __init__(self):
        if not FLASK_AVAILABLE:
            raise ImportError("Flask is required for the dashboard. Install with: pip install flask flask-socketio")
        
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'quality-dashboard-secret'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        self.metrics_collector = QualityMetricsCollector()
        self.setup_routes()
        
        # Start background metrics collection
        self.metrics_thread = None
        self.running = False
    
    def setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            """Main dashboard page."""
            return self.render_dashboard()
        
        @self.app.route('/api/metrics')
        def get_metrics():
            """API endpoint for current metrics."""
            metrics = self.metrics_collector.collect_latest_metrics()
            return jsonify(metrics)
        
        @self.app.route('/api/trends/<int:days>')
        def get_trends(days):
            """API endpoint for trend data."""
            trends = self.metrics_collector._get_trends(days)
            return jsonify(trends)
        
        @self.app.route('/api/health')
        def health_check():
            """Health check endpoint."""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'uptime': time.time()
            })
        
        @self.app.route('/reports/<path:filename>')
        def serve_reports(filename):
            """Serve test report files."""
            return send_from_directory('test_reports', filename)
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection."""
            print('Dashboard client connected')
            emit('status', {'msg': 'Connected to quality dashboard'})
        
        @self.socketio.on('request_metrics')
        def handle_metrics_request():
            """Handle real-time metrics request."""
            metrics = self.metrics_collector.collect_latest_metrics()
            emit('metrics_update', metrics)
    
    def render_dashboard(self):
        """Render the main dashboard HTML."""
        
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quality Dashboard</title>
    <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background-color: #f5f6fa; 
        }
        .container { max-width: 1400px; margin: 0 auto; }
        .header { 
            text-align: center; 
            margin-bottom: 30px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; 
            padding: 30px; 
            border-radius: 12px; 
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .metrics-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
            gap: 20px; 
            margin-bottom: 30px; 
        }
        .metric-card { 
            background: white; 
            padding: 25px; 
            border-radius: 12px; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
            border: 1px solid #e1e8ed;
        }
        .metric-title { 
            font-size: 18px; 
            font-weight: 600; 
            margin-bottom: 15px; 
            color: #2c3e50;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .metric-value { 
            font-size: 32px; 
            font-weight: bold; 
            margin-bottom: 10px; 
        }
        .metric-subtitle { 
            color: #7f8c8d; 
            font-size: 14px; 
        }
        .status-passed { color: #27ae60; }
        .status-failed { color: #e74c3c; }
        .status-warning { color: #f39c12; }
        .progress-bar { 
            width: 100%; 
            height: 20px; 
            background-color: #ecf0f1; 
            border-radius: 10px; 
            overflow: hidden; 
            margin: 10px 0;
        }
        .progress-fill { 
            height: 100%; 
            border-radius: 10px; 
            transition: width 0.3s ease; 
        }
        .progress-high { background: linear-gradient(90deg, #27ae60, #2ecc71); }
        .progress-medium { background: linear-gradient(90deg, #f39c12, #e67e22); }
        .progress-low { background: linear-gradient(90deg, #e74c3c, #c0392b); }
        .chart-container { 
            background: white; 
            padding: 25px; 
            border-radius: 12px; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
            margin-bottom: 20px;
        }
        .recommendations { 
            background: #fff3cd; 
            border: 1px solid #ffeaa7; 
            border-radius: 12px; 
            padding: 20px; 
            margin-bottom: 20px;
        }
        .recommendation-item { 
            margin: 10px 0; 
            padding: 10px; 
            background: white; 
            border-radius: 8px; 
        }
        .system-status { 
            display: flex; 
            justify-content: space-around; 
            flex-wrap: wrap; 
            gap: 15px; 
        }
        .status-item { 
            text-align: center; 
            min-width: 120px; 
        }
        .status-value { 
            font-size: 24px; 
            font-weight: bold; 
            margin-bottom: 5px; 
        }
        .connection-status {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 15px;
            border-radius: 20px;
            color: white;
            font-weight: bold;
            font-size: 12px;
        }
        .connected { background-color: #27ae60; }
        .disconnected { background-color: #e74c3c; }
        .loading { 
            display: inline-block; 
            width: 20px; 
            height: 20px; 
            border: 3px solid #f3f3f3; 
            border-top: 3px solid #3498db; 
            border-radius: 50%; 
            animation: spin 1s linear infinite; 
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .refresh-btn {
            background: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            margin-left: 10px;
        }
        .refresh-btn:hover { background: #2980b9; }
    </style>
</head>
<body>
    <div class="connection-status" id="connectionStatus">Connecting...</div>
    
    <div class="container">
        <div class="header">
            <h1>🎯 Quality Dashboard</h1>
            <p>Real-time monitoring of testing framework and quality gates</p>
            <p>Last updated: <span id="lastUpdated">Loading...</span>
            <button class="refresh-btn" onclick="refreshMetrics()">🔄 Refresh</button>
            </p>
        </div>

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-title">📊 Test Execution Status</div>
                <div class="metric-value" id="testStatus">
                    <span class="loading"></span>
                </div>
                <div class="metric-subtitle" id="testDetails">Loading test data...</div>
            </div>

            <div class="metric-card">
                <div class="metric-title">📈 Code Coverage</div>
                <div class="metric-value" id="coverageValue">
                    <span class="loading"></span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" id="coverageProgress"></div>
                </div>
                <div class="metric-subtitle" id="coverageDetails">Loading coverage data...</div>
            </div>

            <div class="metric-card">
                <div class="metric-title">🎯 Quality Score</div>
                <div class="metric-value" id="qualityValue">
                    <span class="loading"></span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" id="qualityProgress"></div>
                </div>
                <div class="metric-subtitle" id="qualityDetails">Loading quality data...</div>
            </div>

            <div class="metric-card">
                <div class="metric-title">🚀 Deployment Status</div>
                <div class="metric-value" id="deploymentStatus">
                    <span class="loading"></span>
                </div>
                <div class="metric-subtitle" id="deploymentDetails">Loading deployment data...</div>
            </div>
        </div>

        <div class="chart-container">
            <h3>📊 System Health</h3>
            <div class="system-status" id="systemHealth">
                <div class="status-item">
                    <div class="status-value" id="cpuValue">--%</div>
                    <div>CPU Usage</div>
                </div>
                <div class="status-item">
                    <div class="status-value" id="memoryValue">--%</div>
                    <div>Memory Usage</div>
                </div>
                <div class="status-item">
                    <div class="status-value" id="diskValue">--%</div>
                    <div>Disk Usage</div>
                </div>
                <div class="status-item">
                    <div class="status-value" id="processValue">--</div>
                    <div>Processes</div>
                </div>
            </div>
        </div>

        <div class="recommendations" id="recommendationsSection" style="display: none;">
            <h3>💡 Recommendations</h3>
            <div id="recommendationsList"></div>
        </div>

        <div class="chart-container">
            <h3>📈 Quality Trends</h3>
            <div id="trendsChart" style="height: 400px;"></div>
        </div>
    </div>

    <script>
        const socket = io();
        let metricsData = {};

        socket.on('connect', function() {
            document.getElementById('connectionStatus').textContent = 'Connected';
            document.getElementById('connectionStatus').className = 'connection-status connected';
            requestMetrics();
        });

        socket.on('disconnect', function() {
            document.getElementById('connectionStatus').textContent = 'Disconnected';
            document.getElementById('connectionStatus').className = 'connection-status disconnected';
        });

        socket.on('metrics_update', function(data) {
            metricsData = data;
            updateDashboard(data);
        });

        function requestMetrics() {
            socket.emit('request_metrics');
        }

        function refreshMetrics() {
            requestMetrics();
        }

        function updateDashboard(data) {
            document.getElementById('lastUpdated').textContent = new Date().toLocaleString();

            // Update test execution status
            const testData = data.test_execution || {};
            if (testData.status === 'success') {
                const passed = testData.overall_passed;
                document.getElementById('testStatus').innerHTML = `
                    <span class="${passed ? 'status-passed' : 'status-failed'}">
                        ${passed ? '✅ PASSED' : '❌ FAILED'}
                    </span>
                `;
                document.getElementById('testDetails').textContent = 
                    `${testData.suite_count} suites • ${testData.total_duration?.toFixed(1)}s`;
            }

            // Update coverage
            const coverageData = data.coverage_data || {};
            if (coverageData.status === 'success') {
                const coverage = coverageData.line_coverage || 0;
                document.getElementById('coverageValue').innerHTML = `${coverage.toFixed(1)}%`;
                
                const progressClass = coverage >= 85 ? 'progress-high' : coverage >= 70 ? 'progress-medium' : 'progress-low';
                const progressElement = document.getElementById('coverageProgress');
                progressElement.style.width = `${Math.min(coverage, 100)}%`;
                progressElement.className = `progress-fill ${progressClass}`;
                
                document.getElementById('coverageDetails').textContent = 
                    `${coverageData.covered_lines}/${coverageData.total_lines} lines covered`;
            }

            // Update quality score
            const testExec = data.test_execution || {};
            if (testExec.status === 'success') {
                const quality = testExec.quality_score || 0;
                document.getElementById('qualityValue').textContent = `${quality.toFixed(1)}`;
                
                const progressClass = quality >= 80 ? 'progress-high' : quality >= 60 ? 'progress-medium' : 'progress-low';
                const progressElement = document.getElementById('qualityProgress');
                progressElement.style.width = `${Math.min(quality, 100)}%`;
                progressElement.className = `progress-fill ${progressClass}`;
                
                document.getElementById('qualityDetails').textContent = `Quality threshold: 80.0`;
            }

            // Update deployment status
            if (testExec.status === 'success') {
                const approved = testExec.deployment_approved;
                document.getElementById('deploymentStatus').innerHTML = `
                    <span class="${approved ? 'status-passed' : 'status-failed'}">
                        ${approved ? '✅ APPROVED' : '❌ BLOCKED'}
                    </span>
                `;
                document.getElementById('deploymentDetails').textContent = 
                    approved ? 'Ready for deployment' : 'Fix issues before deployment';
            }

            // Update system health
            const systemData = data.system_health || {};
            if (systemData.status === 'success') {
                document.getElementById('cpuValue').textContent = `${systemData.cpu_percent?.toFixed(1)}%`;
                document.getElementById('memoryValue').textContent = `${systemData.memory_percent?.toFixed(1)}%`;
                document.getElementById('diskValue').textContent = `${systemData.disk_percent?.toFixed(1)}%`;
                document.getElementById('processValue').textContent = systemData.process_count || '--';
            }

            // Update recommendations
            const recommendations = testExec.recommendations || [];
            if (recommendations.length > 0) {
                document.getElementById('recommendationsSection').style.display = 'block';
                document.getElementById('recommendationsList').innerHTML = recommendations
                    .map(rec => `<div class="recommendation-item">• ${rec}</div>`)
                    .join('');
            } else {
                document.getElementById('recommendationsSection').style.display = 'none';
            }

            // Update trends chart
            updateTrendsChart(data.trends);
        }

        function updateTrendsChart(trendsData) {
            if (!trendsData || trendsData.status !== 'success') {
                return;
            }

            const coverageTrend = trendsData.coverage_trend?.data || [];
            const qualityTrend = trendsData.quality_trend?.data || [];

            if (coverageTrend.length === 0 && qualityTrend.length === 0) {
                document.getElementById('trendsChart').innerHTML = 
                    '<p style="text-align: center; color: #7f8c8d; margin: 50px 0;">No historical data available</p>';
                return;
            }

            const traces = [];

            if (coverageTrend.length > 0) {
                traces.push({
                    x: coverageTrend.map(d => d[0]),
                    y: coverageTrend.map(d => d[1]),
                    name: 'Coverage %',
                    type: 'scatter',
                    mode: 'lines+markers',
                    line: { color: '#3498db', width: 2 },
                    marker: { size: 6 }
                });
            }

            if (qualityTrend.length > 0) {
                traces.push({
                    x: qualityTrend.map(d => d[0]),
                    y: qualityTrend.map(d => d[1]),
                    name: 'Quality Score',
                    type: 'scatter',
                    mode: 'lines+markers',
                    line: { color: '#e74c3c', width: 2 },
                    marker: { size: 6 },
                    yaxis: 'y2'
                });
            }

            const layout = {
                title: 'Quality Metrics Over Time',
                xaxis: { title: 'Date' },
                yaxis: { 
                    title: 'Coverage %',
                    range: [0, 100]
                },
                yaxis2: {
                    title: 'Quality Score',
                    overlaying: 'y',
                    side: 'right',
                    range: [0, 100]
                },
                margin: { t: 50, r: 80, b: 50, l: 50 },
                showlegend: true,
                legend: { x: 0.02, y: 0.98 }
            };

            Plotly.newPlot('trendsChart', traces, layout, {responsive: true});
        }

        // Auto-refresh every 30 seconds
        setInterval(requestMetrics, 30000);
        
        // Initial load
        requestMetrics();
    </script>
</body>
</html>
"""
        return html_template
    
    def start_background_collection(self):
        """Start background metrics collection."""
        def collect_loop():
            while self.running:
                try:
                    metrics = self.metrics_collector.collect_latest_metrics()
                    self.metrics_collector.store_metrics(metrics)
                    
                    # Emit to connected clients
                    self.socketio.emit('metrics_update', metrics)
                    
                    time.sleep(10)  # Collect every 10 seconds
                except Exception as e:
                    logger.error(f"Error in metrics collection loop: {e}")
                    time.sleep(30)  # Wait longer on error
        
        self.running = True
        self.metrics_thread = threading.Thread(target=collect_loop, daemon=True)
        self.metrics_thread.start()
        logger.info("Background metrics collection started")
    
    def stop_background_collection(self):
        """Stop background metrics collection."""
        self.running = False
        if self.metrics_thread:
            self.metrics_thread.join(timeout=5)
        logger.info("Background metrics collection stopped")
    
    def run(self, host='127.0.0.1', port=8080, debug=False):
        """Run the dashboard server."""
        
        logger.info(f"Starting Quality Dashboard on http://{host}:{port}")
        
        # Start background collection
        self.start_background_collection()
        
        try:
            self.socketio.run(
                self.app, 
                host=host, 
                port=port, 
                debug=debug,
                allow_unsafe_werkzeug=True
            )
        except KeyboardInterrupt:
            logger.info("Dashboard stopped by user")
        finally:
            self.stop_background_collection()


def main():
    """Main entry point for the dashboard."""
    parser = argparse.ArgumentParser(description="Quality Dashboard")
    
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8080, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--collect-only', action='store_true', help='Only collect metrics, no web interface')
    
    args = parser.parse_args()
    
    if args.collect_only:
        # Just collect and display metrics
        collector = QualityMetricsCollector()
        metrics = collector.collect_latest_metrics()
        print(json.dumps(metrics, indent=2))
        return 0
    
    if not FLASK_AVAILABLE:
        print("❌ Flask is not available. Install with: pip install flask flask-socketio")
        return 1
    
    try:
        dashboard = QualityDashboard()
        dashboard.run(host=args.host, port=args.port, debug=args.debug)
        return 0
    except Exception as e:
        logger.error(f"Dashboard failed to start: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())