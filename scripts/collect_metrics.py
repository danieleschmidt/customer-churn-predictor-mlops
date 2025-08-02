#!/usr/bin/env python3
"""Automated metrics collection script for project health monitoring."""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
import requests


class MetricsCollector:
    """Collects various project metrics from multiple sources."""
    
    def __init__(self, github_token: Optional[str] = None):
        self.github_token = github_token or os.getenv('GITHUB_TOKEN')
        self.repo_path = Path(__file__).parent.parent
        self.github_repo = os.getenv('GITHUB_REPOSITORY', 'danieleschmidt/customer-churn-predictor-mlops')
        
    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all available metrics."""
        print("🔍 Collecting project metrics...")
        
        metrics = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'collection_version': '1.0.0'
        }
        
        try:
            metrics['repository'] = self.collect_repository_metrics()
            print("✅ Repository metrics collected")
        except Exception as e:
            print(f"❌ Failed to collect repository metrics: {e}")
            
        try:
            metrics['code_quality'] = self.collect_code_quality_metrics()
            print("✅ Code quality metrics collected")
        except Exception as e:
            print(f"❌ Failed to collect code quality metrics: {e}")
            
        try:
            metrics['testing'] = self.collect_testing_metrics()
            print("✅ Testing metrics collected")
        except Exception as e:
            print(f"❌ Failed to collect testing metrics: {e}")
            
        try:
            metrics['security'] = self.collect_security_metrics()
            print("✅ Security metrics collected")
        except Exception as e:
            print(f"❌ Failed to collect security metrics: {e}")
            
        try:
            metrics['docker'] = self.collect_docker_metrics()
            print("✅ Docker metrics collected")
        except Exception as e:
            print(f"❌ Failed to collect Docker metrics: {e}")
            
        try:
            metrics['dependencies'] = self.collect_dependency_metrics()
            print("✅ Dependency metrics collected")
        except Exception as e:
            print(f"❌ Failed to collect dependency metrics: {e}")
            
        if self.github_token:
            try:
                metrics['github'] = self.collect_github_metrics()
                print("✅ GitHub metrics collected")
            except Exception as e:
                print(f"❌ Failed to collect GitHub metrics: {e}")
        else:
            print("⚠️ GitHub token not available, skipping GitHub metrics")
            
        return metrics
    
    def collect_repository_metrics(self) -> Dict[str, Any]:
        """Collect basic repository metrics."""
        metrics = {}
        
        # Count lines of code
        try:
            result = subprocess.run(
                ['find', str(self.repo_path), '-name', '*.py', '-not', '-path', '*/.*', '-exec', 'wc', '-l', '{}', '+'],
                capture_output=True, text=True, check=True
            )
            lines = sum(int(line.split()[0]) for line in result.stdout.strip().split('\n') if line.strip())
            metrics['lines_of_code'] = lines
        except subprocess.CalledProcessError:
            metrics['lines_of_code'] = 0
            
        # Count files by type
        file_counts = {}
        for pattern, file_type in [
            ('*.py', 'python_files'),
            ('*.yml', 'yaml_files'),
            ('*.yaml', 'yaml_files'),
            ('*.md', 'markdown_files'),
            ('*.json', 'json_files'),
            ('Dockerfile*', 'dockerfiles')
        ]:
            try:
                result = subprocess.run(
                    ['find', str(self.repo_path), '-name', pattern, '-not', '-path', '*/.*'],
                    capture_output=True, text=True, check=True
                )
                count = len([line for line in result.stdout.strip().split('\n') if line.strip()])
                file_counts[file_type] = file_counts.get(file_type, 0) + count
            except subprocess.CalledProcessError:
                file_counts[file_type] = 0
                
        metrics['file_counts'] = file_counts
        
        # Git metrics
        try:
            # Commit count in last 30 days
            since_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            result = subprocess.run(
                ['git', '-C', str(self.repo_path), 'rev-list', '--count', '--since', since_date, 'HEAD'],
                capture_output=True, text=True, check=True
            )
            metrics['commits_last_30_days'] = int(result.stdout.strip())
            
            # Contributors count
            result = subprocess.run(
                ['git', '-C', str(self.repo_path), 'shortlog', '-sn', '--since', since_date],
                capture_output=True, text=True, check=True
            )
            metrics['contributors_last_30_days'] = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
            
        except subprocess.CalledProcessError:
            metrics['commits_last_30_days'] = 0
            metrics['contributors_last_30_days'] = 0
            
        return metrics
    
    def collect_code_quality_metrics(self) -> Dict[str, Any]:
        """Collect code quality metrics."""
        metrics = {}
        
        # Run flake8 for code quality
        try:
            result = subprocess.run(
                ['flake8', '--count', '--select=E9,F63,F7,F82', str(self.repo_path / 'src')],
                capture_output=True, text=True, cwd=self.repo_path
            )
            metrics['syntax_errors'] = int(result.stdout.strip()) if result.stdout.strip() else 0
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            metrics['syntax_errors'] = 0
            
        # Count TODO/FIXME comments
        try:
            result = subprocess.run(
                ['grep', '-r', '-i', '--include=*.py', 'TODO\\|FIXME\\|XXX', str(self.repo_path / 'src')],
                capture_output=True, text=True
            )
            metrics['todo_count'] = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
        except subprocess.CalledProcessError:
            metrics['todo_count'] = 0
            
        # Check if code formatting tools are configured
        metrics['has_black_config'] = (self.repo_path / 'pyproject.toml').exists()
        metrics['has_isort_config'] = (self.repo_path / 'pyproject.toml').exists()
        metrics['has_flake8_config'] = (self.repo_path / 'pyproject.toml').exists()
        
        return metrics
    
    def collect_testing_metrics(self) -> Dict[str, Any]:
        """Collect testing-related metrics."""
        metrics = {}
        
        # Count test files
        try:
            result = subprocess.run(
                ['find', str(self.repo_path / 'tests'), '-name', 'test_*.py'],
                capture_output=True, text=True, check=True
            )
            metrics['test_file_count'] = len([line for line in result.stdout.strip().split('\n') if line.strip()])
        except subprocess.CalledProcessError:
            metrics['test_file_count'] = 0
            
        # Try to get test coverage if available
        coverage_file = self.repo_path / 'coverage.xml'
        if coverage_file.exists():
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(coverage_file)
                root = tree.getroot()
                coverage_elem = root.find('.//coverage')
                if coverage_elem is not None:
                    metrics['test_coverage'] = float(coverage_elem.get('line-rate', 0)) * 100
            except Exception:
                metrics['test_coverage'] = 0
        else:
            metrics['test_coverage'] = 0
            
        # Check testing configuration
        metrics['has_pytest_config'] = (self.repo_path / 'pyproject.toml').exists()
        metrics['has_conftest'] = (self.repo_path / 'tests' / 'conftest.py').exists()
        
        return metrics
    
    def collect_security_metrics(self) -> Dict[str, Any]:
        """Collect security-related metrics."""
        metrics = {}
        
        # Check for security configuration files
        metrics['has_security_policy'] = (self.repo_path / 'SECURITY.md').exists()
        metrics['has_dependabot'] = (self.repo_path / '.github' / 'dependabot.yml').exists()
        metrics['has_codeowners'] = (self.repo_path / '.github' / 'CODEOWNERS').exists()
        
        # Try to run bandit security scan
        try:
            result = subprocess.run(
                ['bandit', '-r', str(self.repo_path / 'src'), '-f', 'json'],
                capture_output=True, text=True, cwd=self.repo_path
            )
            if result.returncode == 0:
                bandit_data = json.loads(result.stdout)
                metrics['security_issues'] = len(bandit_data.get('results', []))
                metrics['security_confidence_high'] = len([
                    r for r in bandit_data.get('results', []) 
                    if r.get('issue_confidence') == 'HIGH'
                ])
            else:
                metrics['security_issues'] = 0
                metrics['security_confidence_high'] = 0
        except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
            metrics['security_issues'] = 0
            metrics['security_confidence_high'] = 0
            
        return metrics
    
    def collect_docker_metrics(self) -> Dict[str, Any]:
        """Collect Docker-related metrics."""
        metrics = {}
        
        # Check for Docker files
        metrics['has_dockerfile'] = (self.repo_path / 'Dockerfile').exists()
        metrics['has_docker_compose'] = (self.repo_path / 'docker-compose.yml').exists()
        metrics['has_dockerignore'] = (self.repo_path / '.dockerignore').exists()
        
        # Count Docker-related files
        docker_files = list(self.repo_path.glob('Dockerfile*')) + list(self.repo_path.glob('docker-compose*.yml'))
        metrics['docker_file_count'] = len(docker_files)
        
        # Try to get Docker image size if image exists
        try:
            result = subprocess.run(
                ['docker', 'images', '--format', 'table {{.Repository}}:{{.Tag}}\\t{{.Size}}'],
                capture_output=True, text=True
            )
            metrics['has_docker_images'] = 'customer-churn-predictor' in result.stdout
        except subprocess.CalledProcessError:
            metrics['has_docker_images'] = False
            
        return metrics
    
    def collect_dependency_metrics(self) -> Dict[str, Any]:
        """Collect dependency-related metrics."""
        metrics = {}
        
        # Count dependencies
        req_files = ['requirements.txt', 'requirements-dev.txt', 'pyproject.toml']
        total_deps = 0
        
        for req_file in req_files:
            file_path = self.repo_path / req_file
            if file_path.exists():
                if req_file.endswith('.txt'):
                    with open(file_path, 'r') as f:
                        deps = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                        total_deps += len(deps)
                        metrics[f'{req_file.replace(".", "_")}_count'] = len(deps)
                        
        metrics['total_dependencies'] = total_deps
        
        # Check for lock files
        metrics['has_lock_files'] = (
            (self.repo_path / 'requirements.lock').exists() or
            (self.repo_path / 'poetry.lock').exists() or
            (self.repo_path / 'Pipfile.lock').exists()
        )
        
        return metrics
    
    def collect_github_metrics(self) -> Dict[str, Any]:
        """Collect GitHub-specific metrics using API."""
        if not self.github_token:
            return {}
            
        headers = {
            'Authorization': f'token {self.github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        base_url = f'https://api.github.com/repos/{self.github_repo}'
        metrics = {}
        
        try:
            # Repository info
            response = requests.get(base_url, headers=headers, timeout=30)
            if response.status_code == 200:
                repo_data = response.json()
                metrics['stars'] = repo_data.get('stargazers_count', 0)
                metrics['forks'] = repo_data.get('forks_count', 0)
                metrics['open_issues'] = repo_data.get('open_issues_count', 0)
                metrics['size_kb'] = repo_data.get('size', 0)
                metrics['default_branch'] = repo_data.get('default_branch', 'main')
                
            # Pull requests
            pr_response = requests.get(f'{base_url}/pulls?state=all&per_page=100', headers=headers, timeout=30)
            if pr_response.status_code == 200:
                prs = pr_response.json()
                metrics['total_prs'] = len(prs)
                metrics['open_prs'] = len([pr for pr in prs if pr['state'] == 'open'])
                
            # Recent releases
            releases_response = requests.get(f'{base_url}/releases?per_page=10', headers=headers, timeout=30)
            if releases_response.status_code == 200:
                releases = releases_response.json()
                metrics['release_count'] = len(releases)
                if releases:
                    metrics['latest_release'] = releases[0]['tag_name']
                    metrics['latest_release_date'] = releases[0]['published_at']
                    
        except requests.RequestException as e:
            print(f"Warning: Failed to collect some GitHub metrics: {e}")
            
        return metrics
    
    def save_metrics(self, metrics: Dict[str, Any], output_file: Optional[str] = None) -> None:
        """Save metrics to file."""
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'metrics_{timestamp}.json'
            
        output_path = self.repo_path / 'docs' / 'status' / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
            
        print(f"📊 Metrics saved to {output_path}")
        
    def update_project_metrics(self, collected_metrics: Dict[str, Any]) -> None:
        """Update the main project metrics file."""
        project_metrics_path = self.repo_path / '.github' / 'project-metrics.json'
        
        if project_metrics_path.exists():
            with open(project_metrics_path, 'r') as f:
                project_metrics = json.load(f)
        else:
            project_metrics = {}
            
        # Update specific metrics based on collected data
        if 'repository' in collected_metrics:
            repo_metrics = collected_metrics['repository']
            if 'technical_metrics' not in project_metrics:
                project_metrics['technical_metrics'] = {}
                
            project_metrics['technical_metrics'].update({
                'code_lines': repo_metrics.get('lines_of_code', 0),
                'commits_last_30_days': repo_metrics.get('commits_last_30_days', 0),
                'contributors_active': repo_metrics.get('contributors_last_30_days', 0)
            })
            
        if 'testing' in collected_metrics:
            test_metrics = collected_metrics['testing']
            if 'quality_gates' not in project_metrics:
                project_metrics['quality_gates'] = {}
                
            if test_metrics.get('test_coverage', 0) > 0:
                project_metrics['sdlc_maturity']['test_coverage'] = test_metrics['test_coverage']
                
        if 'security' in collected_metrics:
            sec_metrics = collected_metrics['security']
            security_score = 85  # Base score
            if sec_metrics.get('security_issues', 0) == 0:
                security_score += 10
            if sec_metrics.get('has_security_policy', False):
                security_score += 5
            project_metrics['sdlc_maturity']['security_score'] = min(security_score, 100)
            
        # Update timestamp
        project_metrics['metadata'] = project_metrics.get('metadata', {})
        project_metrics['metadata']['last_updated'] = datetime.utcnow().isoformat() + 'Z'
        
        # Save updated metrics
        with open(project_metrics_path, 'w') as f:
            json.dump(project_metrics, f, indent=2, sort_keys=True)
            
        print(f"📈 Updated project metrics in {project_metrics_path}")


def main():
    """Main function to run metrics collection."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect project metrics')
    parser.add_argument('--output', '-o', help='Output file name')
    parser.add_argument('--update-project', action='store_true', 
                       help='Update main project metrics file')
    parser.add_argument('--github-token', help='GitHub token for API access')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        print("🚀 Starting metrics collection...")
        
    collector = MetricsCollector(github_token=args.github_token)
    metrics = collector.collect_all_metrics()
    
    # Save detailed metrics
    collector.save_metrics(metrics, args.output)
    
    # Update project metrics if requested
    if args.update_project:
        collector.update_project_metrics(metrics)
        
    if args.verbose:
        print("✅ Metrics collection completed!")
        print(f"📊 Collected {len(metrics)} metric categories")
        
    return 0


if __name__ == '__main__':
    sys.exit(main())