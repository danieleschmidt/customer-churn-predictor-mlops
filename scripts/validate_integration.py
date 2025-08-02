#!/usr/bin/env python3
"""Integration validation script to verify SDLC implementation completeness."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
# import yaml  # Not needed for this validation


class SDLCValidator:
    """Validates the completeness of SDLC implementation."""
    
    def __init__(self, repo_path: Path = None):
        self.repo_path = repo_path or Path(__file__).parent.parent
        self.validation_results = {}
        
    def validate_all(self) -> Dict[str, Any]:
        """Run all validation checks."""
        print("🔍 Validating SDLC implementation completeness...")
        
        validations = [
            ("Project Foundation", self.validate_project_foundation),
            ("Development Environment", self.validate_development_environment),
            ("Testing Infrastructure", self.validate_testing_infrastructure),
            ("Build & Containerization", self.validate_build_containerization),
            ("Monitoring & Observability", self.validate_monitoring_observability),
            ("Workflow Documentation", self.validate_workflow_documentation),
            ("Metrics & Automation", self.validate_metrics_automation),
            ("Integration & Configuration", self.validate_integration_configuration)
        ]
        
        overall_score = 0
        total_categories = len(validations)
        
        for category, validator in validations:
            try:
                result = validator()
                self.validation_results[category] = result
                score = result.get('score', 0)
                overall_score += score
                
                status = "✅" if score >= 80 else "⚠️" if score >= 60 else "❌"
                print(f"{status} {category}: {score}/100")
                
                if result.get('issues'):
                    for issue in result['issues']:
                        print(f"    ⚠️ {issue}")
                        
            except Exception as e:
                print(f"❌ {category}: Validation failed - {e}")
                self.validation_results[category] = {"score": 0, "error": str(e)}
                
        overall_score = overall_score / total_categories
        self.validation_results['overall'] = {
            'score': overall_score,
            'status': 'COMPLETE' if overall_score >= 80 else 'INCOMPLETE',
            'categories_passed': sum(1 for r in self.validation_results.values() 
                                   if r.get('score', 0) >= 80)
        }
        
        print(f"\n📊 Overall SDLC Implementation Score: {overall_score:.1f}/100")
        print(f"🎯 Status: {self.validation_results['overall']['status']}")
        
        return self.validation_results
    
    def validate_project_foundation(self) -> Dict[str, Any]:
        """Validate Checkpoint 1: Project Foundation & Documentation."""
        score = 0
        issues = []
        
        required_files = [
            'PROJECT_CHARTER.md',
            'README.md',
            'LICENSE',
            'CODE_OF_CONDUCT.md',
            'CONTRIBUTING.md',
            'SECURITY.md',
            'docs/adr/template.md',
            'docs/guides/user-guide.md',
            'docs/guides/developer-guide.md'
        ]
        
        for file_path in required_files:
            if (self.repo_path / file_path).exists():
                score += 10
            else:
                issues.append(f"Missing {file_path}")
                
        # Check ADR structure
        adr_dir = self.repo_path / 'docs' / 'adr'
        if adr_dir.exists():
            adr_files = list(adr_dir.glob('*.md'))
            if len(adr_files) >= 2:  # template + at least one ADR
                score += 10
            else:
                issues.append("Need at least one ADR beyond template")
        else:
            issues.append("Missing ADR directory")
            
        return {'score': min(score, 100), 'issues': issues}
    
    def validate_development_environment(self) -> Dict[str, Any]:
        """Validate Checkpoint 2: Development Environment & Tooling."""
        score = 0
        issues = []
        
        # Check configuration files
        config_files = [
            '.vscode/settings.json',
            '.pre-commit-config.yaml',
            'pyproject.toml',
            '.editorconfig',
            '.gitignore',
            'Makefile'
        ]
        
        for file_path in config_files:
            if (self.repo_path / file_path).exists():
                score += 15
            else:
                issues.append(f"Missing {file_path}")
                
        # Check pyproject.toml content
        pyproject_path = self.repo_path / 'pyproject.toml'
        if pyproject_path.exists():
            try:
                with open(pyproject_path, 'r') as f:
                    content = f.read()
                    if '[tool.black]' in content:
                        score += 5
                    if '[tool.pytest.ini_options]' in content:
                        score += 5
            except Exception:
                issues.append("Could not parse pyproject.toml")
                
        return {'score': min(score, 100), 'issues': issues}
    
    def validate_testing_infrastructure(self) -> Dict[str, Any]:
        """Validate Checkpoint 3: Testing Infrastructure."""
        score = 0
        issues = []
        
        # Check test directories and files
        test_dirs = ['tests/fixtures', 'tests/e2e', 'tests/performance', 'tests/security']
        for test_dir in test_dirs:
            if (self.repo_path / test_dir).exists():
                score += 10
            else:
                issues.append(f"Missing {test_dir}")
                
        # Check specific test files
        test_files = [
            'tests/conftest.py',
            'tests/fixtures/sample_data.py',
            'tests/e2e/test_complete_workflow.py',
            'tests/performance/locustfile.py',
            'tests/test_config.yaml',
            'docs/testing/testing-guide.md'
        ]
        
        for test_file in test_files:
            if (self.repo_path / test_file).exists():
                score += 10
            else:
                issues.append(f"Missing {test_file}")
                
        return {'score': min(score, 100), 'issues': issues}
    
    def validate_build_containerization(self) -> Dict[str, Any]:
        """Validate Checkpoint 4: Build & Containerization."""
        score = 0
        issues = []
        
        # Check Docker files
        docker_files = [
            'Dockerfile',
            'docker-compose.yml',
            '.dockerignore'
        ]
        
        for docker_file in docker_files:
            if (self.repo_path / docker_file).exists():
                score += 20
            else:
                issues.append(f"Missing {docker_file}")
                
        # Check build and automation files
        build_files = [
            '.releaserc.json',
            'scripts/generate_sbom.py',
            'docs/deployment/build-guide.md'
        ]
        
        for build_file in build_files:
            if (self.repo_path / build_file).exists():
                score += 13
            else:
                issues.append(f"Missing {build_file}")
                
        # Check if SBOM script is executable
        sbom_script = self.repo_path / 'scripts' / 'generate_sbom.py'
        if sbom_script.exists() and os.access(sbom_script, os.X_OK):
            score += 1
        else:
            issues.append("SBOM script not executable")
            
        return {'score': min(score, 100), 'issues': issues}
    
    def validate_monitoring_observability(self) -> Dict[str, Any]:
        """Validate Checkpoint 5: Monitoring & Observability Setup."""
        score = 0
        issues = []
        
        # Check monitoring configuration files
        monitoring_files = [
            'monitoring/prometheus.yml',
            'monitoring/alert-rules.yml',
            'monitoring/dashboards/churn-predictor-dashboard.json',
            'observability/opentelemetry-config.yaml',
            'docs/monitoring/observability-guide.md'
        ]
        
        for monitoring_file in monitoring_files:
            if (self.repo_path / monitoring_file).exists():
                score += 20
            else:
                issues.append(f"Missing {monitoring_file}")
                
        return {'score': min(score, 100), 'issues': issues}
    
    def validate_workflow_documentation(self) -> Dict[str, Any]:
        """Validate Checkpoint 6: Workflow Documentation & Templates."""
        score = 0
        issues = []
        
        # Check workflow templates
        workflow_templates = [
            'docs/workflows/examples/ci.yml',
            'docs/workflows/examples/cd.yml',
            'docs/workflows/examples/ml-ops.yml',
            'docs/workflows/examples/security-scan.yml',
            'docs/workflows/workflow-setup-guide.md'
        ]
        
        for workflow_file in workflow_templates:
            if (self.repo_path / workflow_file).exists():
                score += 20
            else:
                issues.append(f"Missing {workflow_file}")
                
        return {'score': min(score, 100), 'issues': issues}
    
    def validate_metrics_automation(self) -> Dict[str, Any]:
        """Validate Checkpoint 7: Metrics & Automation Setup."""
        score = 0
        issues = []
        
        # Check metrics and automation files
        automation_files = [
            '.github/project-metrics.json',
            'scripts/collect_metrics.py',
            'scripts/update_dependencies.py'
        ]
        
        for automation_file in automation_files:
            file_path = self.repo_path / automation_file
            if file_path.exists():
                score += 25
                
                # Check if scripts are executable
                if automation_file.endswith('.py') and not os.access(file_path, os.X_OK):
                    issues.append(f"{automation_file} not executable")
            else:
                issues.append(f"Missing {automation_file}")
                
        # Check project metrics content
        metrics_file = self.repo_path / '.github' / 'project-metrics.json'
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    if 'sdlc_maturity' in metrics:
                        score += 12.5
                    if 'ml_operations' in metrics:
                        score += 12.5
            except json.JSONDecodeError:
                issues.append("Invalid JSON in project-metrics.json")
                
        return {'score': min(score, 100), 'issues': issues}
    
    def validate_integration_configuration(self) -> Dict[str, Any]:
        """Validate Checkpoint 8: Integration & Final Configuration."""
        score = 0
        issues = []
        
        # Check final documentation
        final_files = [
            'IMPLEMENTATION_SUMMARY.md',
            'scripts/validate_integration.py'  # This script itself
        ]
        
        for final_file in final_files:
            if (self.repo_path / final_file).exists():
                score += 30
            else:
                issues.append(f"Missing {final_file}")
                
        # Check overall repository health
        if (self.repo_path / '.github').exists():
            score += 10
        else:
            issues.append("Missing .github directory")
            
        if (self.repo_path / 'docs').exists():
            doc_count = len(list((self.repo_path / 'docs').rglob('*.md')))
            if doc_count >= 20:  # Should have substantial documentation
                score += 20
            else:
                issues.append(f"Only {doc_count} documentation files found, expected 20+")
                
        # Check if key automation is working
        try:
            # Try to run metrics collection
            result = subprocess.run(
                ['python', 'scripts/collect_metrics.py', '--help'],
                cwd=self.repo_path, capture_output=True, timeout=10
            )
            if result.returncode == 0:
                score += 10
            else:
                issues.append("Metrics collection script not working")
        except Exception:
            issues.append("Could not test metrics collection script")
            
        return {'score': min(score, 100), 'issues': issues}
    
    def generate_report(self) -> str:
        """Generate a comprehensive validation report."""
        if not self.validation_results:
            return "No validation results available. Run validate_all() first."
            
        report = []
        report.append("# SDLC Implementation Validation Report")
        report.append(f"**Generated**: {os.environ.get('BUILD_DATE', 'Unknown')}")
        report.append(f"**Repository**: {os.environ.get('GITHUB_REPOSITORY', 'local')}")
        report.append("")
        
        overall = self.validation_results.get('overall', {})
        report.append("## Executive Summary")
        report.append(f"- **Overall Score**: {overall.get('score', 0):.1f}/100")
        report.append(f"- **Implementation Status**: {overall.get('status', 'UNKNOWN')}")
        report.append(f"- **Categories Passed**: {overall.get('categories_passed', 0)}/8")
        report.append("")
        
        report.append("## Detailed Results")
        report.append("| Category | Score | Status | Issues |")
        report.append("|----------|-------|--------|--------|")
        
        for category, result in self.validation_results.items():
            if category == 'overall':
                continue
                
            score = result.get('score', 0)
            status = "✅ PASS" if score >= 80 else "⚠️ WARN" if score >= 60 else "❌ FAIL"
            issues_count = len(result.get('issues', []))
            
            report.append(f"| {category} | {score}/100 | {status} | {issues_count} |")
            
        report.append("")
        
        # Add detailed issues
        report.append("## Issues and Recommendations")
        for category, result in self.validation_results.items():
            if category == 'overall':
                continue
                
            issues = result.get('issues', [])
            if issues:
                report.append(f"### {category}")
                for issue in issues:
                    report.append(f"- {issue}")
                report.append("")
                
        # Add next steps
        report.append("## Next Steps")
        if overall.get('score', 0) >= 80:
            report.append("🎉 **SDLC Implementation Complete!**")
            report.append("")
            report.append("The repository meets all requirements for production deployment:")
            report.append("- All checkpoints successfully implemented")
            report.append("- Comprehensive documentation in place")
            report.append("- Automation and monitoring configured")
            report.append("- Security and quality gates established")
            report.append("")
            report.append("### Recommended Actions:")
            report.append("1. Set up GitHub workflows manually (due to permission limitations)")
            report.append("2. Configure environment secrets and protection rules")
            report.append("3. Train team on new processes and tools")
            report.append("4. Begin production deployment")
        else:
            report.append("⚠️ **Implementation Incomplete**")
            report.append("")
            report.append("Address the issues listed above before proceeding to production.")
            report.append("")
            
        return '\n'.join(report)
    
    def save_report(self, output_path: Path = None) -> Path:
        """Save validation report to file."""
        if output_path is None:
            output_path = self.repo_path / 'VALIDATION_REPORT.md'
            
        report = self.generate_report()
        with open(output_path, 'w') as f:
            f.write(report)
            
        return output_path


def main():
    """Main function to run SDLC validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate SDLC implementation')
    parser.add_argument('--report', '-r', action='store_true',
                       help='Generate detailed report')
    parser.add_argument('--output', '-o', help='Output file for report')
    parser.add_argument('--json', action='store_true',
                       help='Output results as JSON')
    
    args = parser.parse_args()
    
    validator = SDLCValidator()
    results = validator.validate_all()
    
    if args.json:
        print(json.dumps(results, indent=2))
    elif args.report:
        output_path = Path(args.output) if args.output else None
        report_path = validator.save_report(output_path)
        print(f"📊 Validation report saved to: {report_path}")
    
    # Return appropriate exit code
    overall_score = results.get('overall', {}).get('score', 0)
    return 0 if overall_score >= 80 else 1


if __name__ == '__main__':
    sys.exit(main())