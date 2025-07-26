"""
Test to verify Python version consistency across configuration files.
"""
import sys
import unittest
from pathlib import Path

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


class TestPythonVersionConsistency(unittest.TestCase):
    """Test Python version consistency across project configurations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.repo_root = Path(__file__).parent.parent
        self.workflow_file = self.repo_root / ".github" / "workflows" / "main.yml"
        self.pyproject_file = self.repo_root / "pyproject.toml"
        self.expected_python_version = "3.12"
    
    def test_current_python_version_is_expected(self):
        """Test that current Python interpreter is the expected version."""
        current_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.assertEqual(
            current_version, 
            self.expected_python_version,
            f"Current Python version {current_version} does not match expected {self.expected_python_version}"
        )
    
    def test_github_workflow_python_version(self):
        """Test that GitHub Actions workflow uses the correct Python version."""
        if not self.workflow_file.exists():
            self.skipTest("GitHub workflow file not found")
        
        if not HAS_YAML:
            # Fallback to text parsing if yaml not available
            with open(self.workflow_file, 'r') as f:
                content = f.read()
            
            # Check for python-version: '3.8' patterns
            self.assertNotIn(
                "python-version: '3.8'",
                content,
                "Found Python 3.8 in workflow, expected 3.12"
            )
            
            self.assertIn(
                f"python-version: '{self.expected_python_version}'",
                content,
                f"Python {self.expected_python_version} not found in workflow"
            )
            return
        
        with open(self.workflow_file, 'r') as f:
            workflow_content = yaml.safe_load(f)
        
        # Check both test and train_on_pr jobs
        jobs = workflow_content.get('jobs', {})
        
        for job_name in ['test', 'train_on_pr']:
            if job_name in jobs:
                steps = jobs[job_name].get('steps', [])
                python_setup_step = None
                
                for step in steps:
                    if step.get('name') == 'Set up Python':
                        python_setup_step = step
                        break
                
                self.assertIsNotNone(
                    python_setup_step, 
                    f"Python setup step not found in {job_name} job"
                )
                
                python_version = python_setup_step.get('with', {}).get('python-version')
                self.assertEqual(
                    python_version, 
                    self.expected_python_version,
                    f"Python version in {job_name} job is {python_version}, expected {self.expected_python_version}"
                )
    
    def test_pyproject_toml_python_version(self):
        """Test that pyproject.toml specifies the correct Python version."""
        if not self.pyproject_file.exists():
            self.skipTest("pyproject.toml file not found")
        
        # Read pyproject.toml content
        with open(self.pyproject_file, 'r') as f:
            content = f.read()
        
        # Simple check for mypy python_version
        self.assertIn(
            f"python_version = '{self.expected_python_version}'",
            content,
            f"pyproject.toml does not specify Python {self.expected_python_version} for mypy"
        )


if __name__ == '__main__':
    unittest.main()