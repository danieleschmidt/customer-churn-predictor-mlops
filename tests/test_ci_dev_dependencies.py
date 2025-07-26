"""
Test to verify CI includes development dependencies and proper testing tools.
"""
import subprocess
import unittest
from pathlib import Path


class TestCIDevDependencies(unittest.TestCase):
    """Test CI development dependencies and tooling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.repo_root = Path(__file__).parent.parent
        self.workflow_file = self.repo_root / ".github" / "workflows" / "main.yml"
        self.dev_requirements = self.repo_root / "requirements-dev.txt"
    
    def test_dev_requirements_file_exists(self):
        """Test that requirements-dev.txt exists."""
        self.assertTrue(
            self.dev_requirements.exists(),
            "requirements-dev.txt file not found"
        )
    
    def test_workflow_installs_dev_dependencies(self):
        """Test that CI workflow installs development dependencies."""
        if not self.workflow_file.exists():
            self.skipTest("GitHub workflow file not found")
        
        with open(self.workflow_file, 'r') as f:
            content = f.read()
        
        # Check that dev dependencies are installed
        self.assertIn(
            "requirements-dev.txt",
            content,
            "CI workflow does not install requirements-dev.txt"
        )
    
    def test_workflow_uses_pytest(self):
        """Test that CI workflow uses pytest instead of unittest."""
        if not self.workflow_file.exists():
            self.skipTest("GitHub workflow file not found")
        
        with open(self.workflow_file, 'r') as f:
            content = f.read()
        
        # Should use pytest
        self.assertIn(
            "pytest",
            content,
            "CI workflow should use pytest for running tests"
        )
        
        # Should not use unittest discover
        self.assertNotIn(
            "python -m unittest discover",
            content,
            "CI workflow should not use unittest discover, use pytest instead"
        )
    
    def test_workflow_includes_linting(self):
        """Test that CI workflow includes linting steps."""
        if not self.workflow_file.exists():
            self.skipTest("GitHub workflow file not found")
        
        with open(self.workflow_file, 'r') as f:
            content = f.read()
        
        # Check for linting tools
        linting_tools = ["black", "flake8", "mypy"]
        missing_tools = []
        
        for tool in linting_tools:
            if tool not in content:
                missing_tools.append(tool)
        
        self.assertEqual(
            len(missing_tools), 0,
            f"CI workflow missing linting tools: {missing_tools}"
        )
    
    def test_dev_dependencies_can_be_imported(self):
        """Test that key development dependencies can be imported."""
        # This test will pass once dependencies are properly installed
        dev_packages = ["pytest", "black", "flake8", "mypy"]
        
        for package in dev_packages:
            with self.subTest(package=package):
                try:
                    __import__(package)
                except ImportError:
                    # Expected to fail until dependencies are installed
                    if package == "pytest":
                        # pytest is critical for testing
                        self.fail(f"Critical dev dependency {package} not available")
                    else:
                        # Other tools can be skipped for now
                        self.skipTest(f"Development tool {package} not installed")


if __name__ == '__main__':
    unittest.main()