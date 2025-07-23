"""Tests for models directory structure and configuration."""

import os
import unittest
from pathlib import Path


class TestModelsDirectoryStructure(unittest.TestCase):
    """Test that the models directory exists and is properly configured."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.repo_root = Path(__file__).parent.parent
        self.models_dir = self.repo_root / "models"
        
    def test_models_directory_exists(self):
        """Test that the models directory exists."""
        self.assertTrue(
            self.models_dir.exists(),
            f"Models directory should exist at {self.models_dir}"
        )
        self.assertTrue(
            self.models_dir.is_dir(),
            f"Models path should be a directory: {self.models_dir}"
        )
    
    def test_models_directory_is_writable(self):
        """Test that the models directory is writable."""
        self.assertTrue(
            os.access(self.models_dir, os.W_OK),
            f"Models directory should be writable: {self.models_dir}"
        )
    
    def test_models_directory_has_gitkeep_for_empty_state(self):
        """Test that models directory has a .gitkeep file to ensure it's tracked."""
        gitkeep_path = self.models_dir / ".gitkeep"
        self.assertTrue(
            gitkeep_path.exists(),
            f".gitkeep file should exist in models directory: {gitkeep_path}"
        )


if __name__ == '__main__':
    unittest.main()