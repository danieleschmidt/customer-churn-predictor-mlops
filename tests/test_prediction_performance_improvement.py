"""Tests for prediction performance improvements - no iterrows() usage."""

import unittest
from pathlib import Path


class TestPredictionPerformanceImprovement(unittest.TestCase):
    """Test that prediction code doesn't use slow iterrows() operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.repo_root = Path(__file__).parent.parent
        self.prediction_script = self.repo_root / "scripts" / "run_prediction.py"
    
    def test_no_iterrows_in_prediction_script(self):
        """Test that run_prediction.py doesn't contain iterrows() calls."""
        with open(self.prediction_script, 'r') as f:
            content = f.read()
        
        # Check that iterrows is not used in the main code flow
        self.assertNotIn(
            ".iterrows()", 
            content,
            "run_prediction.py should not use iterrows() for performance reasons"
        )
    
    def test_no_row_by_row_processing_comments(self):
        """Test that comments about row-by-row processing are removed."""
        with open(self.prediction_script, 'r') as f:
            content = f.read()
        
        # Check that row-by-row processing comments are not present
        self.assertNotIn(
            "row-by-row", 
            content.lower(),
            "Comments about row-by-row processing should be removed"
        )
    
    def test_batch_mode_is_always_used(self):
        """Test that batch_mode logic doesn't include fallback branches."""
        with open(self.prediction_script, 'r') as f:
            content = f.read()
        
        # The fallback logic should be removed
        self.assertNotIn(
            "if not batch_mode:", 
            content,
            "Fallback logic 'if not batch_mode:' should be removed"
        )


if __name__ == '__main__':
    unittest.main()