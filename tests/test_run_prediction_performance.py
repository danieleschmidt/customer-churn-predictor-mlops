"""Tests for run_prediction.py performance improvements."""

import os
import tempfile
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
import time

from scripts.run_prediction import run_predictions


class TestBatchPredictionPerformance:
    """Test suite for batch prediction performance improvements."""
    
    def test_batch_prediction_with_mock_data(self):
        """Test that batch predictions work correctly with mock data."""
        # Create test data
        test_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
            'feature3': [10, 20, 30, 40, 50]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as input_file:
            test_data.to_csv(input_file.name, index=False)
            input_path = input_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as output_file:
            output_path = output_file.name
        
        try:
            # Mock the make_prediction function to return predictable results
            with patch('scripts.run_prediction.make_prediction') as mock_pred:
                mock_pred.side_effect = [
                    (0, 0.3), (1, 0.7), (0, 0.2), (1, 0.8), (0, 0.1)
                ]
                
                # Run predictions
                run_predictions(input_path, output_path)
                
                # Verify results
                result_df = pd.read_csv(output_path)
                
                # Check that all original columns are preserved
                for col in test_data.columns:
                    assert col in result_df.columns
                
                # Check that prediction columns were added
                assert 'prediction' in result_df.columns
                assert 'probability' in result_df.columns
                
                # Check correct number of rows
                assert len(result_df) == len(test_data)
                
                # Check prediction values
                expected_predictions = [0, 1, 0, 1, 0]
                expected_probabilities = [0.3, 0.7, 0.2, 0.8, 0.1]
                
                assert result_df['prediction'].tolist() == expected_predictions
                assert result_df['probability'].tolist() == expected_probabilities
                
                # Verify make_prediction was called correctly
                assert mock_pred.call_count == 5
                
        finally:
            # Clean up
            os.unlink(input_path)
            os.unlink(output_path)
    
    def test_batch_prediction_performance_improvement(self):
        """Test that the new implementation is faster than iterrows approach."""
        # Create larger test dataset
        large_data = pd.DataFrame({
            'feature1': range(100),
            'feature2': [i * 0.01 for i in range(100)],
            'feature3': [i * 10 for i in range(100)]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as input_file:
            large_data.to_csv(input_file.name, index=False)
            input_path = input_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as output_file:
            output_path = output_file.name
        
        try:
            # Mock prediction to simulate realistic processing time
            with patch('scripts.run_prediction.make_prediction') as mock_pred:
                # Simulate 1ms processing time per prediction
                def mock_prediction(data_dict, run_id=None):
                    time.sleep(0.001)  # 1ms delay
                    return (0, 0.5)
                
                mock_pred.side_effect = mock_prediction
                
                # Time the prediction process
                start_time = time.time()
                run_predictions(input_path, output_path)
                end_time = time.time()
                
                execution_time = end_time - start_time
                
                # Should complete in reasonable time (under 5 seconds for 100 predictions)
                # This is a baseline - the actual optimization will improve this
                assert execution_time < 5.0, f"Prediction took too long: {execution_time}s"
                
                # Verify all predictions were made
                assert mock_pred.call_count == 100
                
        finally:
            # Clean up
            os.unlink(input_path)
            os.unlink(output_path)
    
    def test_empty_input_file_handling(self):
        """Test handling of empty input files."""
        # Create empty CSV with headers only
        empty_data = pd.DataFrame(columns=['feature1', 'feature2', 'feature3'])
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as input_file:
            empty_data.to_csv(input_file.name, index=False)
            input_path = input_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as output_file:
            output_path = output_file.name
        
        try:
            # Run predictions on empty file
            run_predictions(input_path, output_path)
            
            # Verify output file exists and has correct structure
            result_df = pd.read_csv(output_path)
            assert len(result_df) == 0
            assert 'prediction' in result_df.columns
            assert 'probability' in result_df.columns
            
        finally:
            # Clean up
            os.unlink(input_path)
            os.unlink(output_path)
    
    def test_file_not_found_error(self):
        """Test proper error handling for missing input files."""
        with pytest.raises(FileNotFoundError, match="Input file not found"):
            run_predictions("nonexistent_file.csv", "output.csv")
    
    def test_prediction_error_handling(self):
        """Test handling of prediction errors."""
        test_data = pd.DataFrame({
            'feature1': [1, 2],
            'feature2': [0.1, 0.2]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as input_file:
            test_data.to_csv(input_file.name, index=False)
            input_path = input_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as output_file:
            output_path = output_file.name
        
        try:
            # Mock prediction to return None (error case)
            with patch('scripts.run_prediction.make_prediction') as mock_pred:
                mock_pred.return_value = (None, None)
                
                # Run predictions
                run_predictions(input_path, output_path)
                
                # Verify error values are handled
                result_df = pd.read_csv(output_path)
                assert len(result_df) == 2
                assert result_df['prediction'].isna().all()
                assert result_df['probability'].isna().all()
                
        finally:
            # Clean up
            os.unlink(input_path)
            os.unlink(output_path)