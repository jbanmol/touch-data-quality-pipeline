#!/usr/bin/env python3
"""
Test Suite for ML-Based Touch Data Cleaning

This test suite validates the ML cleaning pipeline functionality,
ensuring that all original data is preserved while adding valuable metadata.
"""

import unittest
import pandas as pd
import numpy as np
import json
import tempfile
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ML.metadata_enhancer import TouchDataMetadataEnhancer
from ML.feature_engineering import TouchFeatureEngineer
from ML.ml_cleaning_pipeline import MLTouchDataCleaner

class TestTouchFeatureEngineer(unittest.TestCase):
    """Test the feature engineering component."""
    
    def setUp(self):
        """Set up test data."""
        self.feature_engineer = TouchFeatureEngineer()
        
        # Create sample touch data
        self.sample_data = pd.DataFrame({
            'Touchdata_id': [1, 1, 1, 1, 2, 2, 2],
            'event_index': [0, 1, 2, 3, 0, 1, 2],
            'x': [100, 110, 120, 130, 200, 210, 220],
            'y': [100, 105, 110, 115, 200, 205, 210],
            'time': [1000, 1100, 1200, 1300, 1000, 1100, 1200],
            'touchPhase': ['Began', 'Moved', 'Moved', 'Ended', 'Began', 'Moved', 'Ended'],
            'fingerId': [1, 1, 1, 1, 2, 2, 2],
            'color': ['Red', 'Red', 'Red', 'Red', 'Blue', 'Blue', 'Blue'],
            'completionPerc': [0.0, 0.3, 0.6, 1.0, 0.0, 0.5, 1.0],
            'zone': ['Area1', 'Area1', 'Area2', 'Area2', 'Area3', 'Area3', 'Area3']
        })
    
    def test_temporal_features(self):
        """Test temporal feature extraction."""
        result = self.feature_engineer.extract_temporal_features(self.sample_data.copy())
        
        # Check that temporal features are added
        self.assertIn('time_diff', result.columns)
        self.assertIn('cumulative_time', result.columns)
        self.assertIn('sequence_position', result.columns)
        self.assertIn('sequence_length', result.columns)
        
        # Verify time differences are calculated correctly
        seq1_data = result[result['Touchdata_id'] == 1]
        expected_time_diffs = [0, 100, 100, 100]  # First is 0, then differences
        actual_time_diffs = seq1_data['time_diff'].tolist()
        self.assertEqual(actual_time_diffs, expected_time_diffs)
    
    def test_spatial_features(self):
        """Test spatial feature extraction."""
        result = self.feature_engineer.extract_spatial_features(self.sample_data.copy())
        
        # Check that spatial features are added
        self.assertIn('x_diff', result.columns)
        self.assertIn('y_diff', result.columns)
        self.assertIn('distance', result.columns)
        self.assertIn('velocity', result.columns)
        self.assertIn('cumulative_distance', result.columns)
        
        # Verify distance calculations
        seq1_data = result[result['Touchdata_id'] == 1].sort_values('event_index')
        # Distance from (100,100) to (110,105) should be sqrt(100+25) = sqrt(125)
        expected_first_distance = np.sqrt(10**2 + 5**2)
        actual_first_distance = seq1_data.iloc[1]['distance']
        self.assertAlmostEqual(actual_first_distance, expected_first_distance, places=2)
    
    def test_behavioral_features(self):
        """Test behavioral feature extraction."""
        result = self.feature_engineer.extract_behavioral_features(self.sample_data.copy())
        
        # Check that behavioral features are added
        self.assertIn('phase_transition', result.columns)
        self.assertIn('completion_rate', result.columns)
        self.assertIn('zone_change', result.columns)
        self.assertIn('color_change', result.columns)
        
        # Verify phase transitions
        seq1_data = result[result['Touchdata_id'] == 1].sort_values('event_index')
        first_transition = seq1_data.iloc[0]['phase_transition']
        second_transition = seq1_data.iloc[1]['phase_transition']
        
        self.assertEqual(first_transition, 'start_Began')
        self.assertEqual(second_transition, 'Began_to_Moved')
    
    def test_sequence_quality_features(self):
        """Test sequence quality feature extraction."""
        result = self.feature_engineer.extract_sequence_quality_features(self.sample_data.copy())
        
        # Check that quality features are added
        self.assertIn('has_began', result.columns)
        self.assertIn('has_ended', result.columns)
        self.assertIn('sequence_valid_pattern', result.columns)
        
        # Verify sequence completeness indicators
        seq1_data = result[result['Touchdata_id'] == 1]
        self.assertTrue(all(seq1_data['has_began'] == 1))
        self.assertTrue(all(seq1_data['has_ended'] == 1))
        self.assertTrue(all(seq1_data['sequence_valid_pattern'] == 1))
    
    def test_all_features_extraction(self):
        """Test complete feature extraction pipeline."""
        original_cols = set(self.sample_data.columns)
        result = self.feature_engineer.extract_all_features(self.sample_data.copy())
        new_cols = set(result.columns)
        
        # Check that new features were added
        added_features = new_cols - original_cols
        self.assertGreater(len(added_features), 10)  # Should add many features
        
        # Check that feature names are stored
        self.assertGreater(len(self.feature_engineer.feature_names), 10)
        
        # Verify no original data was modified
        for col in original_cols:
            if col in ['x', 'y', 'time', 'touchPhase']:  # Core data columns
                pd.testing.assert_series_equal(
                    self.sample_data[col], 
                    result[col], 
                    check_names=False
                )

class TestTouchDataMetadataEnhancer(unittest.TestCase):
    """Test the metadata enhancement component."""
    
    def setUp(self):
        """Set up test data."""
        self.enhancer = TouchDataMetadataEnhancer()
        
        # Create sample data with different quality levels
        self.sample_data = pd.DataFrame({
            'Touchdata_id': [1, 1, 1, 1, 2, 2, 3, 3, 3],
            'event_index': [0, 1, 2, 3, 0, 1, 0, 1, 2],
            'x': [100, 110, 120, 130, 200, 1000, 300, 310, 320],  # Seq 2 has outlier
            'y': [100, 105, 110, 115, 200, 200, 300, 305, 310],
            'time': [1000, 1100, 1200, 1300, 1000, 1100, 1000, 1100, 1200],
            'touchPhase': ['Began', 'Moved', 'Moved', 'Ended', 'Began', 'Moved', 'Began', 'Moved', 'Canceled'],
            'fingerId': [1, 1, 1, 1, 2, 2, 3, 3, 3],
            'completionPerc': [0.0, 0.3, 0.6, 1.0, 0.0, 0.5, 0.0, 0.2, 0.0]
        })
    
    def test_sequence_quality_analysis(self):
        """Test sequence quality analysis."""
        result = self.enhancer.analyze_sequence_quality(self.sample_data.copy())
        
        # Check that quality columns are added
        self.assertIn('ml_quality_score', result.columns)
        self.assertIn('quality_tier', result.columns)
        self.assertIn('sequence_completeness', result.columns)
        
        # Verify quality scores are between 0 and 1
        self.assertTrue(all(0 <= score <= 1 for score in result['ml_quality_score']))
        
        # Verify quality tiers are valid
        valid_tiers = {'high', 'medium', 'low', 'unknown'}
        self.assertTrue(all(tier in valid_tiers for tier in result['quality_tier']))
        
        # Sequence 1 should have high quality (complete Began->Moved->Moved->Ended)
        seq1_quality = result[result['Touchdata_id'] == 1]['quality_tier'].iloc[0]
        self.assertIn(seq1_quality, ['high', 'medium'])  # Should be good quality
        
        # Sequence 3 should have lower quality (ends with Canceled, no Ended)
        seq3_quality = result[result['Touchdata_id'] == 3]['quality_tier'].iloc[0]
        self.assertIn(seq3_quality, ['low', 'medium'])  # Should be lower quality
    
    def test_behavioral_classification(self):
        """Test behavioral pattern classification."""
        result = self.enhancer.classify_behavioral_patterns(self.sample_data.copy())
        
        # Check that behavioral columns are added
        self.assertIn('behavioral_pattern', result.columns)
        self.assertIn('interaction_style', result.columns)
        self.assertIn('movement_type', result.columns)
        
        # Verify confidence scores are between 0 and 1
        self.assertTrue(all(0 <= conf <= 1 for conf in result['user_intent_confidence']))
        
        # Check that patterns are classified
        patterns = result['behavioral_pattern'].unique()
        self.assertGreater(len(patterns), 0)
    
    def test_anomaly_detection(self):
        """Test anomaly detection."""
        result = self.enhancer.detect_anomalies(self.sample_data.copy())
        
        # Check that anomaly columns are added
        self.assertIn('anomaly_score', result.columns)
        self.assertIn('anomaly_type', result.columns)
        self.assertIn('anomaly_confidence', result.columns)
        
        # Verify anomaly types are valid
        valid_types = {'normal', 'outlier', 'none'}
        anomaly_types = set(result['anomaly_type'].unique())
        self.assertTrue(anomaly_types.issubset(valid_types))
        
        # The outlier in sequence 2 (x=1000) should be detected
        seq2_data = result[result['Touchdata_id'] == 2]
        outlier_row = seq2_data[seq2_data['x'] == 1000]
        if not outlier_row.empty:
            # Should have higher anomaly score
            self.assertGreater(outlier_row['anomaly_score'].iloc[0], 0)

class TestMLTouchDataCleaner(unittest.TestCase):
    """Test the main ML cleaning pipeline."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.cleaner = MLTouchDataCleaner(model_dir=os.path.join(self.temp_dir, 'models'))
        
        # Create sample JSON data
        self.sample_json = {
            "message": "gameData",
            "json": {
                "dataSet": "Coloring",
                "touchData": {
                    "1": [
                        {
                            "x": 800.0, "y": 504.0, "time": 3192.546,
                            "touchPhase": "Began", "fingerId": 0,
                            "color": "RedDefault", "zone": "Wall",
                            "completionPerc": 0.0
                        },
                        {
                            "x": 810.0, "y": 510.0, "time": 3192.646,
                            "touchPhase": "Moved", "fingerId": 0,
                            "color": "RedDefault", "zone": "Wall",
                            "completionPerc": 0.5
                        },
                        {
                            "x": 820.0, "y": 515.0, "time": 3192.746,
                            "touchPhase": "Ended", "fingerId": 0,
                            "color": "RedDefault", "zone": "Wall",
                            "completionPerc": 1.0
                        }
                    ]
                }
            }
        }
    
    def test_json_to_dataframe_conversion(self):
        """Test JSON to DataFrame conversion."""
        df = self.cleaner._json_to_dataframe(self.sample_json)
        
        # Check that DataFrame is created correctly
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 3)  # 3 touch points
        
        # Check required columns
        required_cols = ['x', 'y', 'time', 'touchPhase', 'Touchdata_id', 'event_index']
        for col in required_cols:
            self.assertIn(col, df.columns)
        
        # Verify data integrity
        self.assertEqual(df.iloc[0]['x'], 800.0)
        self.assertEqual(df.iloc[0]['touchPhase'], 'Began')
        self.assertEqual(df.iloc[2]['touchPhase'], 'Ended')
    
    def test_dataframe_to_json_conversion(self):
        """Test DataFrame to JSON conversion."""
        # Convert to DataFrame and back
        df = self.cleaner._json_to_dataframe(self.sample_json)
        
        # Add some ML metadata
        df['ml_quality_score'] = 0.9
        df['quality_tier'] = 'high'
        df['behavioral_pattern'] = 'deliberate'
        
        # Convert back to JSON
        result_json = self.cleaner._dataframe_to_json(df, self.sample_json)
        
        # Check structure is preserved
        self.assertIn('message', result_json)
        self.assertIn('json', result_json)
        self.assertIn('touchData', result_json['json'])
        
        # Check ML metadata is added
        self.assertIn('ml_metadata', result_json)
        self.assertIn('total_sequences', result_json['ml_metadata'])
        self.assertIn('quality_distribution', result_json['ml_metadata'])
        
        # Verify original data is preserved
        touch_data = result_json['json']['touchData']['1']
        self.assertEqual(len(touch_data), 3)
        self.assertEqual(touch_data[0]['x'], 800.0)
        self.assertEqual(touch_data[0]['touchPhase'], 'Began')
    
    def test_clean_and_enhance_data(self):
        """Test the complete data cleaning and enhancement pipeline."""
        df = self.cleaner._json_to_dataframe(self.sample_json)
        original_cols = set(df.columns)
        
        # Apply cleaning and enhancement
        enhanced_df = self.cleaner.clean_and_enhance_data(df)
        new_cols = set(enhanced_df.columns)
        
        # Check that many new columns were added
        added_cols = new_cols - original_cols
        self.assertGreater(len(added_cols), 15)  # Should add many metadata columns
        
        # Check specific metadata columns
        expected_metadata_cols = [
            'ml_quality_score', 'quality_tier', 'behavioral_pattern',
            'anomaly_score', 'usage_recommendations'
        ]
        for col in expected_metadata_cols:
            self.assertIn(col, enhanced_df.columns)
        
        # Verify original data is preserved
        for col in ['x', 'y', 'time', 'touchPhase']:
            pd.testing.assert_series_equal(
                df[col], enhanced_df[col], check_names=False
            )
    
    def test_data_preservation(self):
        """Test that original coordinate data is never modified."""
        df = self.cleaner._json_to_dataframe(self.sample_json)
        original_x = df['x'].copy()
        original_y = df['y'].copy()
        original_time = df['time'].copy()
        original_phase = df['touchPhase'].copy()
        
        # Apply full cleaning pipeline
        enhanced_df = self.cleaner.clean_and_enhance_data(df)
        
        # Verify critical data is unchanged
        pd.testing.assert_series_equal(original_x, enhanced_df['x'], check_names=False)
        pd.testing.assert_series_equal(original_y, enhanced_df['y'], check_names=False)
        pd.testing.assert_series_equal(original_time, enhanced_df['time'], check_names=False)
        pd.testing.assert_series_equal(original_phase, enhanced_df['touchPhase'], check_names=False)
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete ML cleaning system."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a sample JSON file
        self.sample_json_file = os.path.join(self.temp_dir, 'test_coloring.json')
        sample_data = {
            "message": "gameData",
            "json": {
                "dataSet": "Coloring",
                "touchData": {
                    "1": [
                        {"x": 100, "y": 100, "time": 1000, "touchPhase": "Began", "fingerId": 0, "color": "Red", "completionPerc": 0.0},
                        {"x": 110, "y": 105, "time": 1100, "touchPhase": "Moved", "fingerId": 0, "color": "Red", "completionPerc": 0.5},
                        {"x": 120, "y": 110, "time": 1200, "touchPhase": "Ended", "fingerId": 0, "color": "Red", "completionPerc": 1.0}
                    ],
                    "2": [
                        {"x": 200, "y": 200, "time": 2000, "touchPhase": "Began", "fingerId": 1, "color": "Blue", "completionPerc": 0.0},
                        {"x": 210, "y": 205, "time": 2100, "touchPhase": "Canceled", "fingerId": 1, "color": "Blue", "completionPerc": 0.3}
                    ]
                }
            }
        }
        
        with open(self.sample_json_file, 'w') as f:
            json.dump(sample_data, f)
    
    def test_end_to_end_processing(self):
        """Test complete end-to-end processing of a JSON file."""
        cleaner = MLTouchDataCleaner(model_dir=os.path.join(self.temp_dir, 'models'))
        
        # Process the file
        output_file = os.path.join(self.temp_dir, 'enhanced_test.json')
        result = cleaner.process_json_file(self.sample_json_file, output_file)
        
        # Check processing was successful
        self.assertEqual(result['status'], 'success')
        self.assertTrue(os.path.exists(output_file))
        
        # Load and verify enhanced data
        with open(output_file, 'r') as f:
            enhanced_data = json.load(f)
        
        # Check structure is preserved
        self.assertIn('json', enhanced_data)
        self.assertIn('touchData', enhanced_data['json'])
        
        # Check ML metadata is added
        self.assertIn('ml_metadata', enhanced_data)
        
        # Verify statistics
        stats = result['statistics']
        self.assertEqual(stats['original_data_points'], 5)  # 3 + 2 touch points
        self.assertEqual(stats['sequences_processed'], 2)  # 2 sequences
        self.assertGreater(stats['features_added'], 10)  # Many features added
    
    def tearDown(self):
        """Clean up integration test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)
