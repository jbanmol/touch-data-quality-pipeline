#!/usr/bin/env python3
"""
Test Suite for Enhanced ML-Based Data Flagging System

This test suite validates the enhanced ML system functionality,
algorithm comparison, feature engineering, and integration with
the existing data processing pipeline.
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import shutil
import logging

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from ml.enhanced_ml_flagging import EnhancedMLFlaggingSystem
    from ml.algorithm_comparison import MLAlgorithmComparator
    from ml.advanced_feature_engineering import AdvancedTouchFeatureEngineer
    from ml.ml_integration import MLIntegrationManager, enhance_dataframe_with_advanced_ml
    ENHANCED_ML_AVAILABLE = True
except ImportError as e:
    print(f"Enhanced ML components not available: {e}")
    ENHANCED_ML_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestEnhancedMLSystem(unittest.TestCase):
    """Test the enhanced ML flagging system."""
    
    def setUp(self):
        """Set up test data and temporary directories."""
        if not ENHANCED_ML_AVAILABLE:
            self.skipTest("Enhanced ML components not available")
        
        # Create test data
        self.test_data = self._create_test_data()
        
        # Create temporary directory for models
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up temporary files."""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _create_test_data(self):
        """Create realistic test touch data."""
        np.random.seed(42)
        
        data = []
        
        # Create multiple touch sequences
        for touchdata_id in range(1, 6):
            # Generate a realistic touch sequence
            n_points = np.random.randint(5, 20)
            
            # Start position
            start_x = np.random.uniform(100, 900)
            start_y = np.random.uniform(100, 700)
            
            # Generate smooth trajectory
            x_coords = [start_x]
            y_coords = [start_y]
            
            for i in range(1, n_points):
                # Add some noise but keep trajectory smooth
                dx = np.random.normal(0, 20)
                dy = np.random.normal(0, 20)
                x_coords.append(x_coords[-1] + dx)
                y_coords.append(y_coords[-1] + dy)
            
            # Generate time sequence
            base_time = touchdata_id * 1000
            time_intervals = np.random.exponential(0.05, n_points-1)
            times = [base_time] + [base_time + np.sum(time_intervals[:i]) for i in range(1, n_points)]
            
            # Generate touch phases
            phases = ['Began'] + ['Moved'] * (n_points - 2) + ['Ended']
            if np.random.random() < 0.2:  # 20% chance of cancellation
                phases[-1] = 'Canceled'
            
            # Create sequence data
            for i in range(n_points):
                data.append({
                    'Touchdata_id': touchdata_id,
                    'event_index': i,
                    'x': x_coords[i],
                    'y': y_coords[i],
                    'time': times[i],
                    'touchPhase': phases[i],
                    'fingerId': 0,
                    'color': 'RedDefault',
                    'completionPerc': (i / (n_points - 1)) * 100,
                    'zone': 'Wall'
                })
        
        return pd.DataFrame(data)
    
    def test_enhanced_ml_flagging_system_initialization(self):
        """Test that the enhanced ML flagging system initializes correctly."""
        system = EnhancedMLFlaggingSystem(model_dir=self.temp_dir)
        
        self.assertIsNotNone(system.anomaly_algorithms)
        self.assertIsNotNone(system.clustering_algorithms)
        self.assertIsNotNone(system.scalers)
        self.assertIn('isolation_forest', system.anomaly_algorithms)
        self.assertIn('kmeans', system.clustering_algorithms)
    
    def test_feature_extraction(self):
        """Test advanced feature extraction."""
        system = EnhancedMLFlaggingSystem(model_dir=self.temp_dir)
        
        # Extract features
        df_features = system.extract_advanced_features(self.test_data)
        
        # Check that features were added
        self.assertGreater(len(df_features.columns), len(self.test_data.columns))
        
        # Check for specific feature types
        feature_cols = [col for col in df_features.columns if col not in self.test_data.columns]
        
        # Should have temporal features
        temporal_features = [col for col in feature_cols if 'time' in col or 'duration' in col]
        self.assertGreater(len(temporal_features), 0)
        
        # Should have spatial features
        spatial_features = [col for col in feature_cols if 'distance' in col or 'spatial' in col]
        self.assertGreater(len(spatial_features), 0)
        
        # Should have statistical features
        statistical_features = [col for col in feature_cols if any(stat in col for stat in ['mean', 'std', 'skew'])]
        self.assertGreater(len(statistical_features), 0)
    
    def test_algorithm_comparison(self):
        """Test algorithm comparison functionality."""
        system = EnhancedMLFlaggingSystem(model_dir=self.temp_dir)
        
        # Extract features first
        df_features = system.extract_advanced_features(self.test_data)
        
        # Run algorithm comparison
        results = system.compare_algorithms(df_features)
        
        # Check that results contain expected sections
        self.assertIn('anomaly_detection', results)
        self.assertIn('clustering', results)
        self.assertIn('scaling', results)
        
        # Check that best algorithms were selected
        self.assertIsNotNone(system.best_anomaly_algorithm)
        self.assertIsNotNone(system.best_clustering_algorithm)
        self.assertIsNotNone(system.best_scaler)
    
    def test_anomaly_detection(self):
        """Test anomaly detection functionality."""
        system = EnhancedMLFlaggingSystem(model_dir=self.temp_dir)
        
        # Extract features and run comparison first
        df_features = system.extract_advanced_features(self.test_data)
        system.compare_algorithms(df_features)
        
        # Run anomaly detection
        df_anomalies = system.detect_anomalies_advanced(df_features)
        
        # Check that anomaly columns were added
        self.assertIn('anomaly_score', df_anomalies.columns)
        self.assertIn('anomaly_type', df_anomalies.columns)
        self.assertIn('anomaly_confidence', df_anomalies.columns)
        
        # Check data types and ranges
        self.assertTrue(df_anomalies['anomaly_score'].dtype in [np.float64, np.float32])
        self.assertTrue(all(df_anomalies['anomaly_type'].isin(['normal', 'outlier'])))
    
    def test_behavioral_classification(self):
        """Test behavioral pattern classification."""
        system = EnhancedMLFlaggingSystem(model_dir=self.temp_dir)
        
        # Extract features first
        df_features = system.extract_advanced_features(self.test_data)
        
        # Run behavioral classification
        df_behavioral = system.classify_behavioral_patterns_advanced(df_features)
        
        # Check that behavioral columns were added
        self.assertIn('behavioral_pattern', df_behavioral.columns)
        self.assertIn('interaction_style', df_behavioral.columns)
        self.assertIn('user_intent_confidence', df_behavioral.columns)
        self.assertIn('movement_type', df_behavioral.columns)
        
        # Check that confidence values are in valid range
        confidence_values = df_behavioral['user_intent_confidence'].dropna()
        self.assertTrue(all(0.0 <= conf <= 1.0 for conf in confidence_values))
    
    def test_quality_score_calculation(self):
        """Test quality score calculation."""
        system = EnhancedMLFlaggingSystem(model_dir=self.temp_dir)
        
        # Extract features and run analysis
        df_features = system.extract_advanced_features(self.test_data)
        df_anomalies = system.detect_anomalies_advanced(df_features)
        df_behavioral = system.classify_behavioral_patterns_advanced(df_anomalies)
        
        # Calculate quality scores
        df_quality = system.calculate_quality_score_advanced(df_behavioral)
        
        # Check that quality scores were added
        self.assertIn('ml_quality_score', df_quality.columns)
        
        # Check that scores are in valid range (0-100)
        quality_scores = df_quality['ml_quality_score'].dropna()
        self.assertTrue(all(0 <= score <= 100 for score in quality_scores))
    
    def test_consolidated_metadata_generation(self):
        """Test generation of consolidated metadata columns."""
        system = EnhancedMLFlaggingSystem(model_dir=self.temp_dir)
        
        # Run full enhancement
        df_enhanced = system.enhance_dataframe(self.test_data)
        
        # Check that all 4 consolidated columns are present
        required_columns = ['quality_score', 'interaction_type', 'anomaly_flag', 'research_suitability']
        for col in required_columns:
            self.assertIn(col, df_enhanced.columns)
        
        # Check data types and valid values
        self.assertTrue(df_enhanced['quality_score'].dtype in [np.int64, np.int32])
        self.assertTrue(all(0 <= score <= 100 for score in df_enhanced['quality_score']))
        
        valid_interaction_types = ['Precise', 'Quick', 'Hesitant', 'Erratic', 'Unknown']
        self.assertTrue(all(df_enhanced['interaction_type'].isin(valid_interaction_types)))
        
        valid_anomaly_flags = ['None', 'Technical', 'Behavioral', 'Spatial']
        self.assertTrue(all(df_enhanced['anomaly_flag'].isin(valid_anomaly_flags)))
    
    def test_data_preservation(self):
        """Test that original data is preserved during enhancement."""
        system = EnhancedMLFlaggingSystem(model_dir=self.temp_dir)
        
        # Store original data
        original_data = self.test_data.copy()
        
        # Run enhancement
        df_enhanced = system.enhance_dataframe(self.test_data)
        
        # Check that original columns are preserved
        original_columns = ['Touchdata_id', 'event_index', 'x', 'y', 'time', 'touchPhase', 'fingerId']
        for col in original_columns:
            if col in original_data.columns:
                pd.testing.assert_series_equal(
                    df_enhanced[col], 
                    original_data[col], 
                    check_names=False,
                    msg=f"Original data modified in column {col}"
                )


class TestMLIntegration(unittest.TestCase):
    """Test ML integration with existing pipeline."""
    
    def setUp(self):
        """Set up test data."""
        if not ENHANCED_ML_AVAILABLE:
            self.skipTest("Enhanced ML components not available")
        
        self.test_data = self._create_test_data()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files."""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _create_test_data(self):
        """Create test data similar to the main test class."""
        np.random.seed(42)
        
        data = []
        for touchdata_id in range(1, 4):
            n_points = 10
            for i in range(n_points):
                data.append({
                    'Touchdata_id': touchdata_id,
                    'event_index': i,
                    'x': 100 + i * 10 + np.random.normal(0, 5),
                    'y': 200 + i * 5 + np.random.normal(0, 5),
                    'time': touchdata_id * 100 + i * 0.1,
                    'touchPhase': 'Began' if i == 0 else ('Ended' if i == n_points-1 else 'Moved'),
                    'fingerId': 0,
                    'color': 'RedDefault',
                    'completionPerc': (i / (n_points - 1)) * 100,
                    'zone': 'Wall'
                })
        
        return pd.DataFrame(data)
    
    def test_ml_integration_manager(self):
        """Test ML integration manager functionality."""
        manager = MLIntegrationManager(enable_algorithm_comparison=False)
        
        # Test enhancement
        df_enhanced = manager.enhance_dataframe_with_ml(self.test_data)
        
        # Check that consolidated columns are present
        required_columns = ['quality_score', 'interaction_type', 'anomaly_flag', 'research_suitability']
        for col in required_columns:
            self.assertIn(col, df_enhanced.columns)
    
    def test_convenience_function(self):
        """Test the convenience function for ML enhancement."""
        df_enhanced = enhance_dataframe_with_advanced_ml(self.test_data)
        
        # Check that enhancement worked
        self.assertGreater(len(df_enhanced.columns), len(self.test_data.columns))
        
        # Check for consolidated columns
        required_columns = ['quality_score', 'interaction_type', 'anomaly_flag', 'research_suitability']
        for col in required_columns:
            self.assertIn(col, df_enhanced.columns)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
