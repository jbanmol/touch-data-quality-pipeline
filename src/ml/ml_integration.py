#!/usr/bin/env python3
"""
ML Integration Module for Touch Data Processing

This module provides seamless integration between the enhanced ML flagging system
and the existing data processing pipeline, ensuring compatibility and maintaining
all existing functionality while adding advanced ML capabilities.
"""

import pandas as pd
import numpy as np
import logging
import warnings
import os
import sys
from typing import Dict, List, Tuple, Optional, Any

# Add the src directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from .enhanced_ml_flagging import EnhancedMLFlaggingSystem
    from .algorithm_comparison import MLAlgorithmComparator
    from .advanced_feature_engineering import AdvancedTouchFeatureEngineer
    ML_ENHANCED_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Enhanced ML components not available: {e}")
    ML_ENHANCED_AVAILABLE = False

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class MLIntegrationManager:
    """
    Manages integration between ML systems and existing data processing pipeline.
    Provides fallback mechanisms and ensures compatibility.
    """
    
    def __init__(self, enable_algorithm_comparison: bool = True):
        self.enable_algorithm_comparison = enable_algorithm_comparison
        self.ml_system = None
        self.algorithm_comparator = None
        self.feature_engineer = None
        self.is_initialized = False
        self.algorithm_results = {}
        
        if ML_ENHANCED_AVAILABLE:
            try:
                self.ml_system = EnhancedMLFlaggingSystem()
                if enable_algorithm_comparison:
                    self.algorithm_comparator = MLAlgorithmComparator()
                self.feature_engineer = AdvancedTouchFeatureEngineer()
                self.is_initialized = True
                logger.info("ML Integration Manager initialized with enhanced components")
            except Exception as e:
                logger.warning(f"Failed to initialize enhanced ML components: {e}")
                self.is_initialized = False
        else:
            logger.warning("Enhanced ML components not available")
        
    def enhance_dataframe_with_ml(self, df: pd.DataFrame, 
                                 run_algorithm_comparison: bool = False) -> pd.DataFrame:
        """
        Main method to enhance DataFrame with ML-based flagging.
        
        Args:
            df: Input DataFrame with touch data
            run_algorithm_comparison: Whether to run algorithm comparison
            
        Returns:
            DataFrame with enhanced ML metadata
        """
        logger.info("Starting ML enhancement integration...")
        
        if not self.is_initialized:
            logger.warning("ML system not initialized, using fallback")
            return self._apply_basic_fallback(df)
        
        try:
            # Step 1: Run algorithm comparison if requested
            if run_algorithm_comparison and self.algorithm_comparator:
                logger.info("Running algorithm comparison...")
                self.algorithm_results = self.algorithm_comparator.run_comprehensive_comparison(df)
                
                # Print summary
                self.algorithm_comparator.print_summary(self.algorithm_results)
                
                # Save results
                self.algorithm_comparator.save_results(self.algorithm_results)
            
            # Step 2: Apply enhanced ML flagging
            enhanced_df = self.ml_system.enhance_dataframe(df)
            
            # Step 3: Validate results
            validated_df = self._validate_ml_results(enhanced_df, df)
            
            logger.info("ML enhancement integration completed successfully")
            return validated_df
            
        except Exception as e:
            logger.error(f"ML enhancement failed: {e}")
            logger.info("Falling back to basic enhancement")
            return self._apply_basic_fallback(df)
    
    def _validate_ml_results(self, enhanced_df: pd.DataFrame, 
                           original_df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate ML enhancement results and ensure data integrity.
        
        Args:
            enhanced_df: DataFrame with ML enhancements
            original_df: Original DataFrame
            
        Returns:
            Validated DataFrame
        """
        logger.info("Validating ML enhancement results...")
        
        # Check that original data is preserved
        original_cols = ['Touchdata_id', 'event_index', 'x', 'y', 'time', 
                        'touchPhase', 'fingerId', 'color', 'completionPerc', 'zone']
        
        for col in original_cols:
            if col in original_df.columns and col in enhanced_df.columns:
                # Check for data modification (coordinates should never change)
                if col in ['x', 'y']:
                    if not enhanced_df[col].equals(original_df[col]):
                        logger.error(f"Coordinate data modified in column {col}! Restoring original values.")
                        enhanced_df[col] = original_df[col]
                
                # Check for missing data
                if enhanced_df[col].isna().sum() > original_df[col].isna().sum():
                    logger.warning(f"Additional NaN values introduced in column {col}")
        
        # Validate consolidated ML columns
        required_ml_cols = ['quality_score', 'interaction_type', 'anomaly_flag', 'research_suitability']
        for col in required_ml_cols:
            if col not in enhanced_df.columns:
                logger.warning(f"Required ML column {col} missing, adding default values")
                enhanced_df[col] = self._get_default_value(col)
            else:
                # Validate data types and ranges
                enhanced_df = self._validate_ml_column(enhanced_df, col)
        
        # Check for reasonable quality scores
        if 'quality_score' in enhanced_df.columns:
            invalid_scores = (enhanced_df['quality_score'] < 0) | (enhanced_df['quality_score'] > 100)
            if invalid_scores.any():
                logger.warning(f"Invalid quality scores found, clamping to 0-100 range")
                enhanced_df.loc[invalid_scores, 'quality_score'] = enhanced_df.loc[invalid_scores, 'quality_score'].clip(0, 100)
        
        logger.info("ML enhancement validation completed")
        return enhanced_df
    
    def _validate_ml_column(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Validate specific ML column values."""
        if col == 'quality_score':
            # Ensure integer values between 0-100
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(50).astype(int)
            df[col] = df[col].clip(0, 100)
        
        elif col == 'interaction_type':
            # Ensure valid interaction types
            valid_types = ['Precise', 'Quick', 'Hesitant', 'Erratic', 'Unknown']
            invalid_mask = ~df[col].isin(valid_types)
            if invalid_mask.any():
                df.loc[invalid_mask, col] = 'Unknown'
        
        elif col == 'anomaly_flag':
            # Ensure valid anomaly flags
            valid_flags = ['None', 'Technical', 'Behavioral', 'Spatial']
            invalid_mask = ~df[col].isin(valid_flags)
            if invalid_mask.any():
                df.loc[invalid_mask, col] = 'None'
        
        elif col == 'research_suitability':
            # Ensure valid research suitability tags
            valid_tags = ['Timing', 'Spatial', 'Behavioral', 'All', 'Limited']
            # Also allow combinations like "Timing,Spatial"
            def is_valid_suitability(value):
                if pd.isna(value):
                    return False
                if value in valid_tags:
                    return True
                # Check if it's a comma-separated combination
                parts = [part.strip() for part in str(value).split(',')]
                return all(part in valid_tags for part in parts)
            
            invalid_mask = ~df[col].apply(is_valid_suitability)
            if invalid_mask.any():
                df.loc[invalid_mask, col] = 'Limited'
        
        return df
    
    def _get_default_value(self, col: str) -> Any:
        """Get default value for ML column."""
        defaults = {
            'quality_score': 50,
            'interaction_type': 'Unknown',
            'anomaly_flag': 'None',
            'research_suitability': 'Limited'
        }
        return defaults.get(col, 'Unknown')
    
    def _apply_basic_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply basic ML enhancement as fallback."""
        logger.info("Applying basic ML enhancement fallback...")
        
        # Initialize consolidated columns with basic values
        df['quality_score'] = 50  # Medium quality default
        df['interaction_type'] = 'Unknown'
        df['anomaly_flag'] = 'None'
        df['research_suitability'] = 'Limited'
        
        # Basic quality assessment based on sequence completeness
        if 'touchPhase' in df.columns:
            if 'Touchdata_id' in df.columns:
                groupby_col = 'Touchdata_id'
            else:
                groupby_col = 'fingerId'
            
            for seq_id, group in df.groupby(groupby_col):
                mask = df[groupby_col] == seq_id
                phases = group['touchPhase'].tolist()
                
                # Basic quality score based on sequence structure
                if len(phases) >= 2 and phases[0] == 'Began' and phases[-1] == 'Ended':
                    df.loc[mask, 'quality_score'] = 85
                    df.loc[mask, 'research_suitability'] = 'All'
                    df.loc[mask, 'interaction_type'] = 'Precise'
                elif 'Began' in phases or 'Ended' in phases:
                    df.loc[mask, 'quality_score'] = 65
                    df.loc[mask, 'research_suitability'] = 'Spatial'
                    df.loc[mask, 'interaction_type'] = 'Quick'
                else:
                    df.loc[mask, 'quality_score'] = 35
                    df.loc[mask, 'anomaly_flag'] = 'Technical'
                
                # Basic interaction type based on duration
                if 'time' in group.columns and len(group) > 1:
                    duration = group['time'].max() - group['time'].min()
                    if duration < 0.5:
                        df.loc[mask, 'interaction_type'] = 'Quick'
                    elif duration > 3.0:
                        df.loc[mask, 'interaction_type'] = 'Hesitant'
                    else:
                        df.loc[mask, 'interaction_type'] = 'Precise'
                
                # Basic anomaly detection based on sequence issues
                if 'Canceled' in phases:
                    df.loc[mask, 'anomaly_flag'] = 'Behavioral'
                    df.loc[mask, 'quality_score'] = max(30, df.loc[mask, 'quality_score'].iloc[0] - 20)
        
        return df
    
    def get_algorithm_comparison_results(self) -> Dict[str, Any]:
        """Get the results of algorithm comparison."""
        return self.algorithm_results
    
    def save_ml_model_state(self, output_dir: str = "ML/models"):
        """Save the current state of ML models."""
        if not self.is_initialized:
            logger.warning("ML system not initialized, cannot save state")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Save algorithm comparison results
            if self.algorithm_results:
                import json
                results_path = os.path.join(output_dir, "algorithm_comparison_results.json")
                with open(results_path, 'w') as f:
                    json.dump(self.algorithm_results, f, indent=2, default=str)
                logger.info(f"Algorithm comparison results saved to {results_path}")
            
            # Save feature importance if available
            if self.feature_engineer and hasattr(self.feature_engineer, 'feature_importance'):
                importance_path = os.path.join(output_dir, "feature_importance.json")
                with open(importance_path, 'w') as f:
                    json.dump(self.feature_engineer.feature_importance, f, indent=2)
                logger.info(f"Feature importance saved to {importance_path}")
            
        except Exception as e:
            logger.error(f"Failed to save ML model state: {e}")
    
    def load_ml_model_state(self, input_dir: str = "ML/models"):
        """Load previously saved ML model state."""
        if not os.path.exists(input_dir):
            logger.warning(f"ML model directory {input_dir} does not exist")
            return
        
        try:
            # Load algorithm comparison results
            results_path = os.path.join(input_dir, "algorithm_comparison_results.json")
            if os.path.exists(results_path):
                import json
                with open(results_path, 'r') as f:
                    self.algorithm_results = json.load(f)
                logger.info(f"Algorithm comparison results loaded from {results_path}")
            
            # Load feature importance
            importance_path = os.path.join(input_dir, "feature_importance.json")
            if os.path.exists(importance_path) and self.feature_engineer:
                import json
                with open(importance_path, 'r') as f:
                    self.feature_engineer.feature_importance = json.load(f)
                logger.info(f"Feature importance loaded from {importance_path}")
            
        except Exception as e:
            logger.error(f"Failed to load ML model state: {e}")


# Convenience function for easy integration
def enhance_dataframe_with_advanced_ml(df: pd.DataFrame, 
                                     run_algorithm_comparison: bool = False) -> pd.DataFrame:
    """
    Convenience function to enhance DataFrame with advanced ML capabilities.
    
    Args:
        df: Input DataFrame with touch data
        run_algorithm_comparison: Whether to run algorithm comparison
        
    Returns:
        DataFrame with enhanced ML metadata
    """
    manager = MLIntegrationManager()
    return manager.enhance_dataframe_with_ml(df, run_algorithm_comparison)
