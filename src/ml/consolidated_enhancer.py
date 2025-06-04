#!/usr/bin/env python3
"""
ML Consolidated Metadata Enhancer

This module integrates the existing ML pipeline to generate 4 consolidated
metadata columns that provide maximum insight with minimal complexity:

1. Quality Score (0-100): Comprehensive quality metric
2. Interaction Type: "Precise"/"Quick"/"Hesitant"/"Erratic" 
3. Anomaly Flag: "None"/"Technical"/"Behavioral"/"Spatial"
4. Research Suitability: "Timing"/"Spatial"/"Behavioral"/"All"
"""

import pandas as pd
import numpy as np
import logging
import os
import sys
from typing import Dict, List, Tuple, Optional, Any

try:
    from .ml_cleaning_pipeline import MLTouchDataCleaner
    from .feature_engineering import TouchFeatureEngineer
    from .metadata_enhancer import TouchDataMetadataEnhancer
    ML_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ML modules not available: {e}")
    ML_AVAILABLE = False

logger = logging.getLogger(__name__)

class ConsolidatedMLEnhancer:
    """
    Consolidates detailed ML features into 4 simplified metadata columns
    for enhanced data understanding with minimal complexity.
    """
    
    def __init__(self):
        self.ml_cleaner = None
        self.feature_engineer = None
        self.metadata_enhancer = None
        self.is_initialized = False
        
        if ML_AVAILABLE:
            try:
                self.ml_cleaner = MLTouchDataCleaner()
                self.feature_engineer = TouchFeatureEngineer()
                self.metadata_enhancer = TouchDataMetadataEnhancer()
                self.is_initialized = True
                logger.info("ML Consolidated Enhancer initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize ML components: {e}")
                self.is_initialized = False
        else:
            logger.warning("ML modules not available - using fallback implementation")
    
    def enhance_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add consolidated ML metadata columns to the DataFrame.
        
        Args:
            df: DataFrame with touch data
            
        Returns:
            DataFrame with 4 additional consolidated ML columns
        """
        logger.info("Adding consolidated ML metadata columns...")
        
        # Initialize the 4 consolidated columns
        df['quality_score'] = 0
        df['interaction_type'] = 'Unknown'
        df['anomaly_flag'] = 'None'
        df['research_suitability'] = 'Limited'
        
        if not self.is_initialized:
            logger.warning("ML not available - using rule-based fallback")
            return self._apply_fallback_enhancement(df)
        
        try:
            # Apply full ML pipeline to extract detailed features
            enhanced_df = self._apply_ml_pipeline(df)
            
            # Consolidate detailed features into 4 simplified columns
            consolidated_df = self._consolidate_ml_features(enhanced_df)
            
            return consolidated_df
            
        except Exception as e:
            logger.error(f"ML enhancement failed: {e}")
            logger.info("Falling back to rule-based enhancement")
            return self._apply_fallback_enhancement(df)
    
    def _apply_ml_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the full ML pipeline to extract detailed features."""
        logger.info("Applying ML pipeline for feature extraction...")
        
        # Step 1: Feature Engineering
        df_features = self.feature_engineer.extract_all_features(df)
        
        # Step 2: Quality Assessment
        df_quality = self.metadata_enhancer.analyze_sequence_quality(df_features)
        
        # Step 3: Behavioral Classification
        df_behavioral = self.metadata_enhancer.classify_behavioral_patterns(df_quality)
        
        # Step 4: Anomaly Detection
        df_anomalies = self.metadata_enhancer.detect_anomalies(df_behavioral)
        
        return df_anomalies
    
    def _consolidate_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Consolidate detailed ML features into 4 simplified columns."""
        logger.info("Consolidating ML features into simplified metadata...")
        
        # Group by sequence for sequence-level consolidation
        if 'Touchdata_id' in df.columns:
            groupby_col = 'Touchdata_id'
        else:
            groupby_col = 'fingerId'
        
        for seq_id, group in df.groupby(groupby_col):
            mask = df[groupby_col] == seq_id
            
            # 1. Quality Score (0-100)
            quality_score = self._calculate_quality_score(group)
            df.loc[mask, 'quality_score'] = quality_score
            
            # 2. Interaction Type
            interaction_type = self._determine_interaction_type(group)
            df.loc[mask, 'interaction_type'] = interaction_type
            
            # 3. Anomaly Flag
            anomaly_flag = self._determine_anomaly_flag(group)
            df.loc[mask, 'anomaly_flag'] = anomaly_flag
            
            # 4. Research Suitability
            research_suitability = self._determine_research_suitability(
                quality_score, interaction_type, anomaly_flag, group
            )
            df.loc[mask, 'research_suitability'] = research_suitability
        
        return df
    
    def _calculate_quality_score(self, group: pd.DataFrame) -> int:
        """
        Calculate Quality Score (0-100) from ML features.
        
        Formula: (0.4 * sequence_completeness + 0.3 * temporal_consistency + 0.3 * spatial_consistency) * 100
        """
        # Get ML-derived quality components
        sequence_completeness = group['sequence_completeness'].iloc[0] if 'sequence_completeness' in group.columns else 0.5
        temporal_consistency = group['temporal_consistency'].iloc[0] if 'temporal_consistency' in group.columns else 0.5
        spatial_consistency = group['spatial_consistency'].iloc[0] if 'spatial_consistency' in group.columns else 0.5
        
        # Apply weighted formula
        quality_score = (
            0.4 * sequence_completeness +
            0.3 * temporal_consistency +
            0.3 * spatial_consistency
        ) * 100
        
        return int(np.clip(quality_score, 0, 100))
    
    def _determine_interaction_type(self, group: pd.DataFrame) -> str:
        """
        Determine Interaction Type from ML behavioral features.
        
        Categories: "Precise", "Quick", "Hesitant", "Erratic"
        """
        # Use ML behavioral pattern if available
        if 'behavioral_pattern' in group.columns:
            pattern = group['behavioral_pattern'].iloc[0]
            if pattern == 'deliberate':
                return 'Precise'
            elif pattern == 'tap':
                return 'Quick'
            elif pattern == 'hold':
                return 'Hesitant'
            elif pattern in ['complex', 'erratic']:
                return 'Erratic'
        
        # Fallback to velocity and movement analysis
        if 'velocity' in group.columns and len(group) > 2:
            velocities = group['velocity'].dropna()
            if len(velocities) > 0:
                mean_velocity = velocities.mean()
                velocity_std = velocities.std()
                
                # Decision tree based on velocity characteristics
                if mean_velocity < 10:  # Slow movement
                    return 'Precise'
                elif mean_velocity > 50:  # Fast movement
                    return 'Quick'
                elif velocity_std > mean_velocity * 0.8:  # High variability
                    return 'Erratic'
                else:
                    return 'Hesitant'
        
        return 'Unknown'
    
    def _determine_anomaly_flag(self, group: pd.DataFrame) -> str:
        """
        Determine Anomaly Flag from ML anomaly detection and existing flags.
        
        Types: "None", "Technical", "Behavioral", "Spatial"
        """
        # Check ML anomaly detection
        if 'anomaly_type' in group.columns:
            anomaly_type = group['anomaly_type'].iloc[0]
            if anomaly_type == 'outlier':
                # Classify the type of anomaly based on other features
                return self._classify_anomaly_type(group)
        
        # Check existing flags for anomaly classification
        if 'flags' in group.columns:
            flags_str = group['flags'].iloc[0] if not group['flags'].empty else ''
            if isinstance(flags_str, str) and flags_str:
                return self._classify_anomaly_from_flags(flags_str)
        
        return 'None'
    
    def _classify_anomaly_type(self, group: pd.DataFrame) -> str:
        """Classify the specific type of anomaly."""
        # Technical anomalies: sequence structure issues
        if any(col in group.columns for col in ['has_began', 'has_ended', 'sequence_valid_pattern']):
            if (group.get('has_began', [1]).iloc[0] == 0 or 
                group.get('has_ended', [1]).iloc[0] == 0 or
                group.get('sequence_valid_pattern', [1]).iloc[0] == 0):
                return 'Technical'
        
        # Spatial anomalies: movement pattern issues
        if 'velocity_outlier' in group.columns and group['velocity_outlier'].any():
            return 'Spatial'
        
        # Behavioral anomalies: user behavior issues
        if 'behavioral_pattern' in group.columns:
            pattern = group['behavioral_pattern'].iloc[0]
            if pattern in ['interrupted', 'irregular']:
                return 'Behavioral'
        
        return 'Behavioral'  # Default for unclassified anomalies
    
    def _classify_anomaly_from_flags(self, flags_str: str) -> str:
        """Classify anomaly type from existing flags."""
        flags = flags_str.lower()
        
        # Technical flags
        technical_flags = ['missing_began', 'missing_ended', 'improper_sequence_order', 
                          'orphaned_events', 'multiple_end_events', 'unterminated']
        if any(flag in flags for flag in technical_flags):
            return 'Technical'
        
        # Spatial flags
        spatial_flags = ['zero_distance', 'phantom_move', 'overlapping_fingerids']
        if any(flag in flags for flag in spatial_flags):
            return 'Spatial'
        
        # Behavioral flags
        behavioral_flags = ['has_canceled', 'too_few_points', 'short_duration']
        if any(flag in flags for flag in behavioral_flags):
            return 'Behavioral'
        
        return 'Technical'  # Default classification
    
    def _determine_research_suitability(self, quality_score: int, interaction_type: str, 
                                      anomaly_flag: str, group: pd.DataFrame) -> str:
        """
        Determine Research Suitability based on consolidated metrics.
        
        Tags: "Timing", "Spatial", "Behavioral", "All"
        """
        # High quality data suitable for all research
        if quality_score >= 80 and anomaly_flag == 'None':
            return 'All'
        
        # Medium-high quality with specific suitability
        if quality_score >= 60:
            if anomaly_flag == 'None' or anomaly_flag == 'Behavioral':
                if interaction_type in ['Precise', 'Quick']:
                    return 'Timing,Spatial'
                else:
                    return 'Behavioral'
            elif anomaly_flag == 'Technical':
                return 'Behavioral'
            elif anomaly_flag == 'Spatial':
                return 'Timing'
        
        # Lower quality - limited suitability
        if quality_score >= 40:
            if anomaly_flag == 'None':
                return 'Behavioral'
            elif anomaly_flag == 'Behavioral':
                return 'Timing'
        
        return 'Limited'
    
    def _apply_fallback_enhancement(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply rule-based fallback when ML is not available."""
        logger.info("Applying rule-based fallback enhancement...")
        
        # Group by sequence identifier
        if 'Touchdata_id' in df.columns:
            groupby_col = 'Touchdata_id'
        else:
            groupby_col = 'fingerId'
        
        for seq_id, group in df.groupby(groupby_col):
            mask = df[groupby_col] == seq_id
            
            # Simple rule-based quality score
            quality_score = self._calculate_fallback_quality(group)
            df.loc[mask, 'quality_score'] = quality_score
            
            # Simple interaction type classification
            interaction_type = self._classify_fallback_interaction(group)
            df.loc[mask, 'interaction_type'] = interaction_type
            
            # Simple anomaly detection from flags
            anomaly_flag = self._detect_fallback_anomaly(group)
            df.loc[mask, 'anomaly_flag'] = anomaly_flag
            
            # Simple research suitability
            research_suitability = self._determine_fallback_suitability(
                quality_score, interaction_type, anomaly_flag
            )
            df.loc[mask, 'research_suitability'] = research_suitability
        
        return df
    
    def _calculate_fallback_quality(self, group: pd.DataFrame) -> int:
        """Calculate quality score using simple rules."""
        score = 50  # Base score
        
        # Check sequence completeness
        phases = group['touchPhase'].tolist()
        if len(phases) >= 2:
            if phases[0] in ['Began', 'B'] and phases[-1] in ['Ended', 'E']:
                score += 30
            elif phases[-1] in ['Canceled', 'C']:
                score += 15
        
        # Check for flags
        if 'flags' in group.columns:
            flags_str = str(group['flags'].iloc[0])
            if flags_str and flags_str != 'nan' and flags_str != '':
                score -= 20  # Penalize flagged sequences
        
        # Check sequence length
        if len(group) >= 3:
            score += 10
        elif len(group) < 2:
            score -= 20
        
        return int(np.clip(score, 0, 100))
    
    def _classify_fallback_interaction(self, group: pd.DataFrame) -> str:
        """Classify interaction type using simple rules."""
        phases = group['touchPhase'].tolist()
        duration = group['time'].max() - group['time'].min() if len(group) > 1 else 0
        
        if len(phases) == 2 and duration < 200:  # Quick tap
            return 'Quick'
        elif 'Stationary' in phases or 'S' in phases:
            return 'Hesitant'
        elif duration > 1000:  # Long interaction
            return 'Precise'
        else:
            return 'Quick'
    
    def _detect_fallback_anomaly(self, group: pd.DataFrame) -> str:
        """Detect anomalies using simple flag analysis."""
        if 'flags' in group.columns:
            flags_str = str(group['flags'].iloc[0])
            if flags_str and flags_str != 'nan' and flags_str != '':
                return self._classify_anomaly_from_flags(flags_str)
        return 'None'
    
    def _determine_fallback_suitability(self, quality_score: int, interaction_type: str, anomaly_flag: str) -> str:
        """Determine research suitability using simple rules."""
        if quality_score >= 80 and anomaly_flag == 'None':
            return 'All'
        elif quality_score >= 60:
            if anomaly_flag == 'None':
                return 'Timing,Spatial'
            else:
                return 'Behavioral'
        elif quality_score >= 40:
            return 'Behavioral'
        else:
            return 'Limited'
