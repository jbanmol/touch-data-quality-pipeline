#!/usr/bin/env python3
"""
ML-Based Metadata Enhancement for Touch Data

This module provides ML-driven quality assessment and behavioral analysis
without modifying original coordinate data. Adds rich metadata and insights
to help researchers make informed decisions about data usage.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import json
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class TouchDataMetadataEnhancer:
    """
    Advanced ML-based metadata enhancement for touch interaction data.
    Provides quality assessment, behavioral analysis, and usage recommendations
    without modifying original data.
    """
    
    def __init__(self):
        self.quality_models = {}
        self.behavioral_classifiers = {}
        self.pattern_templates = {}
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def analyze_sequence_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze sequence quality and add quality metadata.
        
        Args:
            df: DataFrame with touch data
            
        Returns:
            DataFrame with quality metadata added
        """
        logger.info("Analyzing sequence quality...")
        
        # Initialize quality columns
        df['ml_quality_score'] = 0.0
        df['quality_tier'] = 'unknown'
        df['sequence_completeness'] = 0.0
        df['temporal_consistency'] = 0.0
        df['spatial_consistency'] = 0.0
        
        # Group by sequence identifier
        if 'Touchdata_id' in df.columns:
            groupby_col = 'Touchdata_id'
            sort_cols = ['Touchdata_id', 'event_index']
        else:
            groupby_col = 'fingerId'
            sort_cols = ['fingerId', 'time']
        
        df = df.sort_values(sort_cols)
        
        for seq_id, group in df.groupby(groupby_col):
            # Calculate sequence-level quality metrics
            quality_metrics = self._calculate_sequence_quality_metrics(group)
            
            # Update DataFrame with quality scores
            mask = df[groupby_col] == seq_id
            df.loc[mask, 'ml_quality_score'] = quality_metrics['overall_score']
            df.loc[mask, 'quality_tier'] = quality_metrics['tier']
            df.loc[mask, 'sequence_completeness'] = quality_metrics['completeness']
            df.loc[mask, 'temporal_consistency'] = quality_metrics['temporal_consistency']
            df.loc[mask, 'spatial_consistency'] = quality_metrics['spatial_consistency']
        
        return df
    
    def _calculate_sequence_quality_metrics(self, sequence: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive quality metrics for a single sequence."""
        metrics = {
            'overall_score': 0.0,
            'tier': 'low',
            'completeness': 0.0,
            'temporal_consistency': 0.0,
            'spatial_consistency': 0.0
        }
        
        if len(sequence) < 2:
            return metrics
        
        # 1. Sequence Completeness Score
        phases = sequence['touchPhase'].tolist()
        completeness_score = self._assess_sequence_completeness(phases)
        metrics['completeness'] = completeness_score
        
        # 2. Temporal Consistency Score
        if 'time' in sequence.columns:
            time_diffs = sequence['time'].diff().dropna()
            if len(time_diffs) > 0:
                # Penalize large gaps and negative time differences
                temporal_score = 1.0 - min(1.0, (time_diffs.std() / time_diffs.mean()) if time_diffs.mean() > 0 else 1.0)
                temporal_score *= 0.0 if (time_diffs < 0).any() else 1.0  # Penalize time reversals
                metrics['temporal_consistency'] = max(0.0, temporal_score)
        
        # 3. Spatial Consistency Score
        if 'x' in sequence.columns and 'y' in sequence.columns:
            spatial_score = self._assess_spatial_consistency(sequence)
            metrics['spatial_consistency'] = spatial_score
        
        # 4. Overall Quality Score (weighted combination)
        weights = {'completeness': 0.4, 'temporal': 0.3, 'spatial': 0.3}
        overall = (
            weights['completeness'] * metrics['completeness'] +
            weights['temporal'] * metrics['temporal_consistency'] +
            weights['spatial'] * metrics['spatial_consistency']
        )
        metrics['overall_score'] = overall
        
        # 5. Quality Tier Classification
        if overall >= 0.8:
            metrics['tier'] = 'high'
        elif overall >= 0.5:
            metrics['tier'] = 'medium'
        else:
            metrics['tier'] = 'low'
        
        return metrics
    
    def _assess_sequence_completeness(self, phases: List[str]) -> float:
        """Assess how complete a touch sequence is."""
        if len(phases) < 2:
            return 0.0
        
        score = 0.0
        
        # Check for proper start
        if phases[0] == 'Began':
            score += 0.4
        
        # Check for proper end
        if phases[-1] == 'Ended':
            score += 0.4
        elif phases[-1] == 'Canceled':
            score += 0.2  # Canceled is valid but less complete
        
        # Check for valid middle phases
        middle_phases = phases[1:-1] if len(phases) > 2 else []
        valid_middle = all(phase in ['Moved', 'Stationary', 'Canceled'] for phase in middle_phases)
        if valid_middle:
            score += 0.2
        
        return min(1.0, score)
    
    def _assess_spatial_consistency(self, sequence: pd.DataFrame) -> float:
        """Assess spatial movement consistency."""
        if len(sequence) < 3:
            return 1.0
        
        # Calculate movement distances
        x_diff = sequence['x'].diff().dropna()
        y_diff = sequence['y'].diff().dropna()
        distances = np.sqrt(x_diff**2 + y_diff**2)
        
        if len(distances) == 0:
            return 1.0
        
        # Assess consistency using coefficient of variation
        mean_dist = distances.mean()
        std_dist = distances.std()
        
        if mean_dist == 0:
            return 1.0 if std_dist == 0 else 0.0
        
        cv = std_dist / mean_dist
        consistency_score = max(0.0, 1.0 - min(1.0, cv / 2.0))  # Normalize CV
        
        return consistency_score
    
    def classify_behavioral_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify behavioral patterns in touch sequences.
        
        Args:
            df: DataFrame with touch data
            
        Returns:
            DataFrame with behavioral classification metadata
        """
        logger.info("Classifying behavioral patterns...")
        
        # Initialize behavioral columns
        df['behavioral_pattern'] = 'unknown'
        df['interaction_style'] = 'unknown'
        df['user_intent_confidence'] = 0.0
        df['movement_type'] = 'unknown'
        
        # Group by sequence identifier
        if 'Touchdata_id' in df.columns:
            groupby_col = 'Touchdata_id'
        else:
            groupby_col = 'fingerId'
        
        for seq_id, group in df.groupby(groupby_col):
            # Classify this sequence
            behavioral_info = self._classify_sequence_behavior(group)
            
            # Update DataFrame
            mask = df[groupby_col] == seq_id
            df.loc[mask, 'behavioral_pattern'] = behavioral_info['pattern']
            df.loc[mask, 'interaction_style'] = behavioral_info['style']
            df.loc[mask, 'user_intent_confidence'] = behavioral_info['intent_confidence']
            df.loc[mask, 'movement_type'] = behavioral_info['movement_type']
        
        return df
    
    def _classify_sequence_behavior(self, sequence: pd.DataFrame) -> Dict[str, Any]:
        """Classify behavioral patterns for a single sequence."""
        behavior = {
            'pattern': 'unknown',
            'style': 'unknown',
            'intent_confidence': 0.0,
            'movement_type': 'unknown'
        }
        
        if len(sequence) < 2:
            return behavior
        
        # Analyze movement characteristics
        if 'x' in sequence.columns and 'y' in sequence.columns:
            movement_analysis = self._analyze_movement_characteristics(sequence)
            behavior.update(movement_analysis)
        
        # Analyze touch phases
        phases = sequence['touchPhase'].tolist()
        phase_analysis = self._analyze_phase_patterns(phases)
        behavior.update(phase_analysis)
        
        # Analyze temporal patterns
        if 'time' in sequence.columns:
            temporal_analysis = self._analyze_temporal_patterns(sequence)
            behavior.update(temporal_analysis)
        
        return behavior
    
    def _analyze_movement_characteristics(self, sequence: pd.DataFrame) -> Dict[str, Any]:
        """Analyze spatial movement characteristics."""
        analysis = {}
        
        if len(sequence) < 3:
            analysis['movement_type'] = 'minimal'
            return analysis
        
        # Calculate movement metrics
        x_diff = sequence['x'].diff().dropna()
        y_diff = sequence['y'].diff().dropna()
        distances = np.sqrt(x_diff**2 + y_diff**2)
        
        total_distance = distances.sum()
        max_distance = distances.max() if len(distances) > 0 else 0
        mean_distance = distances.mean() if len(distances) > 0 else 0
        
        # Classify movement type
        if total_distance < 10:
            analysis['movement_type'] = 'stationary'
        elif max_distance > mean_distance * 5:
            analysis['movement_type'] = 'erratic'
        elif distances.std() < mean_distance * 0.5:
            analysis['movement_type'] = 'smooth'
        else:
            analysis['movement_type'] = 'variable'
        
        return analysis
    
    def _analyze_phase_patterns(self, phases: List[str]) -> Dict[str, Any]:
        """Analyze touch phase patterns."""
        analysis = {}
        
        # Determine interaction style based on phase sequence
        if len(phases) < 2:
            analysis['style'] = 'incomplete'
            analysis['intent_confidence'] = 0.1
        elif phases[0] == 'Began' and phases[-1] == 'Ended':
            analysis['style'] = 'deliberate'
            analysis['intent_confidence'] = 0.9
        elif 'Canceled' in phases:
            analysis['style'] = 'interrupted'
            analysis['intent_confidence'] = 0.6
        else:
            analysis['style'] = 'irregular'
            analysis['intent_confidence'] = 0.3
        
        # Determine behavioral pattern
        if len(phases) == 2 and phases == ['Began', 'Ended']:
            analysis['pattern'] = 'tap'
        elif 'Moved' in phases and len(phases) > 3:
            analysis['pattern'] = 'drag'
        elif 'Stationary' in phases:
            analysis['pattern'] = 'hold'
        else:
            analysis['pattern'] = 'complex'
        
        return analysis
    
    def _analyze_temporal_patterns(self, sequence: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal interaction patterns."""
        analysis = {}
        
        if 'time' in sequence.columns and len(sequence) > 1:
            duration = sequence['time'].max() - sequence['time'].min()
            time_diffs = sequence['time'].diff().dropna()
            
            # Classify based on duration and timing
            if duration < 100:  # Less than 100ms
                analysis['temporal_pattern'] = 'quick'
            elif duration > 2000:  # More than 2 seconds
                analysis['temporal_pattern'] = 'slow'
            else:
                analysis['temporal_pattern'] = 'normal'
        
        return analysis
    
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies and add anomaly metadata.
        
        Args:
            df: DataFrame with touch data
            
        Returns:
            DataFrame with anomaly detection metadata
        """
        logger.info("Detecting anomalies...")
        
        # Initialize anomaly columns
        df['anomaly_score'] = 0.0
        df['anomaly_type'] = 'none'
        df['anomaly_confidence'] = 0.0
        
        # Prepare features for anomaly detection
        feature_cols = []
        if 'x' in df.columns and 'y' in df.columns:
            feature_cols.extend(['x', 'y'])
        if 'time' in df.columns:
            feature_cols.append('time')
        if 'completionPerc' in df.columns:
            feature_cols.append('completionPerc')
        
        if len(feature_cols) < 2:
            logger.warning("Insufficient features for anomaly detection")
            return df
        
        # Extract features and detect anomalies
        features = df[feature_cols].fillna(0)
        
        # Use Isolation Forest for anomaly detection
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_scores = iso_forest.fit_predict(features)
        anomaly_scores_normalized = iso_forest.score_samples(features)
        
        # Update DataFrame with anomaly information
        df['anomaly_score'] = -anomaly_scores_normalized  # Convert to positive scores
        df['anomaly_type'] = ['outlier' if score == -1 else 'normal' for score in anomaly_scores]
        df['anomaly_confidence'] = np.abs(anomaly_scores_normalized)
        
        return df
