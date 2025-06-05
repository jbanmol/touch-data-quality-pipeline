#!/usr/bin/env python3
"""
Enhanced ML-Based Data Flagging System for Touch Data

This module provides a comprehensive ML-based flagging system that:
1. Tests and compares multiple traditional ML algorithms
2. Uses advanced feature engineering and statistical methods
3. Provides enhanced quality assessment and anomaly detection
4. Integrates seamlessly with existing validation pipeline
5. Generates the 4 consolidated ML metadata columns

NO neural networks - uses only traditional ML + feature engineering approaches.
"""

import pandas as pd
import numpy as np
import logging
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score
from scipy import stats
from scipy.spatial.distance import euclidean
import joblib
import os

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class EnhancedMLFlaggingSystem:
    """
    Comprehensive ML-based flagging system using traditional ML approaches.
    Tests multiple algorithms and selects the best performing ones.
    """
    
    def __init__(self, model_dir: str = "ML/models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Algorithm configurations
        self.anomaly_algorithms = {
            'isolation_forest': IsolationForest(contamination=0.1, random_state=42),
            'lof': LocalOutlierFactor(n_neighbors=20, contamination=0.1),
            'one_class_svm': OneClassSVM(gamma='scale', nu=0.1),
        }
        
        self.clustering_algorithms = {
            'kmeans': KMeans(n_clusters=4, random_state=42),
            'dbscan': DBSCAN(eps=0.5, min_samples=5),
            'gaussian_mixture': GaussianMixture(n_components=4, random_state=42)
        }
        
        # Scalers for different feature types
        self.scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler()
        }
        
        # Best performing algorithms (will be determined through comparison)
        self.best_anomaly_algorithm = None
        self.best_clustering_algorithm = None
        self.best_scaler = None
        
        # Feature importance tracking
        self.feature_importance = {}
        
        logger.info("Enhanced ML Flagging System initialized")
    
    def extract_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract comprehensive features for ML analysis.
        
        Args:
            df: DataFrame with touch data
            
        Returns:
            DataFrame with advanced features
        """
        logger.info("Extracting advanced features for ML analysis...")
        
        # Make a copy to avoid modifying original
        df_features = df.copy()
        
        # Determine grouping column
        if 'Touchdata_id' in df.columns:
            groupby_col = 'Touchdata_id'
            sort_cols = ['Touchdata_id', 'event_index']
        else:
            groupby_col = 'fingerId'
            sort_cols = ['fingerId', 'time']
        
        df_features = df_features.sort_values(sort_cols)
        
        # Extract features by sequence
        for seq_id, group in df_features.groupby(groupby_col):
            mask = df_features[groupby_col] == seq_id
            features = self._extract_sequence_features(group)
            
            # Apply features to all rows in the sequence
            for feature_name, feature_value in features.items():
                df_features.loc[mask, feature_name] = feature_value
        
        logger.info(f"Extracted {len([c for c in df_features.columns if c not in df.columns])} new features")
        return df_features
    
    def _extract_sequence_features(self, sequence: pd.DataFrame) -> Dict[str, float]:
        """Extract comprehensive features for a single sequence."""
        features = {}
        
        # Basic sequence properties
        features['sequence_length'] = len(sequence)
        features['sequence_duration'] = sequence['time'].max() - sequence['time'].min() if 'time' in sequence.columns else 0
        
        # Temporal features
        if 'time' in sequence.columns and len(sequence) > 1:
            time_diffs = sequence['time'].diff().dropna()
            features['mean_time_interval'] = time_diffs.mean()
            features['std_time_interval'] = time_diffs.std()
            features['time_interval_cv'] = features['std_time_interval'] / (features['mean_time_interval'] + 1e-8)
            features['max_time_gap'] = time_diffs.max()
            features['time_regularity'] = 1.0 / (1.0 + features['time_interval_cv'])
        
        # Spatial features
        if 'x' in sequence.columns and 'y' in sequence.columns and len(sequence) > 1:
            # Movement distances
            x_diffs = sequence['x'].diff().dropna()
            y_diffs = sequence['y'].diff().dropna()
            distances = np.sqrt(x_diffs**2 + y_diffs**2)
            
            features['total_distance'] = distances.sum()
            features['mean_distance'] = distances.mean()
            features['std_distance'] = distances.std()
            features['max_distance'] = distances.max()
            features['distance_cv'] = features['std_distance'] / (features['mean_distance'] + 1e-8)
            
            # Spatial spread
            features['x_range'] = sequence['x'].max() - sequence['x'].min()
            features['y_range'] = sequence['y'].max() - sequence['y'].min()
            features['spatial_spread'] = np.sqrt(features['x_range']**2 + features['y_range']**2)
            
            # Trajectory smoothness
            if len(distances) > 2:
                features['trajectory_smoothness'] = 1.0 / (1.0 + distances.std())
            
            # Direction consistency
            if len(x_diffs) > 1:
                angles = np.arctan2(y_diffs, x_diffs)
                angle_diffs = np.diff(angles)
                # Handle angle wrapping
                angle_diffs = np.abs(np.mod(angle_diffs + np.pi, 2*np.pi) - np.pi)
                features['direction_consistency'] = 1.0 / (1.0 + angle_diffs.std())
        
        # Touch phase features
        if 'touchPhase' in sequence.columns:
            phase_counts = sequence['touchPhase'].value_counts()
            features['phase_diversity'] = len(phase_counts)
            features['began_count'] = phase_counts.get('Began', 0)
            features['ended_count'] = phase_counts.get('Ended', 0)
            features['moved_count'] = phase_counts.get('Moved', 0)
            features['stationary_count'] = phase_counts.get('Stationary', 0)
            features['canceled_count'] = phase_counts.get('Canceled', 0)
            
            # Phase transition patterns
            transitions = []
            for i in range(len(sequence) - 1):
                transitions.append(f"{sequence.iloc[i]['touchPhase']}_to_{sequence.iloc[i+1]['touchPhase']}")
            features['unique_transitions'] = len(set(transitions))
        
        # Completion features
        if 'completionPerc' in sequence.columns:
            completion = sequence['completionPerc']
            features['completion_range'] = completion.max() - completion.min()
            features['completion_rate'] = features['completion_range'] / (features['sequence_duration'] + 1e-8)
            features['completion_consistency'] = 1.0 / (1.0 + completion.diff().std())
        
        # Zone features
        if 'zone' in sequence.columns:
            zone_counts = sequence['zone'].value_counts()
            features['zone_diversity'] = len(zone_counts)
            features['zone_changes'] = (sequence['zone'] != sequence['zone'].shift()).sum()
        
        # Statistical features for numerical columns
        numerical_cols = ['x', 'y', 'time', 'completionPerc']
        for col in numerical_cols:
            if col in sequence.columns:
                values = sequence[col].dropna()
                if len(values) > 0:
                    features[f'{col}_mean'] = values.mean()
                    features[f'{col}_std'] = values.std()
                    features[f'{col}_skew'] = stats.skew(values)
                    features[f'{col}_kurtosis'] = stats.kurtosis(values)
                    features[f'{col}_q25'] = values.quantile(0.25)
                    features[f'{col}_q75'] = values.quantile(0.75)
                    features[f'{col}_iqr'] = features[f'{col}_q75'] - features[f'{col}_q25']
        
        return features
    
    def compare_algorithms(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Compare different ML algorithms and select the best performing ones.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Dictionary with algorithm performance metrics
        """
        logger.info("Comparing ML algorithms for optimal performance...")
        
        # Prepare feature matrix
        feature_cols = [col for col in df.columns if col not in [
            'Touchdata_id', 'event_index', 'x', 'y', 'time', 'touchPhase', 
            'fingerId', 'color', 'completionPerc', 'zone', 'flags'
        ]]
        
        if len(feature_cols) < 3:
            logger.warning("Insufficient features for algorithm comparison")
            return {}
        
        X = df[feature_cols].fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        results = {
            'anomaly_detection': {},
            'clustering': {},
            'scaling': {}
        }
        
        # Test different scalers
        scaler_scores = {}
        for scaler_name, scaler in self.scalers.items():
            try:
                X_scaled = scaler.fit_transform(X)
                # Use variance as a simple metric for scaler effectiveness
                scaler_scores[scaler_name] = np.var(X_scaled)
            except Exception as e:
                logger.warning(f"Scaler {scaler_name} failed: {e}")
                scaler_scores[scaler_name] = float('inf')
        
        # Select best scaler (lowest variance indicates better normalization)
        self.best_scaler = min(scaler_scores.keys(), key=lambda k: scaler_scores[k])
        X_scaled = self.scalers[self.best_scaler].fit_transform(X)
        results['scaling'] = scaler_scores
        
        # Test anomaly detection algorithms
        anomaly_scores = {}
        for algo_name, algorithm in self.anomaly_algorithms.items():
            try:
                if algo_name == 'lof':
                    # LOF doesn't have fit/predict, only fit_predict
                    outliers = algorithm.fit_predict(X_scaled)
                    # Calculate a simple score based on outlier ratio
                    outlier_ratio = np.sum(outliers == -1) / len(outliers)
                    anomaly_scores[algo_name] = 1.0 - abs(outlier_ratio - 0.1)  # Target 10% outliers
                else:
                    algorithm.fit(X_scaled)
                    outliers = algorithm.predict(X_scaled)
                    outlier_ratio = np.sum(outliers == -1) / len(outliers)
                    anomaly_scores[algo_name] = 1.0 - abs(outlier_ratio - 0.1)
            except Exception as e:
                logger.warning(f"Anomaly algorithm {algo_name} failed: {e}")
                anomaly_scores[algo_name] = 0.0
        
        # Select best anomaly detection algorithm
        self.best_anomaly_algorithm = max(anomaly_scores.keys(), key=lambda k: anomaly_scores[k])
        results['anomaly_detection'] = anomaly_scores
        
        # Test clustering algorithms
        clustering_scores = {}
        for algo_name, algorithm in self.clustering_algorithms.items():
            try:
                if algo_name == 'dbscan':
                    labels = algorithm.fit_predict(X_scaled)
                    # For DBSCAN, calculate score based on number of clusters and noise ratio
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    noise_ratio = np.sum(labels == -1) / len(labels)
                    clustering_scores[algo_name] = n_clusters * (1.0 - noise_ratio) if n_clusters > 1 else 0.0
                else:
                    labels = algorithm.fit_predict(X_scaled)
                    if len(set(labels)) > 1:
                        clustering_scores[algo_name] = silhouette_score(X_scaled, labels)
                    else:
                        clustering_scores[algo_name] = 0.0
            except Exception as e:
                logger.warning(f"Clustering algorithm {algo_name} failed: {e}")
                clustering_scores[algo_name] = 0.0
        
        # Select best clustering algorithm
        if clustering_scores:
            self.best_clustering_algorithm = max(clustering_scores.keys(), key=lambda k: clustering_scores[k])
        results['clustering'] = clustering_scores
        
        logger.info(f"Best algorithms selected - Anomaly: {self.best_anomaly_algorithm}, "
                   f"Clustering: {self.best_clustering_algorithm}, Scaler: {self.best_scaler}")

        return results

    def detect_anomalies_advanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced anomaly detection using the best performing algorithm.

        Args:
            df: DataFrame with features

        Returns:
            DataFrame with anomaly detection results
        """
        logger.info("Performing advanced anomaly detection...")

        # Prepare feature matrix
        feature_cols = [col for col in df.columns if col not in [
            'Touchdata_id', 'event_index', 'x', 'y', 'time', 'touchPhase',
            'fingerId', 'color', 'completionPerc', 'zone', 'flags'
        ]]

        if len(feature_cols) < 3:
            logger.warning("Insufficient features for anomaly detection")
            df['anomaly_score'] = 0.0
            df['anomaly_type'] = 'normal'
            df['anomaly_confidence'] = 0.0
            return df

        X = df[feature_cols].fillna(0)
        X = X.replace([np.inf, -np.inf], 0)

        # Scale features
        if self.best_scaler:
            X_scaled = self.scalers[self.best_scaler].fit_transform(X)
        else:
            X_scaled = X.values

        # Apply best anomaly detection algorithm
        if self.best_anomaly_algorithm:
            algorithm = self.anomaly_algorithms[self.best_anomaly_algorithm]

            if self.best_anomaly_algorithm == 'lof':
                outliers = algorithm.fit_predict(X_scaled)
                scores = algorithm.negative_outlier_factor_
                df['anomaly_score'] = -scores  # Convert to positive scores
            else:
                algorithm.fit(X_scaled)
                outliers = algorithm.predict(X_scaled)
                scores = algorithm.score_samples(X_scaled)
                df['anomaly_score'] = -scores  # Convert to positive scores

            df['anomaly_type'] = ['outlier' if o == -1 else 'normal' for o in outliers]
            df['anomaly_confidence'] = np.abs(df['anomaly_score'])
        else:
            # Fallback to statistical outlier detection
            df = self._statistical_outlier_detection(df, feature_cols)

        return df

    def _statistical_outlier_detection(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Fallback statistical outlier detection using Z-score and IQR methods."""
        logger.info("Using statistical outlier detection as fallback...")

        outlier_scores = []

        for _, row in df.iterrows():
            score = 0.0
            count = 0

            for col in feature_cols:
                if col in df.columns:
                    value = row[col]
                    if not np.isnan(value) and not np.isinf(value):
                        # Z-score method
                        col_mean = df[col].mean()
                        col_std = df[col].std()
                        if col_std > 0:
                            z_score = abs((value - col_mean) / col_std)
                            if z_score > 3:  # 3-sigma rule
                                score += z_score
                                count += 1

                        # IQR method
                        q1 = df[col].quantile(0.25)
                        q3 = df[col].quantile(0.75)
                        iqr = q3 - q1
                        if iqr > 0:
                            lower_bound = q1 - 1.5 * iqr
                            upper_bound = q3 + 1.5 * iqr
                            if value < lower_bound or value > upper_bound:
                                score += 1.0
                                count += 1

            outlier_scores.append(score / max(count, 1))

        df['anomaly_score'] = outlier_scores
        threshold = np.percentile(outlier_scores, 90)  # Top 10% as outliers
        df['anomaly_type'] = ['outlier' if score > threshold else 'normal' for score in outlier_scores]
        df['anomaly_confidence'] = df['anomaly_score'] / (max(outlier_scores) + 1e-8)

        return df

    def classify_behavioral_patterns_advanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced behavioral pattern classification using clustering and statistical analysis.

        Args:
            df: DataFrame with features

        Returns:
            DataFrame with behavioral classifications
        """
        logger.info("Performing advanced behavioral pattern classification...")

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
            mask = df[groupby_col] == seq_id

            # Analyze this sequence
            behavioral_info = self._analyze_sequence_behavior(group)

            # Apply to all rows in the sequence
            df.loc[mask, 'behavioral_pattern'] = behavioral_info['pattern']
            df.loc[mask, 'interaction_style'] = behavioral_info['style']
            df.loc[mask, 'user_intent_confidence'] = behavioral_info['intent_confidence']
            df.loc[mask, 'movement_type'] = behavioral_info['movement_type']

        return df

    def _analyze_sequence_behavior(self, sequence: pd.DataFrame) -> Dict[str, Union[str, float]]:
        """Analyze behavioral patterns for a single sequence."""
        analysis = {
            'pattern': 'unknown',
            'style': 'unknown',
            'intent_confidence': 0.5,
            'movement_type': 'unknown'
        }

        if len(sequence) < 2:
            return analysis

        # Analyze movement patterns
        if 'x' in sequence.columns and 'y' in sequence.columns:
            x_diff = sequence['x'].diff().dropna()
            y_diff = sequence['y'].diff().dropna()
            distances = np.sqrt(x_diff**2 + y_diff**2)

            total_distance = distances.sum()
            max_distance = distances.max() if len(distances) > 0 else 0
            mean_distance = distances.mean() if len(distances) > 0 else 0

            # Movement type classification
            if total_distance < 10:
                analysis['movement_type'] = 'stationary'
            elif max_distance > mean_distance * 5:
                analysis['movement_type'] = 'erratic'
            elif distances.std() < mean_distance * 0.5:
                analysis['movement_type'] = 'smooth'
            else:
                analysis['movement_type'] = 'variable'

        # Analyze timing patterns
        if 'time' in sequence.columns and len(sequence) > 2:
            time_diffs = sequence['time'].diff().dropna()
            duration = sequence['time'].max() - sequence['time'].min()

            # Interaction style based on timing
            if duration < 0.5:  # Very quick
                analysis['style'] = 'quick'
                analysis['intent_confidence'] = 0.7
            elif duration > 5.0:  # Very slow/deliberate
                analysis['style'] = 'hesitant'
                analysis['intent_confidence'] = 0.9
            elif time_diffs.std() / time_diffs.mean() < 0.3:  # Consistent timing
                analysis['style'] = 'precise'
                analysis['intent_confidence'] = 0.95
            else:  # Irregular timing
                analysis['style'] = 'erratic'
                analysis['intent_confidence'] = 0.4

        # Analyze touch phase patterns
        if 'touchPhase' in sequence.columns:
            phases = sequence['touchPhase'].tolist()

            # Pattern classification based on phase sequence
            if len(phases) >= 3 and phases[0] == 'Began' and phases[-1] == 'Ended':
                if 'Canceled' in phases:
                    analysis['pattern'] = 'interrupted'
                    analysis['intent_confidence'] *= 0.7
                elif all(p in ['Began', 'Moved', 'Stationary', 'Ended'] for p in phases):
                    analysis['pattern'] = 'complete'
                    analysis['intent_confidence'] *= 1.1
                else:
                    analysis['pattern'] = 'irregular'
                    analysis['intent_confidence'] *= 0.8
            else:
                analysis['pattern'] = 'incomplete'
                analysis['intent_confidence'] *= 0.5

        # Ensure intent confidence is within bounds
        analysis['intent_confidence'] = max(0.0, min(1.0, analysis['intent_confidence']))

        return analysis

    def calculate_quality_score_advanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate advanced quality scores using multiple quality indicators.

        Args:
            df: DataFrame with features and analysis results

        Returns:
            DataFrame with quality scores
        """
        logger.info("Calculating advanced quality scores...")

        # Group by sequence identifier
        if 'Touchdata_id' in df.columns:
            groupby_col = 'Touchdata_id'
        else:
            groupby_col = 'fingerId'

        for seq_id, group in df.groupby(groupby_col):
            mask = df[groupby_col] == seq_id

            # Calculate comprehensive quality score
            quality_score = self._calculate_sequence_quality(group)
            df.loc[mask, 'ml_quality_score'] = quality_score

        return df

    def _calculate_sequence_quality(self, sequence: pd.DataFrame) -> float:
        """Calculate quality score for a single sequence."""
        quality_components = {}

        # 1. Sequence completeness (40% weight)
        if 'touchPhase' in sequence.columns:
            phases = sequence['touchPhase'].tolist()
            if len(phases) >= 2 and phases[0] == 'Began' and phases[-1] == 'Ended':
                quality_components['completeness'] = 1.0
            elif 'Began' in phases or 'Ended' in phases:
                quality_components['completeness'] = 0.6
            else:
                quality_components['completeness'] = 0.2
        else:
            quality_components['completeness'] = 0.5

        # 2. Temporal consistency (30% weight)
        if 'time' in sequence.columns and len(sequence) > 2:
            time_diffs = sequence['time'].diff().dropna()
            if len(time_diffs) > 0:
                cv = time_diffs.std() / (time_diffs.mean() + 1e-8)
                quality_components['temporal'] = max(0.0, 1.0 - cv)

                # Penalize negative time differences
                if (time_diffs < 0).any():
                    quality_components['temporal'] *= 0.5
            else:
                quality_components['temporal'] = 0.5
        else:
            quality_components['temporal'] = 0.5

        # 3. Spatial consistency (20% weight)
        if 'x' in sequence.columns and 'y' in sequence.columns and len(sequence) > 2:
            x_diff = sequence['x'].diff().dropna()
            y_diff = sequence['y'].diff().dropna()
            distances = np.sqrt(x_diff**2 + y_diff**2)

            if len(distances) > 0:
                # Reward smooth movement
                distance_cv = distances.std() / (distances.mean() + 1e-8)
                quality_components['spatial'] = max(0.0, 1.0 - distance_cv * 0.5)

                # Penalize extreme jumps
                max_distance = distances.max()
                mean_distance = distances.mean()
                if max_distance > mean_distance * 10:
                    quality_components['spatial'] *= 0.7
            else:
                quality_components['spatial'] = 0.5
        else:
            quality_components['spatial'] = 0.5

        # 4. Anomaly penalty (10% weight)
        if 'anomaly_type' in sequence.columns:
            anomaly_ratio = (sequence['anomaly_type'] == 'outlier').sum() / len(sequence)
            quality_components['anomaly'] = max(0.0, 1.0 - anomaly_ratio * 2)
        else:
            quality_components['anomaly'] = 1.0

        # Calculate weighted quality score
        weights = {
            'completeness': 0.4,
            'temporal': 0.3,
            'spatial': 0.2,
            'anomaly': 0.1
        }

        quality_score = sum(
            weights[component] * score
            for component, score in quality_components.items()
        )

        # Convert to 0-100 scale
        return max(0, min(100, quality_score * 100))

    def generate_consolidated_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate the 4 consolidated ML metadata columns.

        Args:
            df: DataFrame with all ML analysis results

        Returns:
            DataFrame with consolidated metadata columns
        """
        logger.info("Generating consolidated ML metadata columns...")

        # Initialize consolidated columns
        df['quality_score'] = 0
        df['interaction_type'] = 'Unknown'
        df['anomaly_flag'] = 'None'
        df['research_suitability'] = 'Limited'

        # Group by sequence identifier
        if 'Touchdata_id' in df.columns:
            groupby_col = 'Touchdata_id'
        else:
            groupby_col = 'fingerId'

        for seq_id, group in df.groupby(groupby_col):
            mask = df[groupby_col] == seq_id

            # 1. Quality Score (0-100)
            quality_score = group['ml_quality_score'].iloc[0] if 'ml_quality_score' in group.columns else 50
            df.loc[mask, 'quality_score'] = int(quality_score)

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

    def _determine_interaction_type(self, group: pd.DataFrame) -> str:
        """Determine interaction type based on behavioral analysis."""
        if 'interaction_style' in group.columns:
            style = group['interaction_style'].iloc[0]
            if style in ['precise', 'quick', 'hesitant', 'erratic']:
                return style.capitalize()

        # Fallback analysis
        if 'time' in group.columns and len(group) > 1:
            duration = group['time'].max() - group['time'].min()
            if duration < 0.5:
                return 'Quick'
            elif duration > 3.0:
                return 'Hesitant'
            else:
                return 'Precise'

        return 'Unknown'

    def _determine_anomaly_flag(self, group: pd.DataFrame) -> str:
        """Determine anomaly flag based on anomaly detection results."""
        if 'anomaly_type' in group.columns:
            outlier_ratio = (group['anomaly_type'] == 'outlier').sum() / len(group)

            if outlier_ratio > 0.5:
                # Determine type of anomaly
                if 'touchPhase' in group.columns:
                    phases = group['touchPhase'].tolist()
                    if phases[0] != 'Began' or phases[-1] != 'Ended':
                        return 'Technical'

                if 'movement_type' in group.columns:
                    movement = group['movement_type'].iloc[0]
                    if movement == 'erratic':
                        return 'Behavioral'

                if 'x' in group.columns and 'y' in group.columns:
                    x_range = group['x'].max() - group['x'].min()
                    y_range = group['y'].max() - group['y'].min()
                    if x_range > 1000 or y_range > 1000:  # Large spatial jumps
                        return 'Spatial'

                return 'Behavioral'  # Default for outliers

        return 'None'

    def _determine_research_suitability(self, quality_score: float, interaction_type: str,
                                      anomaly_flag: str, group: pd.DataFrame) -> str:
        """Determine research suitability based on all factors."""
        suitability_tags = []

        # Quality-based suitability
        if quality_score >= 80:
            if anomaly_flag == 'None':
                suitability_tags.extend(['Timing', 'Spatial', 'Behavioral'])
            else:
                # High quality but with anomalies - still useful for specific analysis
                if anomaly_flag == 'Technical':
                    suitability_tags.append('Behavioral')
                elif anomaly_flag == 'Spatial':
                    suitability_tags.append('Timing')
                elif anomaly_flag == 'Behavioral':
                    suitability_tags.extend(['Timing', 'Spatial'])
        elif quality_score >= 60:
            # Medium quality - limited suitability
            if interaction_type == 'Precise':
                suitability_tags.append('Timing')
            if anomaly_flag == 'None':
                suitability_tags.append('Spatial')

        # Interaction type specific suitability
        if interaction_type == 'Precise' and quality_score >= 70:
            if 'Timing' not in suitability_tags:
                suitability_tags.append('Timing')

        if len(suitability_tags) >= 3:
            return 'All'
        elif len(suitability_tags) >= 1:
            return ','.join(sorted(suitability_tags))
        else:
            return 'Limited'

    def enhance_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main method to enhance DataFrame with ML-based flagging.

        Args:
            df: Input DataFrame with touch data

        Returns:
            DataFrame with enhanced ML metadata
        """
        logger.info("Starting enhanced ML-based data flagging...")

        try:
            # Step 1: Extract advanced features
            df_features = self.extract_advanced_features(df)

            # Step 2: Compare and select best algorithms
            algorithm_results = self.compare_algorithms(df_features)
            logger.info(f"Algorithm comparison results: {algorithm_results}")

            # Step 3: Apply advanced anomaly detection
            df_anomalies = self.detect_anomalies_advanced(df_features)

            # Step 4: Classify behavioral patterns
            df_behavioral = self.classify_behavioral_patterns_advanced(df_anomalies)

            # Step 5: Calculate quality scores
            df_quality = self.calculate_quality_score_advanced(df_behavioral)

            # Step 6: Generate consolidated metadata
            df_final = self.generate_consolidated_metadata(df_quality)

            logger.info("Enhanced ML-based data flagging completed successfully")
            return df_final

        except Exception as e:
            logger.error(f"Enhanced ML flagging failed: {e}")
            logger.info("Falling back to basic ML enhancement")
            return self._apply_fallback_enhancement(df)

    def _apply_fallback_enhancement(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback enhancement using basic statistical methods."""
        logger.info("Applying fallback ML enhancement...")

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

                # Basic quality score
                if len(phases) >= 2 and phases[0] == 'Began' and phases[-1] == 'Ended':
                    df.loc[mask, 'quality_score'] = 80
                    df.loc[mask, 'research_suitability'] = 'All'
                elif 'Began' in phases or 'Ended' in phases:
                    df.loc[mask, 'quality_score'] = 60
                    df.loc[mask, 'research_suitability'] = 'Spatial'
                else:
                    df.loc[mask, 'quality_score'] = 30

                # Basic interaction type
                if 'time' in group.columns and len(group) > 1:
                    duration = group['time'].max() - group['time'].min()
                    if duration < 0.5:
                        df.loc[mask, 'interaction_type'] = 'Quick'
                    elif duration > 3.0:
                        df.loc[mask, 'interaction_type'] = 'Hesitant'
                    else:
                        df.loc[mask, 'interaction_type'] = 'Precise'

        return df
