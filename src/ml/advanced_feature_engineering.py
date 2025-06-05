#!/usr/bin/env python3
"""
Advanced Feature Engineering for Touch Data Analysis

This module provides comprehensive feature extraction specifically designed
for touch interaction data, focusing on statistical, temporal, spatial,
and behavioral features that enhance ML model performance.
"""

import pandas as pd
import numpy as np
import logging
import warnings
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats, signal
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import LabelEncoder
import math

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class AdvancedTouchFeatureEngineer:
    """
    Advanced feature engineering for touch interaction data.
    Extracts comprehensive statistical, temporal, spatial, and behavioral features.
    """
    
    def __init__(self):
        self.feature_names = []
        self.label_encoders = {}
        self.feature_importance = {}
        
        logger.info("Advanced Touch Feature Engineer initialized")
    
    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all types of advanced features.
        
        Args:
            df: DataFrame with touch data
            
        Returns:
            DataFrame with comprehensive features
        """
        logger.info("Starting comprehensive advanced feature extraction...")
        
        # Make a copy to avoid modifying original
        df_features = df.copy()
        
        # Determine grouping strategy
        if 'Touchdata_id' in df.columns:
            groupby_col = 'Touchdata_id'
            sort_cols = ['Touchdata_id', 'event_index']
        else:
            groupby_col = 'fingerId'
            sort_cols = ['fingerId', 'time']
        
        df_features = df_features.sort_values(sort_cols)
        
        # Extract features by category
        df_features = self.extract_statistical_features(df_features)
        df_features = self.extract_temporal_features(df_features)
        df_features = self.extract_spatial_features(df_features)
        df_features = self.extract_behavioral_features(df_features)
        df_features = self.extract_sequence_features(df_features)
        df_features = self.extract_interaction_features(df_features)
        
        # Store feature names
        original_cols = set(df.columns)
        new_cols = set(df_features.columns)
        self.feature_names = list(new_cols - original_cols)
        
        logger.info(f"Extracted {len(self.feature_names)} advanced features")
        return df_features
    
    def extract_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract comprehensive statistical features."""
        logger.info("Extracting statistical features...")
        
        # Determine grouping column
        if 'Touchdata_id' in df.columns:
            groupby_col = 'Touchdata_id'
        else:
            groupby_col = 'fingerId'
        
        # Statistical features for numerical columns
        numerical_cols = ['x', 'y', 'time', 'completionPerc']
        
        for col in numerical_cols:
            if col in df.columns:
                # Basic statistics
                df[f'{col}_seq_mean'] = df.groupby(groupby_col)[col].transform('mean')
                df[f'{col}_seq_std'] = df.groupby(groupby_col)[col].transform('std')
                df[f'{col}_seq_min'] = df.groupby(groupby_col)[col].transform('min')
                df[f'{col}_seq_max'] = df.groupby(groupby_col)[col].transform('max')
                df[f'{col}_seq_range'] = df[f'{col}_seq_max'] - df[f'{col}_seq_min']
                
                # Advanced statistics
                df[f'{col}_seq_skew'] = df.groupby(groupby_col)[col].transform(lambda x: stats.skew(x.dropna()))
                df[f'{col}_seq_kurtosis'] = df.groupby(groupby_col)[col].transform(lambda x: stats.kurtosis(x.dropna()))
                
                # Percentiles
                df[f'{col}_seq_q25'] = df.groupby(groupby_col)[col].transform(lambda x: x.quantile(0.25))
                df[f'{col}_seq_q75'] = df.groupby(groupby_col)[col].transform(lambda x: x.quantile(0.75))
                df[f'{col}_seq_iqr'] = df[f'{col}_seq_q75'] - df[f'{col}_seq_q25']
                
                # Coefficient of variation
                df[f'{col}_seq_cv'] = df[f'{col}_seq_std'] / (df[f'{col}_seq_mean'] + 1e-8)
                
                # Z-scores within sequence
                df[f'{col}_z_score'] = (df[col] - df[f'{col}_seq_mean']) / (df[f'{col}_seq_std'] + 1e-8)
        
        return df
    
    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract advanced temporal features."""
        logger.info("Extracting temporal features...")
        
        if 'time' not in df.columns:
            logger.warning("No time column found for temporal features")
            return df
        
        # Determine grouping column
        if 'Touchdata_id' in df.columns:
            groupby_col = 'Touchdata_id'
        else:
            groupby_col = 'fingerId'
        
        # Basic temporal differences
        df['time_diff'] = df.groupby(groupby_col)['time'].diff().fillna(0)
        df['time_diff_normalized'] = df.groupby(groupby_col)['time_diff'].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )
        
        # Temporal acceleration
        df['time_acceleration'] = df.groupby(groupby_col)['time_diff'].diff().fillna(0)
        
        # Rhythm analysis
        df['rhythm_regularity'] = df.groupby(groupby_col)['time_diff'].transform(
            lambda x: 1.0 / (1.0 + x.std()) if len(x) > 1 else 1.0
        )
        
        # Pause detection
        df['is_pause'] = (df['time_diff'] > df.groupby(groupby_col)['time_diff'].transform('mean') * 2).astype(int)
        df['pause_count'] = df.groupby(groupby_col)['is_pause'].transform('sum')
        
        # Temporal patterns
        df['time_since_start'] = df.groupby(groupby_col)['time'].transform(lambda x: x - x.min())
        df['time_to_end'] = df.groupby(groupby_col)['time'].transform(lambda x: x.max() - x)
        df['relative_time_position'] = df['time_since_start'] / (df.groupby(groupby_col)['time'].transform(lambda x: x.max() - x.min()) + 1e-8)
        
        # Frequency domain features (if sequence is long enough)
        for seq_id, group in df.groupby(groupby_col):
            if len(group) >= 8:  # Minimum for meaningful FFT
                mask = df[groupby_col] == seq_id
                time_series = group['time_diff'].values
                
                # Remove DC component and apply window
                time_series = time_series - np.mean(time_series)
                if len(time_series) > 0:
                    # Apply Hamming window
                    windowed = time_series * signal.windows.hamming(len(time_series))
                    
                    # FFT
                    fft_vals = np.fft.fft(windowed)
                    power_spectrum = np.abs(fft_vals) ** 2
                    
                    # Extract frequency features
                    df.loc[mask, 'temporal_energy'] = np.sum(power_spectrum)
                    df.loc[mask, 'temporal_peak_freq'] = np.argmax(power_spectrum)
                    df.loc[mask, 'temporal_spectral_centroid'] = np.sum(np.arange(len(power_spectrum)) * power_spectrum) / (np.sum(power_spectrum) + 1e-8)
        
        # Fill NaN values for frequency features
        freq_features = ['temporal_energy', 'temporal_peak_freq', 'temporal_spectral_centroid']
        for feature in freq_features:
            if feature in df.columns:
                df[feature] = df[feature].fillna(0)
        
        return df
    
    def extract_spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract advanced spatial features."""
        logger.info("Extracting spatial features...")
        
        if 'x' not in df.columns or 'y' not in df.columns:
            logger.warning("No x,y columns found for spatial features")
            return df
        
        # Determine grouping column
        if 'Touchdata_id' in df.columns:
            groupby_col = 'Touchdata_id'
        else:
            groupby_col = 'fingerId'
        
        # Basic spatial differences
        df['x_diff'] = df.groupby(groupby_col)['x'].diff().fillna(0)
        df['y_diff'] = df.groupby(groupby_col)['y'].diff().fillna(0)
        
        # Distance and velocity
        df['distance'] = np.sqrt(df['x_diff']**2 + df['y_diff']**2)
        df['velocity'] = df['distance'] / (df['time_diff'] + 1e-8) if 'time_diff' in df.columns else df['distance']
        df['velocity'] = df['velocity'].replace([np.inf, -np.inf], 0)
        
        # Acceleration
        df['acceleration'] = df.groupby(groupby_col)['velocity'].diff().fillna(0)
        df['acceleration_magnitude'] = np.abs(df['acceleration'])
        
        # Direction and angle features
        df['direction_angle'] = np.arctan2(df['y_diff'], df['x_diff'])
        df['direction_change'] = df.groupby(groupby_col)['direction_angle'].diff().fillna(0)
        
        # Handle angle wrapping for direction change
        df['direction_change'] = np.where(
            df['direction_change'] > np.pi, 
            df['direction_change'] - 2*np.pi, 
            df['direction_change']
        )
        df['direction_change'] = np.where(
            df['direction_change'] < -np.pi, 
            df['direction_change'] + 2*np.pi, 
            df['direction_change']
        )
        
        df['direction_change_magnitude'] = np.abs(df['direction_change'])
        
        # Cumulative features
        df['cumulative_distance'] = df.groupby(groupby_col)['distance'].cumsum()
        df['cumulative_direction_change'] = df.groupby(groupby_col)['direction_change_magnitude'].cumsum()
        
        # Spatial spread and trajectory features
        df['x_range'] = df.groupby(groupby_col)['x'].transform(lambda x: x.max() - x.min())
        df['y_range'] = df.groupby(groupby_col)['y'].transform(lambda x: x.max() - x.min())
        df['spatial_spread'] = np.sqrt(df['x_range']**2 + df['y_range']**2)
        
        # Trajectory smoothness
        df['trajectory_smoothness'] = df.groupby(groupby_col)['distance'].transform(
            lambda x: 1.0 / (1.0 + x.std()) if len(x) > 1 else 1.0
        )
        
        # Direction consistency
        df['direction_consistency'] = df.groupby(groupby_col)['direction_change_magnitude'].transform(
            lambda x: 1.0 / (1.0 + x.std()) if len(x) > 1 else 1.0
        )
        
        # Spatial efficiency (straight line distance vs actual path)
        for seq_id, group in df.groupby(groupby_col):
            if len(group) >= 2:
                mask = df[groupby_col] == seq_id
                start_point = (group['x'].iloc[0], group['y'].iloc[0])
                end_point = (group['x'].iloc[-1], group['y'].iloc[-1])
                straight_distance = euclidean(start_point, end_point)
                total_distance = group['distance'].sum()
                
                efficiency = straight_distance / (total_distance + 1e-8)
                df.loc[mask, 'spatial_efficiency'] = efficiency
        
        # Fill NaN values
        df['spatial_efficiency'] = df['spatial_efficiency'].fillna(0)

        return df

    def extract_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract advanced behavioral features."""
        logger.info("Extracting behavioral features...")

        # Determine grouping column
        if 'Touchdata_id' in df.columns:
            groupby_col = 'Touchdata_id'
        else:
            groupby_col = 'fingerId'

        # Touch phase analysis
        if 'touchPhase' in df.columns:
            # Encode touch phases
            if 'touchPhase' not in self.label_encoders:
                self.label_encoders['touchPhase'] = LabelEncoder()
                # Fit on all possible phases
                all_phases = ['Began', 'Moved', 'Stationary', 'Ended', 'Canceled']
                self.label_encoders['touchPhase'].fit(all_phases)

            df['touchPhase_encoded'] = self.label_encoders['touchPhase'].transform(
                df['touchPhase'].fillna('Unknown')
            )

            # Phase transitions
            df['phase_transition'] = df.groupby(groupby_col)['touchPhase'].shift().fillna('start') + '_to_' + df['touchPhase']

            # Phase duration (approximate)
            if 'time' in df.columns:
                df['phase_duration'] = df.groupby([groupby_col, 'touchPhase'])['time'].transform(
                    lambda x: x.max() - x.min() if len(x) > 1 else 0
                )

        # Pressure/completion analysis
        if 'completionPerc' in df.columns:
            df['completion_rate'] = df.groupby(groupby_col)['completionPerc'].diff().fillna(0)
            df['completion_acceleration'] = df.groupby(groupby_col)['completion_rate'].diff().fillna(0)

            # Completion consistency
            df['completion_consistency'] = df.groupby(groupby_col)['completion_rate'].transform(
                lambda x: 1.0 / (1.0 + x.std()) if len(x) > 1 else 1.0
            )

        # Zone analysis
        if 'zone' in df.columns:
            # Encode zones
            if 'zone' not in self.label_encoders:
                self.label_encoders['zone'] = LabelEncoder()
                unique_zones = df['zone'].dropna().unique()
                self.label_encoders['zone'].fit(unique_zones)

            df['zone_encoded'] = self.label_encoders['zone'].transform(
                df['zone'].fillna('Unknown')
            )

            # Zone transitions
            df['zone_change'] = (df.groupby(groupby_col)['zone'].shift() != df['zone']).astype(int)
            df['zone_stability'] = df.groupby(groupby_col)['zone_change'].transform('sum')

        # Color analysis
        if 'color' in df.columns:
            # Encode colors
            if 'color' not in self.label_encoders:
                self.label_encoders['color'] = LabelEncoder()
                unique_colors = df['color'].dropna().unique()
                self.label_encoders['color'].fit(unique_colors)

            df['color_encoded'] = self.label_encoders['color'].transform(
                df['color'].fillna('Unknown')
            )

            # Color changes
            df['color_change'] = (df.groupby(groupby_col)['color'].shift() != df['color']).astype(int)

        return df

    def extract_sequence_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract sequence-level features."""
        logger.info("Extracting sequence features...")

        # Determine grouping column
        if 'Touchdata_id' in df.columns:
            groupby_col = 'Touchdata_id'
        else:
            groupby_col = 'fingerId'

        # Basic sequence properties
        df['sequence_length'] = df.groupby(groupby_col)[groupby_col].transform('count')

        if 'time' in df.columns:
            df['sequence_duration'] = df.groupby(groupby_col)['time'].transform(lambda x: x.max() - x.min())
            df['sequence_density'] = df['sequence_length'] / (df['sequence_duration'] + 1e-8)

        # Position within sequence
        df['sequence_position'] = df.groupby(groupby_col).cumcount()
        df['relative_sequence_position'] = df['sequence_position'] / (df['sequence_length'] - 1 + 1e-8)

        # Sequence completeness analysis
        if 'touchPhase' in df.columns:
            # Check for proper sequence structure
            df['has_began'] = df.groupby(groupby_col)['touchPhase'].transform(lambda x: ('Began' in x.values).astype(int))
            df['has_ended'] = df.groupby(groupby_col)['touchPhase'].transform(lambda x: ('Ended' in x.values).astype(int))
            df['has_canceled'] = df.groupby(groupby_col)['touchPhase'].transform(lambda x: ('Canceled' in x.values).astype(int))

            # Sequence pattern score
            df['sequence_pattern_score'] = (df['has_began'] + df['has_ended']) / 2.0
            df['sequence_pattern_score'] = df['sequence_pattern_score'] - df['has_canceled'] * 0.3  # Penalty for cancellation

        return df

    def extract_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract interaction-specific features."""
        logger.info("Extracting interaction features...")

        # Determine grouping column
        if 'Touchdata_id' in df.columns:
            groupby_col = 'Touchdata_id'
        else:
            groupby_col = 'fingerId'

        # Multi-touch analysis (if fingerId varies within sequences)
        if 'fingerId' in df.columns:
            df['unique_fingers'] = df.groupby(groupby_col)['fingerId'].transform('nunique')
            df['is_multitouch'] = (df['unique_fingers'] > 1).astype(int)

        # Interaction intensity
        if 'distance' in df.columns:
            df['interaction_intensity'] = df.groupby(groupby_col)['distance'].transform('sum')
            df['max_movement_burst'] = df.groupby(groupby_col)['distance'].transform('max')

        # Hesitation detection
        if 'velocity' in df.columns:
            # Detect sudden stops (low velocity after high velocity)
            df['velocity_drop'] = df.groupby(groupby_col)['velocity'].diff().fillna(0)
            df['hesitation_events'] = ((df['velocity'] < 0.1) & (df['velocity'].shift() > 1.0)).astype(int)
            df['hesitation_count'] = df.groupby(groupby_col)['hesitation_events'].transform('sum')

        # Precision indicators
        if 'direction_change_magnitude' in df.columns:
            df['precision_score'] = df.groupby(groupby_col)['direction_change_magnitude'].transform(
                lambda x: 1.0 / (1.0 + x.mean()) if len(x) > 0 else 1.0
            )

        # Interaction rhythm
        if 'time_diff' in df.columns:
            df['rhythm_score'] = df.groupby(groupby_col)['time_diff'].transform(
                lambda x: 1.0 / (1.0 + x.std()) if len(x) > 1 else 1.0
            )

        return df

    def get_feature_importance(self, df: pd.DataFrame, target_col: str = None) -> Dict[str, float]:
        """
        Calculate feature importance using statistical methods.

        Args:
            df: DataFrame with features
            target_col: Target column for supervised importance (optional)

        Returns:
            Dictionary with feature importance scores
        """
        logger.info("Calculating feature importance...")

        importance_scores = {}

        # Get numerical feature columns
        feature_cols = [col for col in self.feature_names if col in df.columns]
        numerical_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

        if len(numerical_features) == 0:
            logger.warning("No numerical features found for importance calculation")
            return importance_scores

        # Calculate variance-based importance
        for feature in numerical_features:
            values = df[feature].dropna()
            if len(values) > 1:
                # Normalize variance by mean to handle different scales
                variance_score = values.var() / (abs(values.mean()) + 1e-8)
                importance_scores[feature] = variance_score

        # If target column is provided, calculate correlation-based importance
        if target_col and target_col in df.columns:
            target_values = df[target_col].dropna()
            if len(target_values) > 1:
                for feature in numerical_features:
                    if feature != target_col:
                        feature_values = df[feature].dropna()
                        if len(feature_values) > 1:
                            # Calculate correlation
                            correlation = np.corrcoef(
                                df[[feature, target_col]].dropna()[feature],
                                df[[feature, target_col]].dropna()[target_col]
                            )[0, 1]

                            if not np.isnan(correlation):
                                # Combine variance and correlation scores
                                combined_score = (
                                    importance_scores.get(feature, 0) * 0.5 +
                                    abs(correlation) * 0.5
                                )
                                importance_scores[feature] = combined_score

        # Normalize scores
        if importance_scores:
            max_score = max(importance_scores.values())
            if max_score > 0:
                importance_scores = {k: v/max_score for k, v in importance_scores.items()}

        self.feature_importance = importance_scores
        logger.info(f"Calculated importance for {len(importance_scores)} features")

        return importance_scores

    def get_top_features(self, n_features: int = 20) -> List[str]:
        """Get top N most important features."""
        if not self.feature_importance:
            logger.warning("Feature importance not calculated yet")
            return self.feature_names[:n_features]

        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [feature for feature, _ in sorted_features[:n_features]]
