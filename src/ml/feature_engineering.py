#!/usr/bin/env python3
"""
Feature Engineering for Touch Data Cleaning

This module provides comprehensive feature extraction and engineering
for touch interaction data, specifically designed for Coloring touchdata.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from scipy import signal
from scipy.spatial.distance import euclidean
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class TouchFeatureEngineer:
    """
    Advanced feature engineering for touch interaction data.
    Extracts temporal, spatial, and behavioral features for ML models.
    """

    def __init__(self):
        self.feature_names = []
        self.scaler_params = {}

    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based features from touch sequences."""
        logger.info("Extracting temporal features...")

        # Ensure data is sorted by Touchdata_id and event_index
        if 'Touchdata_id' in df.columns and 'event_index' in df.columns:
            df = df.sort_values(['Touchdata_id', 'event_index'])
            groupby_cols = ['Touchdata_id']
        else:
            df = df.sort_values(['fingerId', 'time'])
            groupby_cols = ['fingerId']

        # Time differences
        df['time_diff'] = df.groupby(groupby_cols)['time'].diff().fillna(0)
        df['time_diff_normalized'] = df.groupby(groupby_cols)['time_diff'].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )

        # Cumulative time
        df['cumulative_time'] = df.groupby(groupby_cols)['time'].cumsum()

        # Time acceleration (second derivative)
        df['time_acceleration'] = df.groupby(groupby_cols)['time_diff'].diff().fillna(0)

        # Sequence position features
        df['sequence_position'] = df.groupby(groupby_cols).cumcount()
        df['sequence_length'] = df.groupby(groupby_cols)['sequence_position'].transform('max') + 1
        df['position_ratio'] = df['sequence_position'] / (df['sequence_length'] + 1e-8)

        # Time-based rhythm features
        df['time_rhythm_consistency'] = df.groupby(groupby_cols)['time_diff'].transform(
            lambda x: 1 / (x.std() + 1e-8)
        )

        return df

    def extract_spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract spatial movement features."""
        logger.info("Extracting spatial features...")

        if 'Touchdata_id' in df.columns:
            groupby_cols = ['Touchdata_id']
        else:
            groupby_cols = ['fingerId']

        # Basic spatial differences
        df['x_diff'] = df.groupby(groupby_cols)['x'].diff().fillna(0)
        df['y_diff'] = df.groupby(groupby_cols)['y'].diff().fillna(0)

        # Distance and velocity
        df['distance'] = np.sqrt(df['x_diff']**2 + df['y_diff']**2)
        df['velocity'] = df['distance'] / (df['time_diff'] + 1e-8)
        df['velocity'] = df['velocity'].replace([np.inf, -np.inf], 0)

        # Acceleration
        df['acceleration'] = df.groupby(groupby_cols)['velocity'].diff().fillna(0)

        # Direction and angle features
        df['direction_angle'] = np.arctan2(df['y_diff'], df['x_diff'])
        df['direction_change'] = df.groupby(groupby_cols)['direction_angle'].diff().fillna(0)

        # Cumulative distance
        df['cumulative_distance'] = df.groupby(groupby_cols)['distance'].cumsum()

        # Spatial smoothness
        df['spatial_smoothness'] = df.groupby(groupby_cols)['direction_change'].transform(
            lambda x: 1 / (x.abs().mean() + 1e-8)
        )

        # Bounding box features
        df['bbox_width'] = df.groupby(groupby_cols)['x'].transform(lambda x: x.max() - x.min())
        df['bbox_height'] = df.groupby(groupby_cols)['y'].transform(lambda x: x.max() - x.min())
        df['bbox_area'] = df['bbox_width'] * df['bbox_height']

        return df

    def extract_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract behavioral and interaction features."""
        logger.info("Extracting behavioral features...")

        if 'Touchdata_id' in df.columns:
            groupby_cols = ['Touchdata_id']
        else:
            groupby_cols = ['fingerId']

        # Touch phase transitions
        df['phase_transition'] = df.groupby(groupby_cols)['touchPhase'].shift() + '_to_' + df['touchPhase']
        df['phase_transition'] = df['phase_transition'].fillna('start_' + df['touchPhase'])

        # Pressure/completion features (if available)
        if 'completionPerc' in df.columns:
            df['completion_rate'] = df.groupby(groupby_cols)['completionPerc'].diff().fillna(0)
            df['completion_acceleration'] = df.groupby(groupby_cols)['completion_rate'].diff().fillna(0)

        # Accelerometer features (if available)
        if all(col in df.columns for col in ['accx', 'accy', 'accz']):
            df['acc_magnitude'] = np.sqrt(df['accx']**2 + df['accy']**2 + df['accz']**2)
            df['acc_change'] = df.groupby(groupby_cols)['acc_magnitude'].diff().fillna(0)

        # Zone transition features (if available)
        if 'zone' in df.columns:
            df['zone_change'] = (df.groupby(groupby_cols)['zone'].shift() != df['zone']).astype(int)
            df['zone_stability'] = df.groupby(groupby_cols)['zone_change'].transform('sum')

        # Color change features (if available)
        if 'color' in df.columns:
            df['color_change'] = (df.groupby(groupby_cols)['color'].shift() != df['color']).astype(int)

        return df

    def extract_sequence_quality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features that indicate sequence quality and validity."""
        logger.info("Extracting sequence quality features...")

        if 'Touchdata_id' in df.columns:
            groupby_cols = ['Touchdata_id']
        else:
            groupby_cols = ['fingerId']

        # Sequence completeness indicators
        df['has_began'] = df.groupby(groupby_cols)['touchPhase'].transform(
            lambda x: (x == 'Began').any()
        ).astype(int)

        df['has_ended'] = df.groupby(groupby_cols)['touchPhase'].transform(
            lambda x: (x == 'Ended').any()
        ).astype(int)

        df['has_canceled'] = df.groupby(groupby_cols)['touchPhase'].transform(
            lambda x: (x == 'Canceled').any()
        ).astype(int)

        # Sequence pattern validity - initialize column first
        df['sequence_valid_pattern'] = 0

        # Apply pattern checking to each group
        for group_id, group in df.groupby(groupby_cols):
            pattern_values = self._check_sequence_pattern(group)
            df.loc[group.index, 'sequence_valid_pattern'] = pattern_values

        # Outlier indicators (using std instead of deprecated mad)
        df['velocity_outlier'] = df.groupby(groupby_cols)['velocity'].transform(
            lambda x: (np.abs(x - x.median()) > 3 * x.std()).astype(int) if x.std() > 0 else 0
        )

        df['distance_outlier'] = df.groupby(groupby_cols)['distance'].transform(
            lambda x: (np.abs(x - x.median()) > 3 * x.std()).astype(int) if x.std() > 0 else 0
        )

        # Temporal consistency
        df['time_gap_outlier'] = df.groupby(groupby_cols)['time_diff'].transform(
            lambda x: (x > x.quantile(0.95) * 2).astype(int)
        )

        return df

    def _check_sequence_pattern(self, group: pd.DataFrame) -> np.ndarray:
        """Check if a sequence follows the valid Coloring pattern."""
        if 'event_index' in group.columns:
            group = group.sort_values('event_index')
        else:
            group = group.sort_values('time')

        phases = group['touchPhase'].tolist()

        # Valid pattern: Began → (Moved/Stationary)* → (optional Canceled) → Ended
        if len(phases) < 2:
            return np.array([0] * len(group))

        if phases[0] != 'Began' or phases[-1] != 'Ended':
            return np.array([0] * len(group))

        # Check middle phases
        middle_phases = phases[1:-1]
        valid_middle = all(phase in ['Moved', 'Stationary', 'Canceled'] for phase in middle_phases)

        # Check for at most one Canceled
        canceled_count = middle_phases.count('Canceled')

        is_valid = valid_middle and canceled_count <= 1
        return np.array([1 if is_valid else 0] * len(group))

    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract all feature types and return enhanced DataFrame."""
        logger.info("Starting comprehensive feature extraction...")

        # Make a copy to avoid modifying original
        df_features = df.copy()

        # Extract all feature types
        df_features = self.extract_temporal_features(df_features)
        df_features = self.extract_spatial_features(df_features)
        df_features = self.extract_behavioral_features(df_features)
        df_features = self.extract_sequence_quality_features(df_features)

        # Store feature names for later use
        original_cols = set(df.columns)
        new_cols = set(df_features.columns)
        self.feature_names = list(new_cols - original_cols)

        logger.info(f"Extracted {len(self.feature_names)} new features")

        return df_features

    def get_feature_matrix(self, df: pd.DataFrame, feature_subset: Optional[List[str]] = None) -> np.ndarray:
        """Get numerical feature matrix for ML models."""
        if feature_subset is None:
            feature_subset = self.feature_names

        # Select only numerical features that exist
        available_features = [f for f in feature_subset if f in df.columns]

        if not available_features:
            logger.warning("No features available for feature matrix")
            return np.array([])

        feature_matrix = df[available_features].select_dtypes(include=[np.number]).fillna(0)

        # Replace infinite values
        feature_matrix = feature_matrix.replace([np.inf, -np.inf], 0)

        return feature_matrix.values
