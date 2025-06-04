#!/usr/bin/env python3
"""
Simple ML-Based Touch Data Cleaner

A simplified version that works with basic dependencies and provides
core ML functionality for touch data enhancement without complex transfer learning.
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleTouchDataCleaner:
    """
    Simplified ML-based touch data cleaner that preserves all original data
    while adding quality assessments and behavioral insights.
    """

    def __init__(self):
        logger.info("Initialized Simple ML Touch Data Cleaner")

    def process_json_file(self, json_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a single JSON file and add ML-based metadata.

        Args:
            json_path: Path to input JSON file
            output_path: Path for output JSON file (optional)

        Returns:
            Dictionary with processing results and statistics
        """
        logger.info(f"Processing JSON file: {json_path}")

        try:
            # Load and parse JSON data
            with open(json_path, 'r') as f:
                data = json.load(f)

            # Convert to DataFrame for processing
            df = self._json_to_dataframe(data)

            if df.empty:
                logger.warning(f"No touch data found in {json_path}")
                return {'status': 'error', 'message': 'No touch data found'}

            # Apply ML-based cleaning and enhancement
            enhanced_df = self.clean_and_enhance_data(df)

            # Convert back to JSON format with enhancements
            enhanced_data = self._dataframe_to_json(enhanced_df, original_data=data)

            # Save enhanced data if output path provided
            if output_path:
                with open(output_path, 'w') as f:
                    json.dump(enhanced_data, f, indent=2)
                logger.info(f"Enhanced data saved to: {output_path}")

            # Generate processing statistics
            stats = self._generate_processing_stats(df, enhanced_df)

            return {
                'status': 'success',
                'input_file': json_path,
                'output_file': output_path,
                'statistics': stats,
                'enhanced_data': enhanced_data
            }

        except Exception as e:
            logger.error(f"Error processing {json_path}: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def clean_and_enhance_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply ML-based cleaning and enhancement to touch data.

        Args:
            df: DataFrame with touch data

        Returns:
            Enhanced DataFrame with ML metadata
        """
        logger.info("Applying ML-based data enhancement...")

        # Make a copy to preserve original
        enhanced_df = df.copy()

        # Step 1: Basic Feature Engineering
        enhanced_df = self._extract_basic_features(enhanced_df)

        # Step 2: Quality Assessment
        enhanced_df = self._assess_quality(enhanced_df)

        # Step 3: Behavioral Classification
        enhanced_df = self._classify_behavior(enhanced_df)

        # Step 4: Simple Anomaly Detection
        enhanced_df = self._detect_simple_anomalies(enhanced_df)

        # Step 5: Usage Recommendations
        enhanced_df = self._generate_recommendations(enhanced_df)

        logger.info("ML-based enhancement completed")
        return enhanced_df

    def _json_to_dataframe(self, json_data: Dict) -> pd.DataFrame:
        """Convert JSON touch data to DataFrame format."""
        try:
            touch_data = json_data.get('json', {}).get('touchData', {})

            all_entries = []
            for finger_id, entries in touch_data.items():
                for i, entry in enumerate(entries):
                    processed_entry = entry.copy()

                    # Add metadata fields
                    if 'fingerId' not in processed_entry:
                        processed_entry['fingerId'] = finger_id

                    # Add Touchdata_id and event_index if not present
                    if 'Touchdata_id' not in processed_entry:
                        processed_entry['Touchdata_id'] = finger_id
                    if 'event_index' not in processed_entry:
                        processed_entry['event_index'] = i

                    all_entries.append(processed_entry)

            return pd.DataFrame(all_entries)

        except Exception as e:
            logger.error(f"Error converting JSON to DataFrame: {e}")
            return pd.DataFrame()

    def _extract_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract basic features from touch data."""

        # Group by sequence identifier
        if 'Touchdata_id' in df.columns:
            groupby_col = 'Touchdata_id'
            sort_cols = ['Touchdata_id', 'event_index']
        else:
            groupby_col = 'fingerId'
            sort_cols = ['fingerId', 'time']

        df = df.sort_values(sort_cols)

        # Basic temporal features
        df['time_diff'] = df.groupby(groupby_col)['time'].diff().fillna(0)
        df['sequence_position'] = df.groupby(groupby_col).cumcount()
        df['sequence_length'] = df.groupby(groupby_col)['sequence_position'].transform('max') + 1

        # Basic spatial features
        if 'x' in df.columns and 'y' in df.columns:
            df['x_diff'] = df.groupby(groupby_col)['x'].diff().fillna(0)
            df['y_diff'] = df.groupby(groupby_col)['y'].diff().fillna(0)
            df['distance'] = np.sqrt(df['x_diff']**2 + df['y_diff']**2)
            df['cumulative_distance'] = df.groupby(groupby_col)['distance'].cumsum()

        return df

    def _assess_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assess sequence quality."""

        # Group by sequence identifier
        if 'Touchdata_id' in df.columns:
            groupby_col = 'Touchdata_id'
        else:
            groupby_col = 'fingerId'

        # Initialize quality columns
        df['ml_quality_score'] = 0.0
        df['quality_tier'] = 'unknown'
        df['sequence_completeness'] = 0.0

        for seq_id, group in df.groupby(groupby_col):
            # Calculate quality metrics
            quality_score = self._calculate_sequence_quality(group)

            # Assign quality tier
            if quality_score >= 0.8:
                tier = 'high'
            elif quality_score >= 0.5:
                tier = 'medium'
            else:
                tier = 'low'

            # Update DataFrame
            mask = df[groupby_col] == seq_id
            df.loc[mask, 'ml_quality_score'] = quality_score
            df.loc[mask, 'quality_tier'] = tier
            df.loc[mask, 'sequence_completeness'] = quality_score

        return df

    def _calculate_sequence_quality(self, sequence: pd.DataFrame) -> float:
        """Calculate quality score for a sequence."""
        if len(sequence) < 2:
            return 0.0

        score = 0.0
        phases = sequence['touchPhase'].tolist()

        # Check for proper start
        if phases[0] == 'Began':
            score += 0.4

        # Check for proper end
        if phases[-1] == 'Ended':
            score += 0.4
        elif phases[-1] == 'Canceled':
            score += 0.2

        # Check sequence length (reasonable interaction)
        if 2 <= len(phases) <= 20:
            score += 0.2

        return min(1.0, score)

    def _classify_behavior(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify behavioral patterns."""

        # Group by sequence identifier
        if 'Touchdata_id' in df.columns:
            groupby_col = 'Touchdata_id'
        else:
            groupby_col = 'fingerId'

        # Initialize behavioral columns
        df['behavioral_pattern'] = 'unknown'
        df['interaction_style'] = 'unknown'
        df['user_intent_confidence'] = 0.0

        for seq_id, group in df.groupby(groupby_col):
            phases = group['touchPhase'].tolist()

            # Classify pattern
            if len(phases) == 2 and phases == ['Began', 'Ended']:
                pattern = 'tap'
                style = 'deliberate'
                confidence = 0.9
            elif 'Moved' in phases and len(phases) > 3:
                pattern = 'drag'
                style = 'deliberate'
                confidence = 0.8
            elif 'Canceled' in phases:
                pattern = 'interrupted'
                style = 'interrupted'
                confidence = 0.6
            else:
                pattern = 'complex'
                style = 'unknown'
                confidence = 0.3

            # Update DataFrame
            mask = df[groupby_col] == seq_id
            df.loc[mask, 'behavioral_pattern'] = pattern
            df.loc[mask, 'interaction_style'] = style
            df.loc[mask, 'user_intent_confidence'] = confidence

        return df

    def _detect_simple_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simple anomaly detection based on statistical outliers."""

        # Initialize anomaly columns
        df['anomaly_score'] = 0.0
        df['anomaly_type'] = 'normal'

        # Check for spatial outliers
        if 'distance' in df.columns:
            # Calculate z-scores for movement distances
            mean_dist = df['distance'].mean()
            std_dist = df['distance'].std()

            if std_dist > 0:
                z_scores = np.abs((df['distance'] - mean_dist) / std_dist)
                df['anomaly_score'] = z_scores / 3.0  # Normalize to 0-1 range
                df['anomaly_type'] = np.where(z_scores > 3, 'outlier', 'normal')

        return df

    def _generate_recommendations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate usage recommendations based on quality and behavior."""

        def get_recommendations(row):
            recommendations = []

            # Quality-based recommendations
            quality_score = row.get('ml_quality_score', 0)
            if quality_score > 0.8:
                recommendations.extend(['timing_analysis', 'spatial_analysis'])
            elif quality_score > 0.5:
                recommendations.append('general_analysis')
            else:
                recommendations.append('quality_research')

            # Pattern-based recommendations
            pattern = row.get('behavioral_pattern', 'unknown')
            if pattern == 'tap':
                recommendations.append('interaction_timing_analysis')
            elif pattern == 'drag':
                recommendations.append('movement_analysis')
            elif pattern == 'interrupted':
                recommendations.append('error_analysis')

            return recommendations

        df['usage_recommendations'] = df.apply(get_recommendations, axis=1)
        return df

    def _dataframe_to_json(self, df: pd.DataFrame, original_data: Dict) -> Dict:
        """Convert enhanced DataFrame back to JSON format."""
        try:
            # Start with original structure
            enhanced_data = original_data.copy()

            # Group by Touchdata_id to reconstruct touch sequences
            touch_data = {}

            for touchdata_id, group in df.groupby('Touchdata_id'):
                # Sort by event_index
                group = group.sort_values('event_index')

                # Convert to list of dictionaries
                entries = []
                for _, row in group.iterrows():
                    entry = row.to_dict()

                    # Remove NaN values and convert numpy types
                    clean_entry = {}
                    for k, v in entry.items():
                        if pd.notna(v):
                            # Handle different data types
                            if hasattr(v, 'item'):  # numpy scalar
                                clean_entry[k] = v.item()
                            elif isinstance(v, (list, np.ndarray)):  # arrays/lists
                                clean_entry[k] = list(v) if len(v) > 0 else []
                            elif isinstance(v, (np.integer, np.floating)):  # numpy types
                                clean_entry[k] = v.item()
                            else:
                                clean_entry[k] = v
                    entry = clean_entry

                    entries.append(entry)

                touch_data[str(touchdata_id)] = entries

            # Update the enhanced data structure
            enhanced_data['json']['touchData'] = touch_data

            # Add ML metadata summary
            enhanced_data['ml_metadata'] = {
                'processing_timestamp': pd.Timestamp.now().isoformat(),
                'total_sequences': len(df['Touchdata_id'].unique()),
                'total_touch_points': len(df),
                'quality_distribution': df['quality_tier'].value_counts().to_dict(),
                'behavioral_patterns': df['behavioral_pattern'].value_counts().to_dict()
            }

            return enhanced_data

        except Exception as e:
            logger.error(f"Error converting DataFrame to JSON: {e}")
            return original_data

    def _generate_processing_stats(self, original_df: pd.DataFrame, enhanced_df: pd.DataFrame) -> Dict:
        """Generate statistics about the processing results."""

        stats = {
            'original_data_points': len(original_df),
            'enhanced_data_points': len(enhanced_df),
            'sequences_processed': len(enhanced_df['Touchdata_id'].unique()) if 'Touchdata_id' in enhanced_df.columns else 0,
            'features_added': len(enhanced_df.columns) - len(original_df.columns),
            'quality_distribution': enhanced_df['quality_tier'].value_counts().to_dict() if 'quality_tier' in enhanced_df.columns else {},
            'behavioral_patterns': enhanced_df['behavioral_pattern'].value_counts().to_dict() if 'behavioral_pattern' in enhanced_df.columns else {}
        }

        return stats
