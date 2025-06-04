#!/usr/bin/env python3
"""
ML-Based Cleaning Pipeline for Coloring Touch Data

This module provides the main pipeline for ML-based data cleaning that:
1. Preserves all original data
2. Adds rich metadata and quality assessments
3. Uses transfer learning for behavioral analysis
4. Integrates with existing data processing pipeline
"""

import pandas as pd
import numpy as np
import json
import os
import logging
from typing import Dict, List, Tuple, Optional, Any
import warnings
from pathlib import Path

# Import our ML components
from .metadata_enhancer import TouchDataMetadataEnhancer
from .transfer_learning_model import TouchDataTransferLearner
from .feature_engineering import TouchFeatureEngineer

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MLTouchDataCleaner:
    """
    Main ML-based cleaning pipeline for touch data.
    Focuses on metadata enhancement and quality assessment without modifying original data.
    """

    def __init__(self, model_dir: str = "ML/models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)

        # Initialize components
        self.metadata_enhancer = TouchDataMetadataEnhancer()
        self.feature_engineer = TouchFeatureEngineer()
        self.transfer_learner = None

        # Model paths
        self.pretrained_model_path = self.model_dir / "pretrained_touch_model.pth"
        self.finetuned_model_path = self.model_dir / "finetuned_coloring_model.pth"

        logger.info("Initialized ML Touch Data Cleaner")

    def setup_transfer_learning(self, force_retrain: bool = False):
        """
        Setup and train the transfer learning model if needed.

        Args:
            force_retrain: If True, retrain even if model exists
        """
        logger.info("Setting up transfer learning model...")

        try:
            # Initialize transfer learner
            self.transfer_learner = TouchDataTransferLearner()

            # Check if pre-trained model exists
            if self.pretrained_model_path.exists() and not force_retrain:
                logger.info("Loading existing pre-trained model...")
                self.transfer_learner.load_model(str(self.pretrained_model_path))
            else:
                logger.info("Pre-training new model on synthetic data...")
                self.transfer_learner.pretrain(num_epochs=10, batch_size=8)  # Reduced for faster setup
                self.transfer_learner.save_model(str(self.pretrained_model_path))
                logger.info("Pre-training completed and model saved")

        except Exception as e:
            logger.warning(f"Transfer learning setup failed: {e}")
            logger.warning("Continuing without transfer learning (basic ML features still available)")
            self.transfer_learner = None

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

        # Step 1: Feature Engineering
        logger.info("Step 1: Feature engineering...")
        enhanced_df = self.feature_engineer.extract_all_features(enhanced_df)

        # Step 2: Quality Assessment
        logger.info("Step 2: Quality assessment...")
        enhanced_df = self.metadata_enhancer.analyze_sequence_quality(enhanced_df)

        # Step 3: Behavioral Classification
        logger.info("Step 3: Behavioral classification...")
        enhanced_df = self.metadata_enhancer.classify_behavioral_patterns(enhanced_df)

        # Step 4: Anomaly Detection
        logger.info("Step 4: Anomaly detection...")
        enhanced_df = self.metadata_enhancer.detect_anomalies(enhanced_df)

        # Step 5: Transfer Learning Analysis (if model is available)
        if self.transfer_learner and self.transfer_learner.is_pretrained:
            logger.info("Step 5: Transfer learning analysis...")
            enhanced_df = self._apply_transfer_learning_analysis(enhanced_df)

        # Step 6: Usage Recommendations
        logger.info("Step 6: Generating usage recommendations...")
        enhanced_df = self._generate_usage_recommendations(enhanced_df)

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

                    # Remove NaN values and convert numpy types - fix the array comparison issue
                    cleaned_entry = {}
                    for k, v in entry.items():
                        # Handle different types of values
                        if isinstance(v, (list, np.ndarray)):
                            # For arrays/lists, keep them as is
                            cleaned_entry[k] = v.tolist() if hasattr(v, 'tolist') else list(v)
                        elif pd.isna(v):
                            # Skip NaN values
                            continue
                        elif hasattr(v, 'item'):
                            # Convert numpy scalars
                            cleaned_entry[k] = v.item()
                        else:
                            # Keep regular values
                            cleaned_entry[k] = v

                    entries.append(cleaned_entry)

                touch_data[str(touchdata_id)] = entries

            # Update the enhanced data structure
            enhanced_data['json']['touchData'] = touch_data

            # Add ML metadata summary
            enhanced_data['ml_metadata'] = {
                'processing_timestamp': pd.Timestamp.now().isoformat(),
                'total_sequences': len(df['Touchdata_id'].unique()),
                'total_touch_points': len(df),
                'quality_distribution': {
                    'high': int((df['quality_tier'] == 'high').sum()),
                    'medium': int((df['quality_tier'] == 'medium').sum()),
                    'low': int((df['quality_tier'] == 'low').sum())
                },
                'anomaly_count': int((df['anomaly_type'] == 'outlier').sum()) if 'anomaly_type' in df.columns else 0,
                'behavioral_patterns': df['behavioral_pattern'].value_counts().to_dict() if 'behavioral_pattern' in df.columns else {}
            }

            return enhanced_data

        except Exception as e:
            logger.error(f"Error converting DataFrame to JSON: {e}")
            return original_data

    def _apply_transfer_learning_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply transfer learning model analysis to the data."""
        # This would use the trained model to provide additional insights
        # For now, we'll add placeholder columns that would be filled by the model

        df['ml_quality_prediction'] = 0.0
        df['ml_pattern_prediction'] = 'unknown'
        df['ml_confidence'] = 0.0

        # In a full implementation, this would:
        # 1. Prepare sequences for the model
        # 2. Run inference
        # 3. Add predictions to the DataFrame

        return df

    def _generate_usage_recommendations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate recommendations for data usage based on quality metrics."""

        def get_recommendations(row):
            recommendations = []

            # Quality-based recommendations
            if row.get('ml_quality_score', 0) > 0.8:
                recommendations.extend(['timing_analysis', 'spatial_analysis', 'completion_tracking'])
            elif row.get('ml_quality_score', 0) > 0.5:
                recommendations.extend(['general_analysis', 'pattern_recognition'])
            else:
                recommendations.append('quality_research')

            # Behavioral pattern recommendations
            if row.get('behavioral_pattern') == 'deliberate':
                recommendations.append('user_intent_analysis')
            elif row.get('behavioral_pattern') == 'erratic':
                recommendations.append('error_analysis')

            # Anomaly-based recommendations
            if row.get('anomaly_type') == 'outlier':
                recommendations.append('outlier_investigation')

            return recommendations

        df['usage_recommendations'] = df.apply(get_recommendations, axis=1)

        return df

    def _generate_processing_stats(self, original_df: pd.DataFrame, enhanced_df: pd.DataFrame) -> Dict:
        """Generate statistics about the processing results."""

        stats = {
            'original_data_points': len(original_df),
            'enhanced_data_points': len(enhanced_df),
            'sequences_processed': len(enhanced_df['Touchdata_id'].unique()) if 'Touchdata_id' in enhanced_df.columns else 0,
            'features_added': len(enhanced_df.columns) - len(original_df.columns),
            'quality_distribution': {},
            'behavioral_patterns': {},
            'anomaly_summary': {}
        }

        # Quality distribution
        if 'quality_tier' in enhanced_df.columns:
            stats['quality_distribution'] = enhanced_df['quality_tier'].value_counts().to_dict()

        # Behavioral patterns
        if 'behavioral_pattern' in enhanced_df.columns:
            stats['behavioral_patterns'] = enhanced_df['behavioral_pattern'].value_counts().to_dict()

        # Anomaly summary
        if 'anomaly_type' in enhanced_df.columns:
            stats['anomaly_summary'] = enhanced_df['anomaly_type'].value_counts().to_dict()

        return stats

    def process_directory(self, input_dir: str, output_dir: str, file_pattern: str = "Coloring_*.json") -> Dict:
        """
        Process all JSON files in a directory.

        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            file_pattern: File pattern to match

        Returns:
            Summary of processing results
        """
        logger.info(f"Processing directory: {input_dir}")

        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Find matching files
        json_files = list(input_path.glob(file_pattern))

        if not json_files:
            logger.warning(f"No files matching pattern '{file_pattern}' found in {input_dir}")
            return {'status': 'error', 'message': 'No matching files found'}

        # Setup transfer learning if not already done
        if self.transfer_learner is None:
            self.setup_transfer_learning()

        # Process each file
        results = []
        successful = 0
        failed = 0

        for json_file in json_files:
            output_file = output_path / f"enhanced_{json_file.name}"

            result = self.process_json_file(str(json_file), str(output_file))
            results.append(result)

            if result['status'] == 'success':
                successful += 1
            else:
                failed += 1

        # Generate summary
        summary = {
            'status': 'completed',
            'total_files': len(json_files),
            'successful': successful,
            'failed': failed,
            'results': results
        }

        # Save summary
        summary_file = output_path / "processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Directory processing completed. {successful}/{len(json_files)} files processed successfully")

        return summary
