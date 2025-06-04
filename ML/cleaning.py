import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
import logging
import os
import sys

# Add current directory to path for ML imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Configure logging to match your pipeline
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Import our enhanced ML components
try:
    from ml_cleaning_pipeline import MLTouchDataCleaner
    from metadata_enhancer import TouchDataMetadataEnhancer
    from feature_engineering import TouchFeatureEngineer
    ML_COMPONENTS_AVAILABLE = True
    logger.info("Enhanced ML components loaded successfully")
except ImportError as e:
    logger.warning(f"Advanced ML components not available: {e}")
    logger.warning("Falling back to basic ML cleaning")
    ML_COMPONENTS_AVAILABLE = False

def clean_data_with_ml(df, data_type='Coloring'):
    """
    Clean touch interaction data using ML and rule-based methods.
    Adds flags for outliers, imputed values, duplicates, and invalid touchPhase.

    Args:
        df (DataFrame): Input DataFrame with touch data
        data_type (str): 'Coloring' or 'Tracing' to set valid touchPhase values

    Returns:
        DataFrame: Cleaned DataFrame with new flag columns
    """
    try:
        # Initialize flag columns
        df['outlier'] = False
        df['imputed'] = False
        df['duplicate'] = False
        df['invalid_phase'] = False

        # Step 1: Validate touchPhase values
        valid_phases = (
            ['Began', 'Moved', 'Stationary', 'Ended', 'Canceled']
            if data_type == 'Coloring'
            else ['B', 'M', 'S', 'E']
        )
        invalid_phase_mask = ~df['touchPhase'].isin(valid_phases)
        if invalid_phase_mask.any():
            df.loc[invalid_phase_mask, 'invalid_phase'] = True
            logger.info(f"Flagged {invalid_phase_mask.sum()} rows with invalid touchPhase")

        # Step 2: Detect duplicates (exact or near-identical rows)
        # Consider rows with same x, y, time, fingerId as duplicates
        df['temp_key'] = df[['x', 'y', 'time', 'fingerId']].astype(str).agg('_'.join, axis=1)
        duplicate_mask = df.duplicated(subset=['temp_key'], keep='first')
        if duplicate_mask.any():
            df.loc[duplicate_mask, 'duplicate'] = True
            logger.info(f"Flagged {duplicate_mask.sum()} duplicate rows")
        df = df.drop(columns=['temp_key'])

        # Step 3: Prepare features for ML
        # Numerical features
        num_features = ['x', 'y', 'time', 'completionPerc', 'accx', 'accy', 'accz']
        num_features = [f for f in num_features if f in df.columns]

        # Categorical features (encode for ML)
        cat_features = ['touchPhase', 'color', 'zone']
        cat_features = [f for f in cat_features if f in df.columns]
        encoders = {}
        for col in cat_features:
            encoders[col] = LabelEncoder()
            df[f'{col}_encoded'] = encoders[col].fit_transform(df[col].astype(str))

        # Derived features: time differences and spatial distances
        df = df.sort_values(['fingerId', 'time'])
        df['time_diff'] = df.groupby('fingerId')['time'].diff().fillna(0)
        df['distance'] = np.sqrt(
            df.groupby('fingerId')['x'].diff().fillna(0)**2 +
            df.groupby('fingerId')['y'].diff().fillna(0)**2
        )

        # Features for ML
        ml_features = num_features + [f'{c}_encoded' for c in cat_features] + ['time_diff', 'distance']
        X = df[ml_features].copy()

        # Step 4: Impute missing values using KNN
        imputer = KNNImputer(n_neighbors=5, weights='distance')
        X_imputed = imputer.fit_transform(X)
        X_imputed = pd.DataFrame(X_imputed, columns=ml_features, index=X.index)

        # Check for imputed values
        for col in num_features:
            if col in X:
                imputed_mask = X[col].isna() & ~X_imputed[col].isna()
                if imputed_mask.any():
                    df.loc[imputed_mask, 'imputed'] = True
                    df.loc[imputed_mask, col] = X_imputed.loc[imputed_mask, col]
                    logger.info(f"Imputed {imputed_mask.sum()} missing values in {col}")

        # Step 5: Detect outliers using Isolation Forest
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        outlier_labels = iso_forest.fit_predict(X_imputed)
        outlier_mask = outlier_labels == -1
        if outlier_mask.any():
            df.loc[outlier_mask, 'outlier'] = True
            logger.info(f"Flagged {outlier_mask.sum()} rows as outliers")

        # Step 6: Clean up temporary columns
        df = df.drop(columns=[f'{c}_encoded' for c in cat_features] + ['time_diff', 'distance'], errors='ignore')

        logger.info("Completed ML-based data cleaning")
        return df

    except Exception as e:
        logger.error(f"Error in ML data cleaning: {e}")
        return df

def clean_data_with_enhanced_ml(df, data_type='Coloring'):
    """
    Enhanced ML-based data cleaning that preserves all data while adding rich metadata.
    Uses the new ML pipeline for comprehensive quality assessment and behavioral analysis.

    Args:
        df (DataFrame): Input DataFrame with touch data
        data_type (str): 'Coloring' or 'Tracing' to set processing mode

    Returns:
        DataFrame: Enhanced DataFrame with ML metadata (all original data preserved)
    """
    if not ML_COMPONENTS_AVAILABLE:
        logger.warning("Enhanced ML components not available, using basic cleaning")
        return clean_data_with_ml(df, data_type)

    try:
        logger.info("Starting enhanced ML-based data cleaning...")

        # Initialize the enhanced ML cleaner
        metadata_enhancer = TouchDataMetadataEnhancer()
        feature_engineer = TouchFeatureEngineer()

        # Make a copy to preserve original data
        enhanced_df = df.copy()

        # Step 1: Feature Engineering
        logger.info("Extracting comprehensive features...")
        enhanced_df = feature_engineer.extract_all_features(enhanced_df)

        # Step 2: Quality Assessment
        logger.info("Analyzing sequence quality...")
        enhanced_df = metadata_enhancer.analyze_sequence_quality(enhanced_df)

        # Step 3: Behavioral Classification
        logger.info("Classifying behavioral patterns...")
        enhanced_df = metadata_enhancer.classify_behavioral_patterns(enhanced_df)

        # Step 4: Anomaly Detection
        logger.info("Detecting anomalies...")
        enhanced_df = metadata_enhancer.detect_anomalies(enhanced_df)

        # Step 5: Generate Usage Recommendations
        logger.info("Generating usage recommendations...")
        enhanced_df = _generate_usage_recommendations(enhanced_df)

        # Step 6: Add processing metadata
        enhanced_df['ml_processing_timestamp'] = pd.Timestamp.now()
        enhanced_df['ml_processing_version'] = 'enhanced_v1.0'

        logger.info("Enhanced ML-based data cleaning completed successfully")
        return enhanced_df

    except Exception as e:
        logger.error(f"Error in enhanced ML data cleaning: {e}")
        logger.warning("Falling back to basic ML cleaning")
        return clean_data_with_ml(df, data_type)

def _generate_usage_recommendations(df):
    """Generate usage recommendations based on ML analysis."""
    def get_recommendations(row):
        recommendations = []

        # Quality-based recommendations
        quality_score = row.get('ml_quality_score', 0)
        if quality_score > 0.8:
            recommendations.extend(['timing_analysis', 'spatial_analysis', 'completion_tracking'])
        elif quality_score > 0.5:
            recommendations.extend(['general_analysis', 'pattern_recognition'])
        else:
            recommendations.append('quality_research')

        # Behavioral pattern recommendations
        pattern = row.get('behavioral_pattern', 'unknown')
        if pattern == 'deliberate':
            recommendations.append('user_intent_analysis')
        elif pattern == 'erratic':
            recommendations.append('error_analysis')
        elif pattern == 'tap':
            recommendations.append('interaction_timing_analysis')
        elif pattern == 'drag':
            recommendations.append('movement_analysis')

        # Anomaly-based recommendations
        if row.get('anomaly_type') == 'outlier':
            recommendations.append('outlier_investigation')

        # Quality tier recommendations
        tier = row.get('quality_tier', 'unknown')
        if tier == 'high':
            recommendations.append('primary_analysis')
        elif tier == 'medium':
            recommendations.append('secondary_analysis')
        elif tier == 'low':
            recommendations.append('exploratory_analysis')

        return recommendations

    df['usage_recommendations'] = df.apply(get_recommendations, axis=1)
    return df

def integrate_ml_cleaning(input_csv, output_csv):
    """
    Example function to integrate ML cleaning with your pipeline.

    Args:
        input_csv (str): Path to input CSV
        output_csv (str): Path to output CSV
    """
    try:
        # Load data (similar to your load_and_sort_data)
        df = pd.read_csv(input_csv)
        df = df.sort_values('time')
        logger.info(f"Loaded {len(df)} rows from {input_csv}")

        # Detect data type (similar to your detect_data_type)
        data_type = 'Coloring' if 'color' in df.columns and 'completionPerc' in df.columns else 'Tracing'
        logger.info(f"Detected data type: {data_type}")

        # Apply ML cleaning
        df = clean_data_with_ml(df, data_type)

        # Save cleaned data
        df.to_csv(output_csv, index=False)
        logger.info(f"Saved cleaned data to {output_csv}")

        # Return summary for integration with your summary.csv
        flagged_rows = df[['outlier', 'imputed', 'duplicate', 'invalid_phase']].any(axis=1).sum()
        total_rows = len(df)
        return {
            'filename': output_csv.split('/')[-1].replace('.csv', ''),
            'data_type': data_type,
            'flagged_rows': flagged_rows,
            'total_rows': total_rows,
            'flagged_percentage': round((flagged_rows / total_rows) * 100, 2) if total_rows > 0 else 0.0
        }

    except Exception as e:
        logger.error(f"Error processing {input_csv}: {e}")
        return None