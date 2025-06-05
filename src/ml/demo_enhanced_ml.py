#!/usr/bin/env python3
"""
Enhanced ML System Demonstration

This script demonstrates the capabilities of the enhanced ML-based data flagging
system, including algorithm comparison, advanced feature engineering, and
comprehensive quality assessment.
"""

import pandas as pd
import numpy as np
import logging
import os
import sys
import json
from typing import Dict, Any

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from ml.enhanced_ml_flagging import EnhancedMLFlaggingSystem
    from ml.algorithm_comparison import MLAlgorithmComparator
    from ml.advanced_feature_engineering import AdvancedTouchFeatureEngineer
    from ml.ml_integration import MLIntegrationManager
    ENHANCED_ML_AVAILABLE = True
except ImportError as e:
    print(f"Enhanced ML components not available: {e}")
    ENHANCED_ML_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_demo_data() -> pd.DataFrame:
    """Create realistic demo touch data for testing."""
    logger.info("Creating demo touch data...")
    
    np.random.seed(42)
    data = []
    
    # Create different types of touch sequences for demonstration
    scenarios = [
        {'name': 'precise_drawing', 'n_sequences': 3, 'quality': 'high'},
        {'name': 'quick_taps', 'n_sequences': 2, 'quality': 'medium'},
        {'name': 'erratic_movement', 'n_sequences': 2, 'quality': 'low'},
        {'name': 'interrupted_sequences', 'n_sequences': 2, 'quality': 'low'}
    ]
    
    touchdata_id = 1
    
    for scenario in scenarios:
        for seq in range(scenario['n_sequences']):
            if scenario['name'] == 'precise_drawing':
                # High-quality precise drawing sequences
                n_points = np.random.randint(15, 25)
                start_x, start_y = 200 + seq * 100, 300
                
                # Create smooth curved path
                x_coords = []
                y_coords = []
                for i in range(n_points):
                    t = i / (n_points - 1)
                    x = start_x + 200 * t + 20 * np.sin(4 * np.pi * t)
                    y = start_y + 100 * np.sin(2 * np.pi * t)
                    x_coords.append(x + np.random.normal(0, 2))  # Small noise
                    y_coords.append(y + np.random.normal(0, 2))
                
                # Regular timing
                times = [touchdata_id * 1000 + i * 0.05 for i in range(n_points)]
                phases = ['Began'] + ['Moved'] * (n_points - 2) + ['Ended']
                
            elif scenario['name'] == 'quick_taps':
                # Quick tap sequences
                n_points = np.random.randint(3, 6)
                x_coords = [400 + seq * 50 + np.random.normal(0, 5) for _ in range(n_points)]
                y_coords = [200 + np.random.normal(0, 5) for _ in range(n_points)]
                
                # Very quick timing
                times = [touchdata_id * 1000 + i * 0.02 for i in range(n_points)]
                phases = ['Began'] + ['Stationary'] * (n_points - 2) + ['Ended']
                
            elif scenario['name'] == 'erratic_movement':
                # Erratic movement with large jumps
                n_points = np.random.randint(8, 15)
                x_coords = [600]
                y_coords = [400]
                
                for i in range(1, n_points):
                    # Large random jumps
                    x_coords.append(x_coords[-1] + np.random.normal(0, 50))
                    y_coords.append(y_coords[-1] + np.random.normal(0, 50))
                
                # Irregular timing
                times = [touchdata_id * 1000]
                for i in range(1, n_points):
                    times.append(times[-1] + np.random.exponential(0.1))
                
                phases = ['Began'] + ['Moved'] * (n_points - 2) + ['Ended']
                
            else:  # interrupted_sequences
                # Sequences that get canceled
                n_points = np.random.randint(5, 10)
                x_coords = [800 + i * 10 for i in range(n_points)]
                y_coords = [300 + i * 5 for i in range(n_points)]
                
                times = [touchdata_id * 1000 + i * 0.08 for i in range(n_points)]
                phases = ['Began'] + ['Moved'] * (n_points - 2) + ['Canceled']
            
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
                    'zone': 'Wall',
                    'scenario': scenario['name']
                })
            
            touchdata_id += 1
    
    df = pd.DataFrame(data)
    logger.info(f"Created demo data with {len(df)} rows and {touchdata_id-1} sequences")
    return df

def demonstrate_algorithm_comparison(df: pd.DataFrame) -> Dict[str, Any]:
    """Demonstrate algorithm comparison functionality."""
    logger.info("=== ALGORITHM COMPARISON DEMONSTRATION ===")
    
    comparator = MLAlgorithmComparator()
    results = comparator.run_comprehensive_comparison(df)
    
    # Print detailed results
    comparator.print_summary(results)
    
    return results

def demonstrate_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Demonstrate advanced feature engineering."""
    logger.info("=== FEATURE ENGINEERING DEMONSTRATION ===")
    
    engineer = AdvancedTouchFeatureEngineer()
    df_features = engineer.extract_all_features(df)
    
    # Show feature extraction results
    original_cols = len(df.columns)
    new_cols = len(df_features.columns)
    logger.info(f"Original columns: {original_cols}")
    logger.info(f"Enhanced columns: {new_cols}")
    logger.info(f"New features added: {new_cols - original_cols}")
    
    # Show some example features
    feature_names = engineer.feature_names[:10]  # Show first 10 features
    logger.info(f"Example features: {feature_names}")
    
    # Calculate feature importance
    importance = engineer.get_feature_importance(df_features)
    top_features = engineer.get_top_features(10)
    
    logger.info("Top 10 most important features:")
    for i, feature in enumerate(top_features, 1):
        score = importance.get(feature, 0)
        logger.info(f"  {i:2d}. {feature}: {score:.3f}")
    
    return df_features

def demonstrate_ml_flagging(df: pd.DataFrame) -> pd.DataFrame:
    """Demonstrate the complete ML flagging system."""
    logger.info("=== ML FLAGGING SYSTEM DEMONSTRATION ===")
    
    # Create ML system
    ml_system = EnhancedMLFlaggingSystem()
    
    # Run complete enhancement
    df_enhanced = ml_system.enhance_dataframe(df)
    
    # Show results by sequence
    logger.info("ML Flagging Results by Sequence:")
    logger.info("-" * 80)
    
    for touchdata_id, group in df_enhanced.groupby('Touchdata_id'):
        scenario = group['scenario'].iloc[0] if 'scenario' in group.columns else 'unknown'
        quality_score = group['quality_score'].iloc[0]
        interaction_type = group['interaction_type'].iloc[0]
        anomaly_flag = group['anomaly_flag'].iloc[0]
        research_suitability = group['research_suitability'].iloc[0]
        
        logger.info(f"Sequence {touchdata_id} ({scenario}):")
        logger.info(f"  Quality Score: {quality_score}")
        logger.info(f"  Interaction Type: {interaction_type}")
        logger.info(f"  Anomaly Flag: {anomaly_flag}")
        logger.info(f"  Research Suitability: {research_suitability}")
        logger.info("")
    
    return df_enhanced

def demonstrate_integration(df: pd.DataFrame) -> pd.DataFrame:
    """Demonstrate ML integration with existing pipeline."""
    logger.info("=== INTEGRATION DEMONSTRATION ===")
    
    # Use integration manager
    manager = MLIntegrationManager(enable_algorithm_comparison=True)
    df_integrated = manager.enhance_dataframe_with_ml(df, run_algorithm_comparison=True)
    
    # Show integration results
    logger.info("Integration completed successfully!")
    logger.info(f"Final DataFrame shape: {df_integrated.shape}")
    
    # Check data integrity
    original_cols = ['Touchdata_id', 'event_index', 'x', 'y', 'time', 'touchPhase']
    for col in original_cols:
        if col in df.columns and col in df_integrated.columns:
            if df[col].equals(df_integrated[col]):
                logger.info(f"✓ {col}: Data preserved")
            else:
                logger.warning(f"✗ {col}: Data modified!")
    
    return df_integrated

def save_demo_results(df_enhanced: pd.DataFrame, algorithm_results: Dict[str, Any]):
    """Save demonstration results for review."""
    logger.info("=== SAVING DEMO RESULTS ===")
    
    # Create output directory
    output_dir = "ML/demo_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save enhanced data
    output_file = os.path.join(output_dir, "demo_enhanced_data.csv")
    df_enhanced.to_csv(output_file, index=False)
    logger.info(f"Enhanced data saved to: {output_file}")
    
    # Save algorithm comparison results
    results_file = os.path.join(output_dir, "demo_algorithm_results.json")
    with open(results_file, 'w') as f:
        json.dump(algorithm_results, f, indent=2, default=str)
    logger.info(f"Algorithm results saved to: {results_file}")
    
    # Create summary report
    summary_file = os.path.join(output_dir, "demo_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("Enhanced ML System Demonstration Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total sequences processed: {df_enhanced['Touchdata_id'].nunique()}\n")
        f.write(f"Total data points: {len(df_enhanced)}\n\n")
        
        f.write("Quality Score Distribution:\n")
        quality_dist = df_enhanced.groupby('Touchdata_id')['quality_score'].first().value_counts().sort_index()
        for score, count in quality_dist.items():
            f.write(f"  Score {score}: {count} sequences\n")
        
        f.write("\nInteraction Type Distribution:\n")
        interaction_dist = df_enhanced.groupby('Touchdata_id')['interaction_type'].first().value_counts()
        for itype, count in interaction_dist.items():
            f.write(f"  {itype}: {count} sequences\n")
        
        f.write("\nAnomaly Flag Distribution:\n")
        anomaly_dist = df_enhanced.groupby('Touchdata_id')['anomaly_flag'].first().value_counts()
        for flag, count in anomaly_dist.items():
            f.write(f"  {flag}: {count} sequences\n")
    
    logger.info(f"Summary report saved to: {summary_file}")

def main():
    """Main demonstration function."""
    if not ENHANCED_ML_AVAILABLE:
        print("Enhanced ML components not available. Please install required dependencies.")
        return
    
    logger.info("Starting Enhanced ML System Demonstration")
    logger.info("=" * 60)
    
    try:
        # Step 1: Create demo data
        df = create_demo_data()
        
        # Step 2: Demonstrate algorithm comparison
        algorithm_results = demonstrate_algorithm_comparison(df)
        
        # Step 3: Demonstrate feature engineering
        df_features = demonstrate_feature_engineering(df)
        
        # Step 4: Demonstrate ML flagging
        df_flagged = demonstrate_ml_flagging(df)
        
        # Step 5: Demonstrate integration
        df_final = demonstrate_integration(df)
        
        # Step 6: Save results
        save_demo_results(df_final, algorithm_results)
        
        logger.info("=" * 60)
        logger.info("Enhanced ML System Demonstration completed successfully!")
        logger.info("Check the ML/demo_results/ directory for detailed outputs.")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise

if __name__ == "__main__":
    main()
