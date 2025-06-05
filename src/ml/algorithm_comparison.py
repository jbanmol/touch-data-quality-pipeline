#!/usr/bin/env python3
"""
ML Algorithm Comparison and Benchmarking for Touch Data

This module provides comprehensive comparison of different machine learning
algorithms for touch data analysis, helping to select the best performing
methods for anomaly detection, clustering, and behavioral classification.
"""

import pandas as pd
import numpy as np
import logging
import time
import warnings
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.model_selection import cross_val_score
from scipy import stats

# Optional visualization imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class MLAlgorithmComparator:
    """
    Comprehensive comparison of ML algorithms for touch data analysis.
    Tests multiple algorithms and provides detailed performance metrics.
    """
    
    def __init__(self):
        self.anomaly_algorithms = {
            'isolation_forest': {
                'algorithm': IsolationForest,
                'params': {'contamination': 0.1, 'random_state': 42, 'n_estimators': 100}
            },
            'lof': {
                'algorithm': LocalOutlierFactor,
                'params': {'n_neighbors': 20, 'contamination': 0.1}
            },
            'one_class_svm': {
                'algorithm': OneClassSVM,
                'params': {'gamma': 'scale', 'nu': 0.1}
            },
            'elliptic_envelope': {
                'algorithm': None,  # Will import if available
                'params': {'contamination': 0.1, 'random_state': 42}
            }
        }
        
        self.clustering_algorithms = {
            'kmeans': {
                'algorithm': KMeans,
                'params': {'n_clusters': 4, 'random_state': 42, 'n_init': 10}
            },
            'dbscan': {
                'algorithm': DBSCAN,
                'params': {'eps': 0.5, 'min_samples': 5}
            },
            'gaussian_mixture': {
                'algorithm': GaussianMixture,
                'params': {'n_components': 4, 'random_state': 42}
            },
            'agglomerative': {
                'algorithm': AgglomerativeClustering,
                'params': {'n_clusters': 4, 'linkage': 'ward'}
            }
        }
        
        self.scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler()
        }
        
        self.results = {}
        
        # Try to import additional algorithms
        try:
            from sklearn.covariance import EllipticEnvelope
            self.anomaly_algorithms['elliptic_envelope']['algorithm'] = EllipticEnvelope
        except ImportError:
            logger.warning("EllipticEnvelope not available")
            del self.anomaly_algorithms['elliptic_envelope']
        
        logger.info("ML Algorithm Comparator initialized")
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare feature matrix for algorithm comparison.
        
        Args:
            df: DataFrame with touch data
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        # Select relevant features for ML analysis
        feature_cols = []
        
        # Basic features
        basic_features = ['x', 'y', 'time', 'completionPerc']
        for col in basic_features:
            if col in df.columns:
                feature_cols.append(col)
        
        # Advanced features (if available)
        advanced_features = [
            'sequence_length', 'sequence_duration', 'mean_time_interval',
            'std_time_interval', 'total_distance', 'mean_distance',
            'spatial_spread', 'trajectory_smoothness', 'direction_consistency'
        ]
        for col in advanced_features:
            if col in df.columns:
                feature_cols.append(col)
        
        if len(feature_cols) < 2:
            logger.warning("Insufficient features for algorithm comparison")
            return np.array([]), []
        
        # Create feature matrix
        X = df[feature_cols].fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        logger.info(f"Prepared feature matrix with {X.shape[1]} features and {X.shape[0]} samples")
        return X.values, feature_cols
    
    def compare_scalers(self, X: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Compare different scaling methods.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary with scaler performance metrics
        """
        logger.info("Comparing scaling methods...")
        
        scaler_results = {}
        
        for scaler_name, scaler in self.scalers.items():
            try:
                start_time = time.time()
                X_scaled = scaler.fit_transform(X)
                scaling_time = time.time() - start_time
                
                # Calculate metrics
                metrics = {
                    'mean_variance': np.mean(np.var(X_scaled, axis=0)),
                    'max_variance': np.max(np.var(X_scaled, axis=0)),
                    'mean_absolute_mean': np.mean(np.abs(np.mean(X_scaled, axis=0))),
                    'scaling_time': scaling_time,
                    'condition_number': np.linalg.cond(np.cov(X_scaled.T))
                }
                
                # Overall score (lower is better for variance and mean, lower condition number is better)
                metrics['overall_score'] = (
                    1.0 / (1.0 + metrics['mean_variance']) +
                    1.0 / (1.0 + metrics['mean_absolute_mean']) +
                    1.0 / (1.0 + np.log(metrics['condition_number']))
                ) / 3.0
                
                scaler_results[scaler_name] = metrics
                
            except Exception as e:
                logger.warning(f"Scaler {scaler_name} failed: {e}")
                scaler_results[scaler_name] = {'overall_score': 0.0, 'error': str(e)}
        
        return scaler_results
    
    def compare_anomaly_detection(self, X: np.ndarray, X_scaled: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Compare anomaly detection algorithms.
        
        Args:
            X: Original feature matrix
            X_scaled: Scaled feature matrix
            
        Returns:
            Dictionary with anomaly detection performance metrics
        """
        logger.info("Comparing anomaly detection algorithms...")
        
        anomaly_results = {}
        
        for algo_name, algo_config in self.anomaly_algorithms.items():
            if algo_config['algorithm'] is None:
                continue
                
            try:
                start_time = time.time()
                
                # Initialize algorithm
                algorithm = algo_config['algorithm'](**algo_config['params'])
                
                # Fit and predict
                if algo_name == 'lof':
                    # LOF doesn't have separate fit/predict
                    outliers = algorithm.fit_predict(X_scaled)
                    scores = algorithm.negative_outlier_factor_
                else:
                    algorithm.fit(X_scaled)
                    outliers = algorithm.predict(X_scaled)
                    if hasattr(algorithm, 'score_samples'):
                        scores = algorithm.score_samples(X_scaled)
                    else:
                        scores = algorithm.decision_function(X_scaled)
                
                training_time = time.time() - start_time
                
                # Calculate metrics
                outlier_ratio = np.sum(outliers == -1) / len(outliers)
                
                metrics = {
                    'outlier_ratio': outlier_ratio,
                    'training_time': training_time,
                    'score_range': np.max(scores) - np.min(scores),
                    'score_std': np.std(scores)
                }
                
                # Target outlier ratio is 10%, so score based on how close we get
                metrics['ratio_score'] = 1.0 - abs(outlier_ratio - 0.1)
                
                # Overall score
                metrics['overall_score'] = (
                    metrics['ratio_score'] * 0.6 +
                    (1.0 / (1.0 + training_time)) * 0.2 +
                    (metrics['score_range'] / (metrics['score_range'] + 1.0)) * 0.2
                )
                
                anomaly_results[algo_name] = metrics
                
            except Exception as e:
                logger.warning(f"Anomaly algorithm {algo_name} failed: {e}")
                anomaly_results[algo_name] = {'overall_score': 0.0, 'error': str(e)}
        
        return anomaly_results
    
    def compare_clustering(self, X: np.ndarray, X_scaled: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Compare clustering algorithms.
        
        Args:
            X: Original feature matrix
            X_scaled: Scaled feature matrix
            
        Returns:
            Dictionary with clustering performance metrics
        """
        logger.info("Comparing clustering algorithms...")
        
        clustering_results = {}
        
        for algo_name, algo_config in self.clustering_algorithms.items():
            try:
                start_time = time.time()
                
                # Initialize algorithm
                algorithm = algo_config['algorithm'](**algo_config['params'])
                
                # Fit and predict
                labels = algorithm.fit_predict(X_scaled)
                training_time = time.time() - start_time
                
                # Calculate metrics
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                noise_ratio = np.sum(labels == -1) / len(labels) if -1 in labels else 0.0
                
                metrics = {
                    'n_clusters': n_clusters,
                    'noise_ratio': noise_ratio,
                    'training_time': training_time
                }
                
                # Calculate silhouette score if we have valid clusters
                if n_clusters > 1 and n_clusters < len(X_scaled) - 1:
                    try:
                        metrics['silhouette_score'] = silhouette_score(X_scaled, labels)
                    except:
                        metrics['silhouette_score'] = 0.0
                else:
                    metrics['silhouette_score'] = 0.0
                
                # Calculate cluster balance (how evenly distributed are the clusters)
                if n_clusters > 1:
                    cluster_sizes = [np.sum(labels == i) for i in range(n_clusters)]
                    cluster_balance = 1.0 - np.std(cluster_sizes) / np.mean(cluster_sizes)
                    metrics['cluster_balance'] = max(0.0, cluster_balance)
                else:
                    metrics['cluster_balance'] = 0.0
                
                # Overall score
                metrics['overall_score'] = (
                    metrics['silhouette_score'] * 0.4 +
                    (1.0 - noise_ratio) * 0.3 +
                    metrics['cluster_balance'] * 0.2 +
                    (1.0 / (1.0 + training_time)) * 0.1
                )
                
                clustering_results[algo_name] = metrics
                
            except Exception as e:
                logger.warning(f"Clustering algorithm {algo_name} failed: {e}")
                clustering_results[algo_name] = {'overall_score': 0.0, 'error': str(e)}
        
        return clustering_results

    def run_comprehensive_comparison(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run comprehensive comparison of all algorithms.

        Args:
            df: DataFrame with touch data

        Returns:
            Dictionary with complete comparison results
        """
        logger.info("Starting comprehensive ML algorithm comparison...")

        # Prepare features
        X, feature_names = self.prepare_features(df)

        if len(X) == 0:
            logger.error("No features available for comparison")
            return {}

        results = {
            'dataset_info': {
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'feature_names': feature_names
            }
        }

        # Compare scalers
        scaler_results = self.compare_scalers(X)
        results['scalers'] = scaler_results

        # Select best scaler
        best_scaler_name = max(scaler_results.keys(),
                              key=lambda k: scaler_results[k].get('overall_score', 0))
        best_scaler = self.scalers[best_scaler_name]
        X_scaled = best_scaler.fit_transform(X)

        results['best_scaler'] = best_scaler_name

        # Compare anomaly detection algorithms
        anomaly_results = self.compare_anomaly_detection(X, X_scaled)
        results['anomaly_detection'] = anomaly_results

        # Select best anomaly detection algorithm
        if anomaly_results:
            best_anomaly_name = max(anomaly_results.keys(),
                                   key=lambda k: anomaly_results[k].get('overall_score', 0))
            results['best_anomaly_algorithm'] = best_anomaly_name

        # Compare clustering algorithms
        clustering_results = self.compare_clustering(X, X_scaled)
        results['clustering'] = clustering_results

        # Select best clustering algorithm
        if clustering_results:
            best_clustering_name = max(clustering_results.keys(),
                                      key=lambda k: clustering_results[k].get('overall_score', 0))
            results['best_clustering_algorithm'] = best_clustering_name

        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(results)

        logger.info("Comprehensive algorithm comparison completed")
        return results

    def _generate_recommendations(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Generate algorithm recommendations based on comparison results."""
        recommendations = {}

        # Scaler recommendation
        scaler_scores = {name: data.get('overall_score', 0)
                        for name, data in results.get('scalers', {}).items()}
        if scaler_scores:
            best_scaler = max(scaler_scores.keys(), key=lambda k: scaler_scores[k])
            recommendations['scaler'] = f"Use {best_scaler} scaling (score: {scaler_scores[best_scaler]:.3f})"

        # Anomaly detection recommendation
        anomaly_scores = {name: data.get('overall_score', 0)
                         for name, data in results.get('anomaly_detection', {}).items()}
        if anomaly_scores:
            best_anomaly = max(anomaly_scores.keys(), key=lambda k: anomaly_scores[k])
            recommendations['anomaly_detection'] = f"Use {best_anomaly} (score: {anomaly_scores[best_anomaly]:.3f})"

        # Clustering recommendation
        clustering_scores = {name: data.get('overall_score', 0)
                           for name, data in results.get('clustering', {}).items()}
        if clustering_scores:
            best_clustering = max(clustering_scores.keys(), key=lambda k: clustering_scores[k])
            recommendations['clustering'] = f"Use {best_clustering} (score: {clustering_scores[best_clustering]:.3f})"

        return recommendations

    def save_results(self, results: Dict[str, Any], output_path: str = "ML/algorithm_comparison_results.json"):
        """Save comparison results to JSON file."""
        import json

        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        results_serializable = convert_numpy(results)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)

        logger.info(f"Algorithm comparison results saved to {output_path}")

    def print_summary(self, results: Dict[str, Any]):
        """Print a summary of the comparison results."""
        print("\n" + "="*60)
        print("ML ALGORITHM COMPARISON SUMMARY")
        print("="*60)

        # Dataset info
        dataset_info = results.get('dataset_info', {})
        print(f"\nDataset: {dataset_info.get('n_samples', 0)} samples, {dataset_info.get('n_features', 0)} features")

        # Best algorithms
        print(f"\nBest Scaler: {results.get('best_scaler', 'Unknown')}")
        print(f"Best Anomaly Detection: {results.get('best_anomaly_algorithm', 'Unknown')}")
        print(f"Best Clustering: {results.get('best_clustering_algorithm', 'Unknown')}")

        # Detailed results
        print("\n" + "-"*40)
        print("SCALER COMPARISON")
        print("-"*40)
        for name, data in results.get('scalers', {}).items():
            score = data.get('overall_score', 0)
            print(f"{name:15}: {score:.3f}")

        print("\n" + "-"*40)
        print("ANOMALY DETECTION COMPARISON")
        print("-"*40)
        for name, data in results.get('anomaly_detection', {}).items():
            score = data.get('overall_score', 0)
            outlier_ratio = data.get('outlier_ratio', 0)
            print(f"{name:20}: {score:.3f} (outliers: {outlier_ratio:.1%})")

        print("\n" + "-"*40)
        print("CLUSTERING COMPARISON")
        print("-"*40)
        for name, data in results.get('clustering', {}).items():
            score = data.get('overall_score', 0)
            n_clusters = data.get('n_clusters', 0)
            silhouette = data.get('silhouette_score', 0)
            print(f"{name:20}: {score:.3f} (clusters: {n_clusters}, silhouette: {silhouette:.3f})")

        # Recommendations
        print("\n" + "-"*40)
        print("RECOMMENDATIONS")
        print("-"*40)
        for category, recommendation in results.get('recommendations', {}).items():
            print(f"{category}: {recommendation}")

        print("\n" + "="*60)
