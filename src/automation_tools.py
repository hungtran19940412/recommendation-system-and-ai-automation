import logging
from typing import Dict, Optional, List, Union
import numpy as np
from datetime import datetime, timedelta
import threading
import time
import os
import json
from collections import defaultdict
import pandas as pd
from sklearn.metrics import precision_score, recall_score, ndcg_score, average_precision_score
import mlflow
from scipy import stats

logger = logging.getLogger(__name__)

class AutomationTools:
    def __init__(self, metrics_dir: str = "models/metrics", 
                 feedback_dir: str = "data/feedback",
                 mlflow_uri: Optional[str] = None):
        """Initialize automation tools for model management and optimization."""
        self.metrics_dir = metrics_dir
        self.feedback_dir = feedback_dir
        self.model_status = {
            "last_training": None,
            "last_evaluation": None,
            "performance_metrics": {},
            "is_training": False,
            "needs_retraining": False,
            "model_version": "1.0.0",
            "health_status": "healthy"
        }
        self.performance_threshold = 0.7
        self.evaluation_interval = timedelta(hours=24)
        self.monitoring_thread = None
        self.feedback_buffer = defaultdict(list)
        self.feedback_flush_threshold = 100
        
        # Initialize MLflow
        if mlflow_uri:
            mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("recommendation_system")
        
        # Create necessary directories
        os.makedirs(metrics_dir, exist_ok=True)
        os.makedirs(feedback_dir, exist_ok=True)
        
        self.start_monitoring()

    def start_monitoring(self):
        """Start background monitoring of model performance."""
        if not self.monitoring_thread:
            self.monitoring_thread = threading.Thread(
                target=self._monitor_performance,
                daemon=True
            )
            self.monitoring_thread.start()
            logger.info("Started model performance monitoring")

    def _monitor_performance(self):
        """Continuously monitor model performance."""
        while True:
            try:
                self._evaluate_model_performance()
                time.sleep(3600)  # Check every hour
            except Exception as e:
                logger.error(f"Error in performance monitoring: {str(e)}")
                time.sleep(3600)  # Wait before retrying

    def _evaluate_model_performance(self):
        """Evaluate model performance and determine if retraining is needed."""
        if (not self.model_status["last_evaluation"] or 
            datetime.now() - self.model_status["last_evaluation"] > self.evaluation_interval):
            
            try:
                # Calculate performance metrics
                metrics = self._calculate_performance_metrics()
                
                # Update status
                self.model_status["last_evaluation"] = datetime.now()
                self.model_status["performance_metrics"] = metrics
                
                # Check if retraining is needed
                if metrics["average_precision"] < self.performance_threshold:
                    self.model_status["needs_retraining"] = True
                    logger.warning("Model performance below threshold, marking for retraining")
                
                self._save_metrics(metrics)
                logger.info("Model evaluation completed successfully")
            except Exception as e:
                logger.error(f"Error evaluating model performance: {str(e)}")

    def _calculate_performance_metrics(self) -> Dict:
        """Calculate various performance metrics for the model."""
        # Implement actual metric calculation based on your evaluation data
        # This is a placeholder implementation
        return {
            "average_precision": np.random.uniform(0.6, 0.9),
            "recall": np.random.uniform(0.6, 0.9),
            "f1_score": np.random.uniform(0.6, 0.9),
            "timestamp": datetime.now().isoformat()
        }

    def _save_metrics(self, metrics: Dict):
        """Save performance metrics to disk."""
        try:
            metrics_file = os.path.join(self.metrics_dir, 
                                      f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"Saved metrics to {metrics_file}")
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")

    def get_model_status(self) -> Dict:
        """Get current status of the model."""
        return {
            "last_training": self.model_status["last_training"].isoformat() 
                if self.model_status["last_training"] else None,
            "last_evaluation": self.model_status["last_evaluation"].isoformat() 
                if self.model_status["last_evaluation"] else None,
            "performance_metrics": self.model_status["performance_metrics"],
            "is_training": self.model_status["is_training"],
            "needs_retraining": self.model_status["needs_retraining"]
        }

    def trigger_retraining(self, force: bool = False):
        """Trigger model retraining process."""
        if self.model_status["is_training"]:
            raise RuntimeError("Model is already being trained")

        if not force and not self.model_status["needs_retraining"]:
            logger.info("Model retraining not needed")
            return

        try:
            self.model_status["is_training"] = True
            
            # Start retraining in a separate thread
            training_thread = threading.Thread(
                target=self._retrain_model,
                daemon=True
            )
            training_thread.start()
            
            logger.info("Model retraining triggered successfully")
        except Exception as e:
            self.model_status["is_training"] = False
            logger.error(f"Error triggering model retraining: {str(e)}")
            raise

    def _retrain_model(self):
        """Perform model retraining."""
        try:
            # Implement actual model retraining logic here
            logger.info("Starting model retraining process")
            
            # Simulate training time
            time.sleep(300)  # 5 minutes
            
            # Update status
            self.model_status["last_training"] = datetime.now()
            self.model_status["needs_retraining"] = False
            self.model_status["is_training"] = False
            
            logger.info("Model retraining completed successfully")
        except Exception as e:
            logger.error(f"Error in model retraining: {str(e)}")
            self.model_status["is_training"] = False

    def optimize_hyperparameters(self):
        """Optimize model hyperparameters using automated techniques."""
        try:
            # Implement hyperparameter optimization logic
            # This could use techniques like Bayesian Optimization
            logger.info("Starting hyperparameter optimization")
            
            # Placeholder for optimization logic
            optimal_params = {
                "learning_rate": 0.01,
                "num_factors": 100,
                "regularization": 0.01
            }
            
            return optimal_params
        except Exception as e:
            logger.error(f"Error in hyperparameter optimization: {str(e)}")
            return None

    def automated_feature_selection(self):
        """Perform automated feature selection."""
        try:
            # Implement feature selection logic
            # This could use techniques like LASSO, Random Forest importance, etc.
            logger.info("Starting automated feature selection")
            
            # Placeholder for feature selection logic
            selected_features = [
                "user_interaction_count",
                "item_popularity",
                "category_preference"
            ]
            
            return selected_features
        except Exception as e:
            logger.error(f"Error in feature selection: {str(e)}")
            return []

    def generate_performance_report(self) -> Optional[Dict]:
        """Generate a comprehensive performance report."""
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "model_status": self.get_model_status(),
                "performance_trends": self._calculate_performance_trends(),
                "recommendations": self._generate_improvement_recommendations()
            }
            
            return report
        except Exception as e:
            logger.error(f"Error generating performance report: {str(e)}")
            return None

    def _calculate_performance_trends(self) -> Dict:
        """Calculate performance trends over time."""
        # Implement trend calculation logic
        return {
            "precision_trend": "stable",
            "recall_trend": "improving",
            "latency_trend": "stable"
        }

    def _generate_improvement_recommendations(self) -> List[str]:
        """Generate recommendations for model improvement."""
        return [
            "Consider increasing model complexity",
            "Collect more user interaction data",
            "Optimize feature engineering pipeline"
        ]

    def record_feedback(self, user_id: str, item_id: str, interaction_type: str,
                       rating: Optional[float] = None, timestamp: Optional[datetime] = None,
                       metadata: Optional[Dict] = None):
        """Record user feedback and interaction data."""
        try:
            feedback_data = {
                "user_id": user_id,
                "item_id": item_id,
                "interaction_type": interaction_type,
                "rating": rating,
                "timestamp": timestamp or datetime.now(),
                "metadata": metadata or {}
            }
            
            # Add to buffer
            self.feedback_buffer[user_id].append(feedback_data)
            
            # Flush if threshold reached
            if len(self.feedback_buffer[user_id]) >= self.feedback_flush_threshold:
                self._flush_feedback(user_id)
            
            # Log to MLflow
            with mlflow.start_run(nested=True):
                mlflow.log_metric(f"interaction_{interaction_type}", 1)
                if rating:
                    mlflow.log_metric("rating", rating)
            
            logger.info(f"Recorded feedback from user {user_id} for item {item_id}")
        except Exception as e:
            logger.error(f"Error recording feedback: {str(e)}")
            raise

    def _flush_feedback(self, user_id: str):
        """Flush feedback buffer to disk."""
        try:
            if not self.feedback_buffer[user_id]:
                return
                
            feedback_file = os.path.join(
                self.feedback_dir,
                f"feedback_{datetime.now().strftime('%Y%m%d')}.jsonl"
            )
            
            # Write feedback data
            with open(feedback_file, 'a') as f:
                for feedback in self.feedback_buffer[user_id]:
                    json.dump(feedback, f)
                    f.write('\n')
            
            # Clear buffer
            self.feedback_buffer[user_id] = []
            
        except Exception as e:
            logger.error(f"Error flushing feedback: {str(e)}")

    def get_model_metrics(self) -> Dict[str, float]:
        """Get current model performance metrics."""
        try:
            metrics = {
                "precision": self.model_status["performance_metrics"].get("average_precision", 0.0),
                "recall": self.model_status["performance_metrics"].get("recall", 0.0),
                "ndcg": self.model_status["performance_metrics"].get("ndcg", 0.0),
                "map": self.model_status["performance_metrics"].get("map", 0.0),
                "latency": self._calculate_average_latency(),
                "last_updated": self.model_status["last_evaluation"] or datetime.now()
            }
            return metrics
        except Exception as e:
            logger.error(f"Error getting model metrics: {str(e)}")
            raise

    def get_popular_items(self, timeframe: str = "day", limit: int = 10) -> List[Dict]:
        """Get most popular items within a timeframe."""
        try:
            # Load feedback data
            feedback_data = self._load_feedback_data(timeframe)
            
            # Calculate popularity
            item_counts = defaultdict(lambda: {"count": 0, "rating": 0.0, "interactions": defaultdict(int)})
            
            for feedback in feedback_data:
                item_id = feedback["item_id"]
                item_counts[item_id]["count"] += 1
                if feedback.get("rating"):
                    item_counts[item_id]["rating"] += feedback["rating"]
                item_counts[item_id]["interactions"][feedback["interaction_type"]] += 1
            
            # Calculate average ratings and sort
            popular_items = []
            for item_id, stats in item_counts.items():
                avg_rating = stats["rating"] / stats["count"] if stats["count"] > 0 else 0
                popular_items.append({
                    "item_id": item_id,
                    "total_interactions": stats["count"],
                    "average_rating": avg_rating,
                    "interaction_breakdown": dict(stats["interactions"])
                })
            
            # Sort by total interactions
            popular_items.sort(key=lambda x: x["total_interactions"], reverse=True)
            
            return popular_items[:limit]
        except Exception as e:
            logger.error(f"Error getting popular items: {str(e)}")
            return []

    def _load_feedback_data(self, timeframe: str) -> List[Dict]:
        """Load feedback data for the specified timeframe."""
        try:
            end_date = datetime.now()
            if timeframe == "day":
                start_date = end_date - timedelta(days=1)
            elif timeframe == "week":
                start_date = end_date - timedelta(weeks=1)
            elif timeframe == "month":
                start_date = end_date - timedelta(days=30)
            else:
                raise ValueError(f"Invalid timeframe: {timeframe}")
            
            feedback_data = []
            for filename in os.listdir(self.feedback_dir):
                if not filename.startswith("feedback_"):
                    continue
                    
                file_date = datetime.strptime(filename.split("_")[1].split(".")[0], "%Y%m%d")
                if start_date <= file_date <= end_date:
                    with open(os.path.join(self.feedback_dir, filename), 'r') as f:
                        for line in f:
                            feedback = json.loads(line.strip())
                            feedback_data.append(feedback)
            
            return feedback_data
        except Exception as e:
            logger.error(f"Error loading feedback data: {str(e)}")
            return []

    def get_model_version(self) -> str:
        """Get current model version."""
        return self.model_status["model_version"]

    def get_last_training_time(self) -> Optional[str]:
        """Get timestamp of last model training."""
        return self.model_status["last_training"].isoformat() if self.model_status["last_training"] else None

    def get_health_metrics(self) -> Dict[str, Union[str, float]]:
        """Get model health metrics."""
        try:
            # Calculate basic health metrics
            metrics = {
                "status": self.model_status["health_status"],
                "uptime": self._calculate_uptime(),
                "error_rate": self._calculate_error_rate(),
                "latency_p95": self._calculate_latency_percentile(95),
                "memory_usage": self._get_memory_usage(),
                "prediction_drift": self._calculate_prediction_drift()
            }
            return metrics
        except Exception as e:
            logger.error(f"Error getting health metrics: {str(e)}")
            return {"status": "error", "error": str(e)}

    def _calculate_uptime(self) -> float:
        """Calculate model uptime in hours."""
        if not self.model_status["last_training"]:
            return 0.0
        return (datetime.now() - self.model_status["last_training"]).total_seconds() / 3600

    def _calculate_error_rate(self) -> float:
        """Calculate recent error rate."""
        try:
            # Load recent metrics
            recent_metrics = self._load_recent_metrics()
            if not recent_metrics:
                return 0.0
            
            error_counts = sum(1 for m in recent_metrics if m.get("error_count", 0) > 0)
            return error_counts / len(recent_metrics)
        except Exception:
            return 0.0

    def _calculate_latency_percentile(self, percentile: int) -> float:
        """Calculate latency percentile."""
        try:
            recent_metrics = self._load_recent_metrics()
            if not recent_metrics:
                return 0.0
            
            latencies = [m.get("latency", 0) for m in recent_metrics]
            return float(np.percentile(latencies, percentile))
        except Exception:
            return 0.0

    def _calculate_average_latency(self) -> float:
        """Calculate average latency."""
        try:
            recent_metrics = self._load_recent_metrics()
            if not recent_metrics:
                return 0.0
            
            latencies = [m.get("latency", 0) for m in recent_metrics]
            return float(np.mean(latencies))
        except Exception:
            return 0.0

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0

    def _calculate_prediction_drift(self) -> float:
        """Calculate prediction distribution drift."""
        try:
            # Load historical and recent predictions
            historical = self._load_historical_predictions()
            recent = self._load_recent_predictions()
            
            if not historical or not recent:
                return 0.0
            
            # Calculate KL divergence
            hist_dist = np.histogram(historical, bins=20, density=True)[0]
            recent_dist = np.histogram(recent, bins=20, density=True)[0]
            
            # Add small epsilon to avoid division by zero
            epsilon = 1e-10
            hist_dist += epsilon
            recent_dist += epsilon
            
            return float(stats.entropy(recent_dist, hist_dist))
        except Exception:
            return 0.0

    def _load_recent_metrics(self) -> List[Dict]:
        """Load metrics from the last 24 hours."""
        try:
            metrics = []
            current_time = datetime.now()
            
            for filename in os.listdir(self.metrics_dir):
                if not filename.startswith("metrics_"):
                    continue
                
                file_time = datetime.strptime(filename.split("_")[1].split(".")[0], "%Y%m%d_%H%M%S")
                if current_time - file_time <= timedelta(hours=24):
                    with open(os.path.join(self.metrics_dir, filename), 'r') as f:
                        metrics.append(json.load(f))
            
            return metrics
        except Exception:
            return []

    def log_recommendation_request(self, user_id: str, recommendations: List[Dict],
                                 context: Optional[Dict] = None):
        """Log recommendation request for analytics."""
        try:
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id,
                "recommendations": recommendations,
                "context": context or {},
                "model_version": self.get_model_version()
            }
            
            # Log to MLflow
            with mlflow.start_run(nested=True):
                mlflow.log_metric("recommendation_count", len(recommendations))
                mlflow.log_params(context or {})
            
            # Save to disk
            log_file = os.path.join(
                self.metrics_dir,
                f"recommendations_{datetime.now().strftime('%Y%m%d')}.jsonl"
            )
            
            with open(log_file, 'a') as f:
                json.dump(log_data, f)
                f.write('\n')
                
        except Exception as e:
            logger.error(f"Error logging recommendation request: {str(e)}")
