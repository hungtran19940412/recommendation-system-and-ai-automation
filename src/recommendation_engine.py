import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import logging
from typing import List, Dict, Optional, Tuple
import joblib
import os
from datetime import datetime
import torch
import torch.nn as nn
import yaml
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

class NeuralCF(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int, layers: List[int]):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        layer_dims = [embedding_dim * 2] + layers
        self.layers = nn.ModuleList([
            nn.Linear(layer_dims[i], layer_dims[i+1])
            for i in range(len(layer_dims)-1)
        ])
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(layer_dims[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_input: torch.Tensor, item_input: torch.Tensor) -> torch.Tensor:
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)
        
        x = torch.cat([user_embedded, item_embedded], dim=-1)
        for layer in self.layers:
            x = nn.ReLU()(layer(x))
            x = self.dropout(x)
        
        output = self.sigmoid(self.output(x))
        return output.squeeze()

class RecommendationEngine:
    def __init__(self, model_path: str = "models", config_path: str = "config/model_config.yaml"):
        """Initialize the recommendation engine with multiple recommendation approaches."""
        self.model_path = model_path
        self.config_path = config_path
        self.collaborative_model = None
        self.neural_cf_model = None
        self.content_based_model = None
        self.bert_model = None
        self.bert_tokenizer = None
        self.user_profiles = {}
        self.item_features = {}
        self.load_config()
        self.initialize_models()

    def load_config(self):
        """Load model configuration from yaml file."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            self.config = {}

    def initialize_models(self):
        """Initialize or load pre-trained models."""
        try:
            # Load collaborative filtering models
            collab_model_path = os.path.join(self.model_path, "collaborative_filtering/svd_model.joblib")
            neural_cf_path = os.path.join(self.model_path, "collaborative_filtering/neural_cf.pt")
            
            if os.path.exists(collab_model_path):
                self.collaborative_model = joblib.load(collab_model_path)
            else:
                self.collaborative_model = TruncatedSVD(
                    n_components=self.config.get('collaborative_filtering', {}).get('matrix_factorization', {}).get('n_factors', 100)
                )

            if os.path.exists(neural_cf_path):
                self.neural_cf_model = torch.load(neural_cf_path)
            
            # Initialize BERT model for text processing
            self.bert_model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
            self.bert_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
            
            # Initialize content-based model
            self.load_item_features()
            
            logger.info("Models initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise

    def load_item_features(self):
        """Load item features for content-based filtering."""
        try:
            features_path = os.path.join(self.model_path, "content_based/item_features.joblib")
            if os.path.exists(features_path):
                self.item_features = joblib.load(features_path)
            logger.info("Item features loaded successfully")
        except Exception as e:
            logger.error(f"Error loading item features: {str(e)}")
            self.item_features = {}

    def update_user_profile(self, user_id: str, preferences: Optional[Dict] = None, 
                          history: Optional[List[str]] = None):
        """Update user profile with new preferences and interaction history."""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                "preferences": {},
                "history": [],
                "last_updated": None
            }

        if preferences:
            self.user_profiles[user_id]["preferences"].update(preferences)
        
        if history:
            self.user_profiles[user_id]["history"].extend(history)
            # Keep only recent history
            self.user_profiles[user_id]["history"] = self.user_profiles[user_id]["history"][-1000:]

        self.user_profiles[user_id]["last_updated"] = datetime.now()
        logger.info(f"Updated profile for user {user_id}")

    def get_collaborative_recommendations(self, user_id: str, num_recommendations: int) -> List[Dict]:
        """Generate recommendations using collaborative filtering."""
        try:
            if not self.collaborative_model:
                return []

            # Get user's interaction history
            user_history = self.user_profiles.get(user_id, {}).get("history", [])
            if not user_history:
                return []

            # Generate recommendations using SVD
            user_vector = self._create_user_vector(user_history)
            similar_items = self._find_similar_items(user_vector)

            return self._format_recommendations(similar_items[:num_recommendations])
        except Exception as e:
            logger.error(f"Error in collaborative filtering: {str(e)}")
            return []

    def get_neural_cf_recommendations(self, user_id: str, num_recommendations: int) -> List[Dict]:
        """Generate recommendations using neural collaborative filtering."""
        try:
            if not self.neural_cf_model:
                return []

            user_tensor = torch.tensor([int(user_id)])
            all_items = torch.arange(self.neural_cf_model.item_embedding.num_embeddings)
            
            with torch.no_grad():
                scores = self.neural_cf_model(
                    user_tensor.repeat(len(all_items)), 
                    all_items
                )
            
            # Get top-k items
            top_k = torch.topk(scores, k=num_recommendations)
            recommendations = []
            
            for idx, score in zip(top_k.indices.tolist(), top_k.values.tolist()):
                recommendations.append({
                    "item_id": str(idx),
                    "score": score,
                    "source": "neural_cf"
                })
            
            return recommendations
        except Exception as e:
            logger.error(f"Error in neural collaborative filtering: {str(e)}")
            return []

    def get_content_based_recommendations(self, user_id: str, num_recommendations: int) -> List[Dict]:
        """Generate recommendations using content-based filtering."""
        try:
            user_preferences = self.user_profiles.get(user_id, {}).get("preferences", {})
            if not user_preferences or not self.item_features:
                return []

            # Calculate similarity between user preferences and items
            user_profile_vector = self._create_user_profile_vector(user_preferences)
            similar_items = self._find_similar_items_content(user_profile_vector)

            return self._format_recommendations(similar_items[:num_recommendations])
        except Exception as e:
            logger.error(f"Error in content-based filtering: {str(e)}")
            return []

    def get_recommendations(self, user_id: str, num_recommendations: int = 10, 
                          filters: Optional[Dict] = None) -> List[Dict]:
        """Get hybrid recommendations combining multiple approaches."""
        try:
            # Get recommendations from all approaches
            collab_recommendations = self.get_collaborative_recommendations(
                user_id, num_recommendations)
            neural_recommendations = self.get_neural_cf_recommendations(
                user_id, num_recommendations)
            content_recommendations = self.get_content_based_recommendations(
                user_id, num_recommendations)

            # Combine and rank recommendations
            combined_recommendations = self._combine_recommendations(
                collab_recommendations, neural_recommendations, content_recommendations)

            # Apply filters if any
            if filters:
                combined_recommendations = self._apply_filters(combined_recommendations, filters)

            return combined_recommendations[:num_recommendations]
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return []

    def _create_user_vector(self, user_history: List[str]) -> np.ndarray:
        """Create a user vector from interaction history."""
        # Implementation depends on your specific data structure
        pass

    def _create_user_profile_vector(self, preferences: Dict) -> np.ndarray:
        """Create a user profile vector from preferences."""
        # Implementation depends on your feature representation
        pass

    def _find_similar_items(self, user_vector: np.ndarray) -> List[Dict]:
        """Find items similar to the user vector."""
        # Implementation depends on your similarity metric
        pass

    def _find_similar_items_content(self, profile_vector: np.ndarray) -> List[Dict]:
        """Find items similar to the user profile vector."""
        # Implementation depends on your content representation
        pass

    def _combine_recommendations(self, collab_recs: List[Dict], 
                               neural_recs: List[Dict],
                               content_recs: List[Dict]) -> List[Dict]:
        """Combine and rank recommendations from different approaches."""
        combined = {}
        
        # Get weights from config
        weights = self.config.get('hybrid_model', {}).get('weights', {})
        collab_weight = weights.get('collaborative', 0.4)
        neural_weight = weights.get('neural', 0.3)
        content_weight = weights.get('content_based', 0.3)

        # Combine collaborative filtering recommendations
        for rec in collab_recs:
            item_id = rec["item_id"]
            combined[item_id] = {
                "item_id": item_id,
                "score": rec["score"] * collab_weight,
                "metadata": rec.get("metadata", {}),
                "sources": ["collaborative"]
            }

        # Combine neural CF recommendations
        for rec in neural_recs:
            item_id = rec["item_id"]
            if item_id in combined:
                combined[item_id]["score"] += rec["score"] * neural_weight
                combined[item_id]["sources"].append("neural_cf")
            else:
                combined[item_id] = {
                    "item_id": item_id,
                    "score": rec["score"] * neural_weight,
                    "metadata": rec.get("metadata", {}),
                    "sources": ["neural_cf"]
                }

        # Combine content-based recommendations
        for rec in content_recs:
            item_id = rec["item_id"]
            if item_id in combined:
                combined[item_id]["score"] += rec["score"] * content_weight
                combined[item_id]["sources"].append("content_based")
            else:
                combined[item_id] = {
                    "item_id": item_id,
                    "score": rec["score"] * content_weight,
                    "metadata": rec.get("metadata", {}),
                    "sources": ["content_based"]
                }

        # Sort by score
        sorted_recommendations = sorted(
            combined.values(), 
            key=lambda x: x["score"], 
            reverse=True
        )

        return sorted_recommendations

    def _apply_filters(self, recommendations: List[Dict], filters: Dict) -> List[Dict]:
        """Apply filters to recommendations."""
        filtered_recommendations = []
        
        for rec in recommendations:
            if self._matches_filters(rec, filters):
                filtered_recommendations.append(rec)
                
        return filtered_recommendations

    def _matches_filters(self, recommendation: Dict, filters: Dict) -> bool:
        """Check if a recommendation matches the given filters."""
        metadata = recommendation.get("metadata", {})
        
        for key, value in filters.items():
            if key not in metadata or metadata[key] != value:
                return False
                
        return True

    def save_models(self):
        """Save trained models to disk."""
        try:
            if self.collaborative_model:
                collab_path = os.path.join(self.model_path, "collaborative_filtering/svd_model.joblib")
                os.makedirs(os.path.dirname(collab_path), exist_ok=True)
                joblib.dump(self.collaborative_model, collab_path)

            if self.neural_cf_model:
                neural_cf_path = os.path.join(self.model_path, "collaborative_filtering/neural_cf.pt")
                os.makedirs(os.path.dirname(neural_cf_path), exist_ok=True)
                torch.save(self.neural_cf_model, neural_cf_path)

            if self.item_features:
                features_path = os.path.join(self.model_path, "content_based/item_features.joblib")
                os.makedirs(os.path.dirname(features_path), exist_ok=True)
                joblib.dump(self.item_features, features_path)

            logger.info("Models saved successfully")
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            raise

    def _format_recommendations(self, recommendations: List[Dict]) -> List[Dict]:
        """Format recommendations with consistent structure."""
        formatted = []
        for rec in recommendations:
            formatted.append({
                "item_id": rec["item_id"],
                "score": float(rec["score"]),  # Ensure score is JSON serializable
                "metadata": rec.get("metadata", {}),
                "explanation": rec.get("explanation", "")
            })
        return formatted
