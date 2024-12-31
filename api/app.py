from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import uvicorn
import logging
import sys
import os
from datetime import datetime

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.recommendation_engine import RecommendationEngine
from src.automation_tools import AutomationTools

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI-Powered Recommendation System",
    description="API for personalized recommendations with AI-based automation",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class UserProfile(BaseModel):
    user_id: str
    preferences: Optional[Dict] = Field(default=None, description="User preferences for content-based filtering")
    history: Optional[List[str]] = Field(default=None, description="User interaction history")
    metadata: Optional[Dict] = Field(default=None, description="Additional user metadata")

class RecommendationRequest(BaseModel):
    user_id: str
    num_recommendations: int = Field(default=10, ge=1, le=100)
    filters: Optional[Dict] = Field(default=None, description="Filters to apply to recommendations")
    context: Optional[Dict] = Field(default=None, description="Contextual information for recommendations")

class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[Dict]
    explanation: Optional[str] = None
    metadata: Optional[Dict] = None

class FeedbackRequest(BaseModel):
    user_id: str
    item_id: str
    interaction_type: str = Field(..., description="Type of interaction (click, like, purchase, etc.)")
    rating: Optional[float] = Field(None, ge=0, le=5, description="Optional rating (0-5)")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict] = None

class ModelMetrics(BaseModel):
    precision: float
    recall: float
    ndcg: float
    map: float
    latency: float
    last_updated: datetime

# Initialize recommendation engine and automation tools
recommendation_engine = None
automation_tools = None

@app.on_event("startup")
async def startup_event():
    global recommendation_engine, automation_tools
    try:
        recommendation_engine = RecommendationEngine()
        automation_tools = AutomationTools()
        logger.info("Successfully initialized recommendation engine and automation tools")
    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}")
        raise

@app.get("/")
async def root():
    return {"message": "Welcome to the AI-Powered Recommendation System API"}

@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest, background_tasks: BackgroundTasks):
    try:
        # Get recommendations
        recommendations = recommendation_engine.get_recommendations(
            user_id=request.user_id,
            num_recommendations=request.num_recommendations,
            filters=request.filters
        )
        
        # Log request for analytics in background
        background_tasks.add_task(
            automation_tools.log_recommendation_request,
            user_id=request.user_id,
            recommendations=recommendations,
            context=request.context
        )
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            explanation="Personalized recommendations based on user preferences and behavior",
            metadata={
                "timestamp": datetime.now().isoformat(),
                "model_version": automation_tools.get_model_version(),
                "recommendation_sources": [rec.get("sources", []) for rec in recommendations]
            }
        )
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/user/profile")
async def update_user_profile(profile: UserProfile):
    try:
        recommendation_engine.update_user_profile(
            user_id=profile.user_id,
            preferences=profile.preferences,
            history=profile.history
        )
        return {"message": "User profile updated successfully"}
    except Exception as e:
        logger.error(f"Error updating user profile: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def record_feedback(feedback: FeedbackRequest, background_tasks: BackgroundTasks):
    """Record user feedback and interaction data"""
    try:
        # Record feedback
        automation_tools.record_feedback(
            user_id=feedback.user_id,
            item_id=feedback.item_id,
            interaction_type=feedback.interaction_type,
            rating=feedback.rating,
            timestamp=feedback.timestamp,
            metadata=feedback.metadata
        )
        
        # Update user profile in background
        background_tasks.add_task(
            recommendation_engine.update_user_profile,
            user_id=feedback.user_id,
            history=[feedback.item_id]
        )
        
        return {"message": "Feedback recorded successfully"}
    except Exception as e:
        logger.error(f"Error recording feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/model-performance", response_model=ModelMetrics)
async def get_model_performance():
    """Get current model performance metrics"""
    try:
        metrics = automation_tools.get_model_metrics()
        return ModelMetrics(**metrics)
    except Exception as e:
        logger.error(f"Error getting model performance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/popular-items")
async def get_popular_items(timeframe: str = "day", limit: int = 10):
    """Get most popular items within a timeframe"""
    try:
        popular_items = automation_tools.get_popular_items(timeframe, limit)
        return {"items": popular_items}
    except Exception as e:
        logger.error(f"Error getting popular items: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/status")
async def get_model_status():
    """Get current model status and health metrics"""
    try:
        status = automation_tools.get_model_status()
        return {
            "status": status,
            "last_training": automation_tools.get_last_training_time(),
            "model_version": automation_tools.get_model_version(),
            "health_metrics": automation_tools.get_health_metrics()
        }
    except Exception as e:
        logger.error(f"Error getting model status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/model/retrain")
async def trigger_model_retraining(background_tasks: BackgroundTasks):
    """Trigger model retraining in the background"""
    try:
        background_tasks.add_task(automation_tools.trigger_retraining)
        return {
            "message": "Model retraining triggered successfully",
            "status": "pending",
            "triggered_at": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error triggering model retraining: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
