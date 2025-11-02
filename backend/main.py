from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from model import load_model, predict
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global dictionary to store models
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown"""
    logger.info("Loading model on startup...")
    try:
        # Path to your fine-tuned checkpoint
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, "model_checkpoint", "checkpoint-3056")
        
        # Base model name for tokenizer - adjust if you used a different variant
        base_model_name = "distilbert-base-uncased"
        # Or if you used: "distilbert-base-cased"
        
        logger.info(f"Checkpoint path: {model_path}")
        logger.info(f"Base model for tokenizer: {base_model_name}")
        
        # Load model and tokenizer
        model, tokenizer = load_model(model_path, base_model_name)
        ml_models["model"] = model
        ml_models["tokenizer"] = tokenizer
        logger.info("Model and tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        ml_models["model"] = None
        ml_models["tokenizer"] = None
    
    yield
    
    logger.info("Shutting down and cleaning up resources...")
    ml_models.clear()

app = FastAPI(
    title="WAF Transformer API",
    version="1.0.0",
    lifespan=lifespan
)

class DetectionRequest(BaseModel):
    sequence: str

class DetectionResponse(BaseModel):
    anomaly: bool
    score: float
    sequence: str

@app.get("/")
async def root():
    return {"message": "WAF Transformer API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": ml_models.get("model") is not None,
        "tokenizer_loaded": ml_models.get("tokenizer") is not None
    }

@app.post("/detect", response_model=DetectionResponse)
async def detect(request: DetectionRequest):
    model = ml_models.get("model")
    tokenizer = ml_models.get("tokenizer")
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        is_anomaly, score = predict(model, tokenizer, request.sequence)
        
        logger.info(f"Detection: {request.sequence[:50]}... -> {is_anomaly} (score: {score:.3f})")
        
        return DetectionResponse(
            anomaly=is_anomaly,
            score=score,
            sequence=request.sequence
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
