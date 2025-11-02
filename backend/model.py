import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import logging

logger = logging.getLogger(__name__)

def load_model(model_path: str, base_model_name: str = "distilbert-base-uncased"):
    """Load DistilBERT model from checkpoint and tokenizer from base model"""
    try:
        logger.info(f"Loading model from: {model_path}")
        logger.info(f"Loading tokenizer from: {base_model_name}")
        
        # Convert to absolute path if relative
        if model_path.startswith('./') or model_path.startswith('.\\'):
            model_path = os.path.abspath(model_path)
        
        # Verify checkpoint path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        # Load tokenizer from base model
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )
        
        # Load fine-tuned DistilBERT model from checkpoint
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True
        )
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        logger.info(f"Model loaded successfully on {device}")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def predict(model, tokenizer, sequence: str):
    """Run prediction on input sequence for vulnerability detection"""
    try:
        # Tokenize input
        inputs = tokenizer(
            sequence,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Move to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            # For binary classification (anomaly detection)
            probs = torch.softmax(logits, dim=-1)
            
            # Assuming class 1 is "anomaly/vulnerable"
            anomaly_prob = probs[0, 1].item()
            is_anomaly = anomaly_prob > 0.5
        
        return is_anomaly, anomaly_prob
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise
