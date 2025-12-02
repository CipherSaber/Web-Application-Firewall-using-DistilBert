# Web Application Firewall using DistilBERT 🛡️

A machine learning-powered Web Application Firewall that leverages DistilBERT to detect and prevent malicious web requests in real-time.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-311/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg)](https://fastapi.tiangolo.com)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-orange)](https://huggingface.co/spaces)

## 🌟 Features

- **Real-time Detection**: Instant analysis of incoming HTTP requests
- **High Accuracy**: Fine-tuned DistilBERT model for precise threat detection
- **Easy Integration**: RESTful API interface using FastAPI
- **Interactive Dashboard**: Streamlit-based monitoring interface
- **Low Latency**: Optimized for production environments

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Virtual environment (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/CipherSaber/Web-Application-Firewall-using-DistilBert.git
cd Web-Application-Firewall-using-DistilBert

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🏃‍♂️ Running the Application

### Start the Backend Server

```bash
uvicorn backend.main:app --reload
```

### Launch the Dashboard

```bash
streamlit run frontend/dashboard.py
```

## 🔧 Architecture

```
├── backend/
│   ├── main.py      # FastAPI server
│   ├── model.py     # ML model implementation
│   └── train.py     # Training pipeline
├── frontend/
│   └── dashboard.py # Streamlit dashboard
└── model_checkpoint/# Trained model files
```

## 🤖 Model Details

Our WAF uses a fine-tuned version of DistilBERT, optimized for detecting:
- SQL Injection attacks
- Cross-Site Scripting (XSS)
- Path Traversal
- Command Injection
- And other web-based attacks

The model is trained on the CSIC 2010 HTTP Dataset and achieves:
- Accuracy: >95%
- F1-Score: >0.94
- Low false-positive rate: <1%

The model is available to use here (https://huggingface.co/jacpacd/waf-distilbert)

## 🔌 API Endpoints

### Prediction Endpoint
```http
POST /predict
Content-Type: application/json

{
    "request": "GET /admin?id=1 OR 1=1"
}
```

### Response Format
```json
{
    "is_malicious": true,
    "confidence": 0.98,
    "attack_type": "sql_injection",
    "processing_time": "45ms"
}
```

## 📊 Performance Metrics

- Average Response Time: <100ms
- Memory Footprint: ~500MB
- Concurrent Request Handling: Up to 100 req/sec

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests.

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

##  Acknowledgments

- [Hugging Face](https://huggingface.co/) for the Transformers library
- [FastAPI](https://fastapi.tiangolo.com/) for the efficient API framework
- [Streamlit](https://streamlit.io/) for the dashboard interface
- CSIC 2010 HTTP Dataset for training data
