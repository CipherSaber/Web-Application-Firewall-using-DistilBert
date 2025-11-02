import streamlit as st
import requests
import json
import time
import os
from openai import OpenAI

st.set_page_config(
    page_title="WAF Transformer ",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ WAF Transformer ")
st.markdown("**Real-time Web Application Firewall using Transformer-based AI**")

# Hardcoded API key (not displayed in UI)
OPENROUTER_API_KEY = "YOUR API KEY"

# Sidebar for configuration
st.sidebar.title("Configuration")
backend_url = st.sidebar.text_input("Backend URL", value="http://localhost:8080")
threshold_display = st.sidebar.slider("Display Threshold", 0.0, 1.0, 0.5, 0.01)

# OpenRouter configuration
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 AI Explanation")
ai_model = st.sidebar.selectbox("AI Model", [
    "mistralai/devstral-small"
], index=0)

# Initialize OpenRouter client
def get_ai_explanation(request_sequence: str, is_anomaly: bool, score: float):
    """Get AI explanation for the detection result using OpenRouter"""
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
            default_headers={
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "WAF Transformer MVP"
            }
        )
        
        prompt = f"""You are a cybersecurity expert analyzing HTTP requests for a Web Application Firewall.

**HTTP Request:**
{request_sequence}

**Detection Result:**
- Classification: {"Malicious" if is_anomaly else "Benign"}
- Confidence Score: {score:.3f}

Please provide:
1. A detailed explanation of why this request was classified as {"malicious" if is_anomaly else "benign"}
2. Specific security concerns or attack patterns detected (if malicious)
3. Recommended actions for security teams
4. Any false positive considerations (if applicable)

Keep your response clear, technical, and actionable."""

        response = client.chat.completions.create(
            model=ai_model,
            messages=[
                {"role": "system", "content": "You are an expert cybersecurity analyst specializing in web application security and HTTP request analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"❌ Error getting AI explanation: {str(e)}"

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("HTTP Request Analysis")
    
    # Manual entry only
    request_sequence = st.text_area(
        "Enter HTTP Request Sequence:",
        height=150,
        placeholder="Example: GET /index.html Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    )

    # Detection button
    if st.button("🔍 Analyze Request", type="primary"):
        if request_sequence:
            with st.spinner("Analyzing request..."):
                try:
                    # Send request to FastAPI backend
                    url = f"{backend_url}/detect"
                    payload = {"sequence": request_sequence}
                    
                    start_time = time.time()
                    response = requests.post(url, json=payload, timeout=10)
                    end_time = time.time()
                    
                    if response.status_code == 200:
                        result = response.json()
                        anomaly = result["anomaly"]
                        score = result["score"]
                        
                        # Display results
                        st.subheader("🔍 Detection Results")
                        
                        col_result1, col_result2, col_result3 = st.columns(3)
                        
                        with col_result1:
                            if anomaly:
                                st.error(f"🚨 **MALICIOUS**")
                            else:
                                st.success(f"✅ **BENIGN**")
                        
                        with col_result2:
                            st.metric("Confidence Score", f"{score:.3f}")
                        
                        with col_result3:
                            st.metric("Response Time", f"{(end_time - start_time)*1000:.0f} ms")
                        
                        # Progress bar for score
                        st.progress(score)
                        
                        # Detailed analysis
                        st.subheader("📊 Detailed Analysis")
                        analysis_data = {
                            "Request": request_sequence,
                            "Classification": "Malicious" if anomaly else "Benign",
                            "Confidence Score": f"{score:.3f}",
                            "Risk Level": "High" if score > 0.8 else "Medium" if score > 0.5 else "Low",
                            "Processing Time": f"{(end_time - start_time)*1000:.0f} ms"
                        }
                        
                        for key, value in analysis_data.items():
                            st.write(f"**{key}:** {value}")
                        
                        # AI Explanation Section
                        st.markdown("---")
                        st.subheader("🤖 AI Security Analysis (Mistral Devstral)")
                        
                        with st.spinner("Generating AI explanation..."):
                            explanation = get_ai_explanation(request_sequence, anomaly, score)
                            st.markdown(explanation)
                            
                    else:
                        st.error(f"❌ Backend Error: {response.status_code}")
                        st.write(response.text)
                        
                except requests.exceptions.ConnectionError:
                    st.error("❌ **Cannot connect to backend server**")
                    st.info(f"Make sure FastAPI is running at {backend_url}")
                    
                except requests.exceptions.Timeout:
                    st.error("⏱️ **Request timeout**")
                    st.info("The backend server took too long to respond")
                    
                except Exception as e:
                    st.error(f"❌ **Unexpected error:** {str(e)}")
        else:
            st.warning("⚠️ Please enter a request sequence")

with col2:
    st.subheader("ℹ️ Information")
    
    # System status
    try:
        health_response = requests.get(f"{backend_url}/health", timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            st.success("🟢 Backend Online")
            st.write(f"**Model Loaded:** {'✅' if health_data.get('model_loaded') else '❌'}")
            st.write(f"**AI Model:** {ai_model}")
        else:
            st.error("🔴 Backend Offline")
    except:
        st.error("🔴 Backend Offline")
    
    st.markdown("---")
    
    # Help section
    with st.expander("📖 How to Use"):
        st.markdown("""
        1. Enter your HTTP request sequence in the text area
        2. Click **Analyze Request** to get real-time detection
        3. View the classification result and confidence score
        4. Read the AI-powered security explanation
        
        **Request Format:**
        ```
        METHOD /path User-Agent
        ```
        
        **Example:**
        ```
        GET /index.html Mozilla/5.0
        ```
        """)
    
    with st.expander("🎯 About the Model"):
        st.markdown("""
        - **Detection Model**: DistilBERT-based Transformer
        - **AI Explanation**: Mistral Devstral (Code-focused AI)
        - **Training**: CSIC 2010 HTTP Dataset
        - **Features**: Real-time detection, GPU acceleration
        - **Accuracy**: Trained on 60K+ HTTP requests
        """)
    
    with st.expander("🤖 About Mistral Devstral"):
        st.markdown("""
        **Mistral Devstral** is a code-specialized AI model:
        - Optimized for technical analysis
        - Fast inference (small variant)
        - Excellent for security pattern recognition
        - Understands HTTP protocols and web attacks
        """)

# Footer
st.markdown("---")
st.markdown("*WAF Transformer - Powered by DistilBERT, FastAPI, and Mistral Devstral*")
