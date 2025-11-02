import requests
import json

# Test cases
test_cases = {
    "benign": [
        "GET /index.html Mozilla/5.0",
        "POST /login Mozilla/5.0",
        "GET /home Mozilla/5.0",
        "GET /contact Mozilla/5.0",
        "GET /images/logo.png Mozilla/5.0"
    ],
    "malicious": [
        "GET /etc/passwd Mozilla/5.0",
        "POST /admin/delete?id=1234 Chrome/94.0",
        "GET /wp-login.php Mozilla/5.0",
        "POST /api/delete_user HTTP/1.1 Chrome/90.0",
        "GET /admin/config.yaml Mozilla/5.0"
    ]
}

def test_api(base_url="http://localhost:8080"):
    print("🧪 Testing WAF Transformer API\n")
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Status: {response.json()}")
        else:
            print("❌ Health check failed")
            return
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        return
    
    print("\n" + "="*50)
    
    # Test benign requests
    print("🟢 Testing BENIGN requests:")
    for i, request in enumerate(test_cases["benign"], 1):
        try:
            response = requests.post(f"{base_url}/detect", json={"sequence": request})
            if response.status_code == 200:
                result = response.json()
                status = "✅" if not result["anomaly"] else "❌"
                print(f"{status} Test {i}: {result['anomaly']} (score: {result['score']:.3f})")
                print(f"   Request: {request[:60]}...")
            else:
                print(f"❌ Test {i}: API Error {response.status_code}")
        except Exception as e:
            print(f"❌ Test {i}: {e}")
    
    print("\n" + "="*50)
    
    # Test malicious requests
    print("🔴 Testing MALICIOUS requests:")
    for i, request in enumerate(test_cases["malicious"], 1):
        try:
            response = requests.post(f"{base_url}/detect", json={"sequence": request})
            if response.status_code == 200:
                result = response.json()
                status = "✅" if result["anomaly"] else "❌"
                print(f"{status} Test {i}: {result['anomaly']} (score: {result['score']:.3f})")
                print(f"   Request: {request[:60]}...")
            else:
                print(f"❌ Test {i}: API Error {response.status_code}")
        except Exception as e:
            print(f"❌ Test {i}: {e}")

if __name__ == "__main__":
    test_api()
