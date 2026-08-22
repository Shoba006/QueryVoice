import requests
import json

BASE_URL = "http://localhost:8000"

def test_endpoints():
    print("Testing /health endpoint...")
    r = requests.get(f"{BASE_URL}/health")
    print("Health Status Code:", r.status_code)
    print("Health Response:", r.json())

    print("\nTesting /ask text query endpoint...")
    payload = {
        "text_query": "भारत की राजधानी क्या है?",
        "chunk_strategy": "semantic"
    }
    r = requests.post(f"{BASE_URL}/ask", data=payload)
    print("Ask Status Code:", r.status_code)
    resp = r.json()
    print("\nTesting /ask text query endpoint (Warm Pipeline Call)...")
    r2 = requests.post(f"{BASE_URL}/ask", data=payload)
    resp2 = r2.json()
    print("Warm Call Answer:", resp2.get("answer"))
    print("Warm Call Latency Breakdown:", resp2.get("latency_breakdown"))

if __name__ == "__main__":
    try:
        test_endpoints()
    except Exception as e:
        print("Server test error:", e)
