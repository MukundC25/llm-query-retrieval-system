"""
Test script for the LLM Query-Retrieval System API
"""

import requests
import json
import time
import os
from typing import Dict, Any

# Configuration
API_BASE_URL = "http://localhost:8000"  # Change this to your deployed URL
BEARER_TOKEN = "12776c804e23764323a141d7736af662e2e2d41a9deaf12e331188a32e1c299f"

# Sample test data
SAMPLE_DOCUMENT_URL = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"

SAMPLE_QUESTIONS = [
    "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
    "What is the waiting period for pre-existing diseases (PED) to be covered?",
    "Does this policy cover maternity expenses, and what are the conditions?",
    "What is the waiting period for cataract surgery?",
    "Are the medical expenses for an organ donor covered under this policy?",
    "What is the No Claim Discount (NCD) offered in this policy?",
    "Is there a benefit for preventive health check-ups?",
    "How does the policy define a 'Hospital'?",
    "What is the extent of coverage for AYUSH treatments?",
    "Are there any sub-limits on room rent and ICU charges for Plan A?"
]

def make_request(endpoint: str, method: str = "GET", data: Dict[Any, Any] = None) -> Dict[Any, Any]:
    """Make HTTP request to API"""
    
    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}",
        "Content-Type": "application/json"
    }
    
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=data)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response text: {e.response.text}")
        return {"error": str(e)}

def test_health_check():
    """Test health check endpoint"""
    print("Testing health check...")
    
    result = make_request("/")
    print(f"Health check result: {result}")
    
    return result.get("status") == "healthy"

def test_main_endpoint():
    """Test main query processing endpoint"""
    print("Testing main endpoint...")
    
    payload = {
        "documents": SAMPLE_DOCUMENT_URL,
        "questions": SAMPLE_QUESTIONS
    }
    
    start_time = time.time()
    result = make_request("/api/v1/hackrx/run", "POST", payload)
    end_time = time.time()
    
    processing_time = end_time - start_time
    print(f"Processing time: {processing_time:.2f} seconds")
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return False
    
    if "answers" not in result:
        print("Error: No 'answers' field in response")
        return False
    
    answers = result["answers"]
    print(f"Received {len(answers)} answers")
    
    # Print first few answers
    for i, answer in enumerate(answers[:3]):
        print(f"Q{i+1}: {SAMPLE_QUESTIONS[i]}")
        print(f"A{i+1}: {answer}")
        print("-" * 50)
    
    return len(answers) == len(SAMPLE_QUESTIONS)

def test_single_question():
    """Test with a single question"""
    print("Testing single question...")
    
    payload = {
        "documents": SAMPLE_DOCUMENT_URL,
        "questions": ["What is the grace period for premium payment?"]
    }
    
    start_time = time.time()
    result = make_request("/api/v1/hackrx/run", "POST", payload)
    end_time = time.time()
    
    processing_time = end_time - start_time
    print(f"Single question processing time: {processing_time:.2f} seconds")
    
    if "answers" in result:
        print(f"Answer: {result['answers'][0]}")
        return True
    
    return False

def test_invalid_token():
    """Test with invalid token"""
    print("Testing invalid token...")
    
    headers = {
        "Authorization": "Bearer invalid_token",
        "Content-Type": "application/json"
    }
    
    payload = {
        "documents": SAMPLE_DOCUMENT_URL,
        "questions": ["Test question"]
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/hackrx/run",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 401:
            print("Correctly rejected invalid token")
            return True
        else:
            print(f"Unexpected status code: {response.status_code}")
            return False
    
    except Exception as e:
        print(f"Error testing invalid token: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("LLM Query-Retrieval System API Tests")
    print("=" * 60)
    
    tests = [
        ("Health Check", test_health_check),
        ("Invalid Token", test_invalid_token),
        ("Single Question", test_single_question),
        ("Main Endpoint", test_main_endpoint),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        
        try:
            success = test_func()
            results[test_name] = success
            print(f"Result: {'PASS' if success else 'FAIL'}")
        except Exception as e:
            print(f"Error: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    
    for test_name, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"{test_name}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")

if __name__ == "__main__":
    # Check if API URL is provided via environment variable
    if "API_URL" in os.environ:
        API_BASE_URL = os.environ["API_URL"]
        print(f"Using API URL from environment: {API_BASE_URL}")
    
    run_all_tests()
