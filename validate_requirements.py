#!/usr/bin/env python3
"""
Requirements Validation Script
Validates that the LLM Query-Retrieval System meets all specified requirements
"""

import requests
import json
import time
import sys
from typing import Dict, List, Any

# Configuration
BEARER_TOKEN = "12776c804e23764323a141d7736af662e2e2d41a9deaf12e331188a32e1c299f"
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

class RequirementsValidator:
    """Validates system requirements"""
    
    def __init__(self, api_url: str):
        self.api_url = api_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {BEARER_TOKEN}",
            "Content-Type": "application/json"
        }
        self.results = {}
    
    def validate_all(self) -> Dict[str, bool]:
        """Run all validation tests"""
        print("🔍 Validating LLM Query-Retrieval System Requirements")
        print("=" * 60)
        
        tests = [
            ("API Endpoint Structure", self.validate_api_structure),
            ("Authentication", self.validate_authentication),
            ("Input Format", self.validate_input_format),
            ("Output Format", self.validate_output_format),
            ("Response Time", self.validate_response_time),
            ("Document Processing", self.validate_document_processing),
            ("Query Processing", self.validate_query_processing),
            ("Error Handling", self.validate_error_handling),
            ("HTTPS Security", self.validate_https),
        ]
        
        for test_name, test_func in tests:
            print(f"\n📋 {test_name}")
            print("-" * 40)
            
            try:
                success = test_func()
                self.results[test_name] = success
                print(f"Result: {'✅ PASS' if success else '❌ FAIL'}")
            except Exception as e:
                print(f"❌ ERROR: {e}")
                self.results[test_name] = False
        
        self.print_summary()
        return self.results
    
    def validate_api_structure(self) -> bool:
        """Validate API endpoint structure"""
        try:
            # Check health endpoint
            response = requests.get(f"{self.api_url}/", headers=self.headers, timeout=10)
            if response.status_code != 200:
                print(f"Health endpoint failed: {response.status_code}")
                return False
            
            # Check main endpoint exists
            test_payload = {
                "documents": SAMPLE_DOCUMENT_URL,
                "questions": ["Test question"]
            }
            
            response = requests.post(
                f"{self.api_url}/api/v1/hackrx/run",
                headers=self.headers,
                json=test_payload,
                timeout=5
            )
            
            # Should not be 404
            if response.status_code == 404:
                print("Main endpoint not found")
                return False
            
            print("✓ API endpoints are accessible")
            return True
            
        except Exception as e:
            print(f"API structure validation failed: {e}")
            return False
    
    def validate_authentication(self) -> bool:
        """Validate bearer token authentication"""
        try:
            # Test with invalid token
            invalid_headers = {
                "Authorization": "Bearer invalid_token",
                "Content-Type": "application/json"
            }
            
            response = requests.get(
                f"{self.api_url}/",
                headers=invalid_headers,
                timeout=10
            )
            
            if response.status_code != 401:
                print(f"Invalid token should return 401, got {response.status_code}")
                return False
            
            # Test with valid token
            response = requests.get(
                f"{self.api_url}/",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code != 200:
                print(f"Valid token should return 200, got {response.status_code}")
                return False
            
            print("✓ Bearer token authentication working")
            return True
            
        except Exception as e:
            print(f"Authentication validation failed: {e}")
            return False
    
    def validate_input_format(self) -> bool:
        """Validate input format requirements"""
        try:
            # Test correct format
            correct_payload = {
                "documents": SAMPLE_DOCUMENT_URL,
                "questions": ["Test question"]
            }
            
            response = requests.post(
                f"{self.api_url}/api/v1/hackrx/run",
                headers=self.headers,
                json=correct_payload,
                timeout=10
            )
            
            # Should not be 422 (validation error)
            if response.status_code == 422:
                print(f"Valid payload rejected: {response.text}")
                return False
            
            # Test invalid format (missing questions)
            invalid_payload = {
                "documents": SAMPLE_DOCUMENT_URL
            }
            
            response = requests.post(
                f"{self.api_url}/api/v1/hackrx/run",
                headers=self.headers,
                json=invalid_payload,
                timeout=10
            )
            
            if response.status_code != 422:
                print(f"Invalid payload should return 422, got {response.status_code}")
                return False
            
            print("✓ Input format validation working")
            return True
            
        except Exception as e:
            print(f"Input format validation failed: {e}")
            return False
    
    def validate_output_format(self) -> bool:
        """Validate output format requirements"""
        try:
            payload = {
                "documents": SAMPLE_DOCUMENT_URL,
                "questions": ["What is the grace period for premium payment?"]
            }
            
            response = requests.post(
                f"{self.api_url}/api/v1/hackrx/run",
                headers=self.headers,
                json=payload,
                timeout=35
            )
            
            if response.status_code != 200:
                print(f"Request failed with status {response.status_code}")
                return False
            
            try:
                data = response.json()
            except json.JSONDecodeError:
                print("Response is not valid JSON")
                return False
            
            # Check required fields
            if "answers" not in data:
                print("Response missing 'answers' field")
                return False
            
            if not isinstance(data["answers"], list):
                print("'answers' field is not a list")
                return False
            
            if len(data["answers"]) != 1:
                print(f"Expected 1 answer, got {len(data['answers'])}")
                return False
            
            print("✓ Output format is correct")
            return True
            
        except Exception as e:
            print(f"Output format validation failed: {e}")
            return False
    
    def validate_response_time(self) -> bool:
        """Validate response time requirement (<30 seconds)"""
        try:
            payload = {
                "documents": SAMPLE_DOCUMENT_URL,
                "questions": SAMPLE_QUESTIONS[:3]  # Test with 3 questions
            }
            
            start_time = time.time()
            response = requests.post(
                f"{self.api_url}/api/v1/hackrx/run",
                headers=self.headers,
                json=payload,
                timeout=35
            )
            end_time = time.time()
            
            response_time = end_time - start_time
            
            if response_time > 30:
                print(f"Response time {response_time:.2f}s exceeds 30s requirement")
                return False
            
            print(f"✓ Response time: {response_time:.2f}s (within 30s requirement)")
            return True
            
        except requests.Timeout:
            print("Request timed out (>30s)")
            return False
        except Exception as e:
            print(f"Response time validation failed: {e}")
            return False
    
    def validate_document_processing(self) -> bool:
        """Validate document processing capabilities"""
        try:
            payload = {
                "documents": SAMPLE_DOCUMENT_URL,
                "questions": ["What type of document is this?"]
            }
            
            response = requests.post(
                f"{self.api_url}/api/v1/hackrx/run",
                headers=self.headers,
                json=payload,
                timeout=35
            )
            
            if response.status_code != 200:
                print(f"Document processing failed: {response.status_code}")
                return False
            
            data = response.json()
            answer = data["answers"][0]
            
            # Check if answer indicates successful processing
            if "error" in answer.lower() and "unable" in answer.lower():
                print(f"Document processing error: {answer}")
                return False
            
            print("✓ Document processing successful")
            return True
            
        except Exception as e:
            print(f"Document processing validation failed: {e}")
            return False
    
    def validate_query_processing(self) -> bool:
        """Validate query processing with sample questions"""
        try:
            payload = {
                "documents": SAMPLE_DOCUMENT_URL,
                "questions": SAMPLE_QUESTIONS
            }
            
            response = requests.post(
                f"{self.api_url}/api/v1/hackrx/run",
                headers=self.headers,
                json=payload,
                timeout=35
            )
            
            if response.status_code != 200:
                print(f"Query processing failed: {response.status_code}")
                return False
            
            data = response.json()
            answers = data["answers"]
            
            if len(answers) != len(SAMPLE_QUESTIONS):
                print(f"Expected {len(SAMPLE_QUESTIONS)} answers, got {len(answers)}")
                return False
            
            # Check if answers are meaningful (not all errors)
            error_count = sum(1 for answer in answers if "error" in answer.lower())
            if error_count > len(answers) * 0.5:  # More than 50% errors
                print(f"Too many error responses: {error_count}/{len(answers)}")
                return False
            
            print(f"✓ Query processing successful: {len(answers)} answers generated")
            return True
            
        except Exception as e:
            print(f"Query processing validation failed: {e}")
            return False
    
    def validate_error_handling(self) -> bool:
        """Validate error handling"""
        try:
            # Test with invalid document URL
            payload = {
                "documents": "https://invalid-url.com/nonexistent.pdf",
                "questions": ["Test question"]
            }
            
            response = requests.post(
                f"{self.api_url}/api/v1/hackrx/run",
                headers=self.headers,
                json=payload,
                timeout=35
            )
            
            # Should still return 200 with error message in answers
            if response.status_code != 200:
                print(f"Error handling failed: {response.status_code}")
                return False
            
            data = response.json()
            if "answers" not in data:
                print("Error response missing answers field")
                return False
            
            print("✓ Error handling working correctly")
            return True
            
        except Exception as e:
            print(f"Error handling validation failed: {e}")
            return False
    
    def validate_https(self) -> bool:
        """Validate HTTPS requirement"""
        if not self.api_url.startswith('https://'):
            print("API URL does not use HTTPS")
            return False
        
        print("✓ HTTPS requirement met")
        return True
    
    def print_summary(self):
        """Print validation summary"""
        print("\n" + "=" * 60)
        print("📊 VALIDATION SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.results)
        passed_tests = sum(self.results.values())
        
        for test_name, success in self.results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{test_name}: {status}")
        
        print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 ALL REQUIREMENTS VALIDATED SUCCESSFULLY!")
        else:
            print("⚠️  Some requirements need attention")

def main():
    """Main validation function"""
    if len(sys.argv) != 2:
        print("Usage: python validate_requirements.py <API_URL>")
        print("Example: python validate_requirements.py https://your-api.vercel.app")
        sys.exit(1)
    
    api_url = sys.argv[1]
    validator = RequirementsValidator(api_url)
    results = validator.validate_all()
    
    # Exit with error code if any tests failed
    if not all(results.values()):
        sys.exit(1)

if __name__ == "__main__":
    main()
