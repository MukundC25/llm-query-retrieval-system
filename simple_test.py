#!/usr/bin/env python3
"""
Simple test to check Gemini API
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_gemini_api():
    """Test Gemini API directly"""
    print("Testing Gemini API...")
    
    # Check API key
    api_key = os.getenv('GEMINI_API_KEY')
    print(f"API Key: {api_key[:10]}..." if api_key else "No API key found")
    
    if not api_key or api_key == 'your_gemini_api_key_here':
        print("❌ Invalid API key")
        return False
    
    try:
        import google.generativeai as genai
        print("✅ Gemini library imported")
        
        genai.configure(api_key=api_key)
        print("✅ Gemini configured")
        
        model = genai.GenerativeModel('gemini-pro')
        print("✅ Model created")
        
        # Test simple generation
        response = model.generate_content("Say hello")
        print(f"✅ API test successful: {response.text}")
        
        return True
        
    except Exception as e:
        print(f"❌ Gemini API error: {e}")
        return False

if __name__ == "__main__":
    test_gemini_api()
