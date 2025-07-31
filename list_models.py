#!/usr/bin/env python3
"""
List available Gemini models
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def list_gemini_models():
    """List available Gemini models"""
    print("Listing available Gemini models...")
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key or api_key == 'your_gemini_api_key_here':
        print("❌ Invalid API key")
        return
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        print("Available models:")
        for model in genai.list_models():
            if 'generateContent' in model.supported_generation_methods:
                print(f"✅ {model.name} - {model.display_name}")
        
        # Test with the correct model name
        print("\nTesting with gemini-1.5-flash...")
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Say hello")
        print(f"✅ Test successful: {response.text}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    list_gemini_models()
