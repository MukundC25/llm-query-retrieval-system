#!/usr/bin/env python3
"""
Test the fixed system to verify it's working properly
"""
import requests
import json
import time

def test_fixed_system():
    """Test the fixed system with a simple question"""
    
    url = "https://llm-query-retrieval-system-production.up.railway.app/api/v1/hackrx/run"
    
    headers = {
        "Authorization": "Bearer 12776c804e23764323a141d7736af662e2e2d41a9deaf12e331188a32e1c299f",
        "Content-Type": "application/json"
    }
    
    # Test with BAJAJ document
    test_doc = "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/BAJHLIP23020V012223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D"
    
    # Simple test questions
    test_questions = [
        "Does this policy cover AYUSH treatments?",
        "What are the exclusions in this policy?",
        "What type of insurance policy is this?"
    ]
    
    print("🔧 TESTING FIXED SYSTEM")
    print("=" * 40)
    
    payload = {
        "documents": test_doc,
        "questions": test_questions
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        end_time = time.time()
        
        processing_time = end_time - start_time
        print(f"⏱️  Response time: {processing_time:.2f} seconds")
        print(f"📊 Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            answers = data.get('answers', [])
            
            print(f"\n📝 RESULTS:")
            print("=" * 40)
            
            working_count = 0
            
            for i, (question, answer) in enumerate(zip(test_questions, answers)):
                print(f"\n❓ Q{i+1}: {question}")
                print(f"💬 A{i+1}: {answer}")
                
                # Check if it's working (not an error and has content)
                answer_lower = answer.lower()
                is_working = (
                    "error" not in answer_lower and 
                    "quota" not in answer_lower and
                    "429" not in answer_lower and
                    len(answer.strip()) > 10
                )
                
                if is_working:
                    print(f"✅ WORKING")
                    working_count += 1
                else:
                    print(f"❌ NOT WORKING")
                
                print("-" * 40)
            
            print(f"\n🏆 SYSTEM STATUS:")
            print(f"✅ Working responses: {working_count}/{len(test_questions)}")
            print(f"⏱️  Response time: {processing_time:.2f}s")
            
            if working_count >= 2:
                print("🎉 SYSTEM IS WORKING! Back to normal operation.")
                return True
            elif working_count >= 1:
                print("⚠️  SYSTEM PARTIALLY WORKING - Some issues remain")
                return False
            else:
                print("❌ SYSTEM STILL HAS ISSUES")
                return False
            
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_fixed_system()
    if success:
        print("\n🚀 SYSTEM FIXED AND READY!")
    else:
        print("\n🔧 SYSTEM STILL NEEDS WORK")
