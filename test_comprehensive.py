#!/usr/bin/env python3
"""
Comprehensive test with the exact hackathon questions
"""
import requests
import json
import time

def test_comprehensive():
    """Test with the exact questions from hackathon requirements"""
    
    url = "https://llm-query-retrieval-system-production.up.railway.app/api/v1/hackrx/run"
    
    headers = {
        "Authorization": "Bearer 12776c804e23764323a141d7736af662e2e2d41a9deaf12e331188a32e1c299f",
        "Content-Type": "application/json"
    }
    
    # Exact payload from hackathon requirements
    payload = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
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
    }
    
    print("🧪 Testing with Comprehensive Hackathon Questions...")
    print(f"URL: {url}")
    print(f"Questions: {len(payload['questions'])}")
    
    start_time = time.time()
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=35)
        end_time = time.time()
        
        processing_time = end_time - start_time
        print(f"⏱️  Response time: {processing_time:.2f} seconds")
        print(f"📊 Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Success!")
            print(f"📝 Answers received: {len(data.get('answers', []))}")
            
            # Count how many answers contain actual information vs errors
            informative_answers = 0
            error_answers = 0
            
            for i, answer in enumerate(data.get('answers', [])):
                print(f"\n❓ Q{i+1}: {payload['questions'][i]}")
                print(f"💬 A{i+1}: {answer}")
                
                if "not available" in answer.lower() or "error" in answer.lower():
                    error_answers += 1
                else:
                    informative_answers += 1
            
            print(f"\n📊 Results Summary:")
            print(f"✅ Informative answers: {informative_answers}")
            print(f"❌ Error/Not available: {error_answers}")
            print(f"🎯 Success rate: {(informative_answers/len(payload['questions'])*100):.1f}%")
            
            return informative_answers > 0
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.Timeout:
        print("❌ Request timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_comprehensive()
    if success:
        print("\n🎉 API is providing informative answers!")
    else:
        print("\n💥 API needs improvement in answer quality!")
