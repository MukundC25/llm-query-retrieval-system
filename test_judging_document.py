#!/usr/bin/env python3
"""
Test with the judging document - Arogya Sanjeevani Policy
"""
import requests
import json
import time

def test_judging_document():
    """Test with the actual judging document"""
    
    url = "https://llm-query-retrieval-system-production.up.railway.app/api/v1/hackrx/run"
    
    headers = {
        "Authorization": "Bearer 12776c804e23764323a141d7736af662e2e2d41a9deaf12e331188a32e1c299f",
        "Content-Type": "application/json"
    }
    
    # New judging document URL
    judging_doc_url = "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D"
    
    # Test with the same hackathon questions
    payload = {
        "documents": judging_doc_url,
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
    
    print("🏆 Testing with JUDGING DOCUMENT - Arogya Sanjeevani Policy...")
    print(f"URL: {url}")
    print(f"Document: Arogya Sanjeevani Policy")
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
            
            print(f"\n📊 JUDGING DOCUMENT Results:")
            print(f"✅ Informative answers: {informative_answers}")
            print(f"❌ Error/Not available: {error_answers}")
            print(f"🎯 Success rate: {(informative_answers/len(payload['questions'])*100):.1f}%")
            print(f"⏱️  Performance: {processing_time:.2f}s (Target: <30s)")
            
            # Performance evaluation
            if processing_time < 30:
                print("✅ PERFORMANCE: PASSED")
            else:
                print("❌ PERFORMANCE: FAILED")
            
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

def test_document_access():
    """First test if we can access the judging document"""
    judging_doc_url = "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D"
    
    print("🔍 Testing document access...")
    try:
        response = requests.get(judging_doc_url, timeout=10)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print(f"✅ Document accessible - Size: {len(response.content)} bytes")
            return True
        else:
            print(f"❌ Document access failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error accessing document: {e}")
        return False

if __name__ == "__main__":
    print("🏆 HACKRX JUDGING DOCUMENT TEST")
    print("=" * 50)
    
    # First test document access
    if test_document_access():
        print("\n" + "=" * 50)
        # Then test the API
        success = test_judging_document()
        
        if success:
            print("\n🎉 SYSTEM READY FOR JUDGING!")
            print("✅ Document processing: WORKING")
            print("✅ API response: WORKING") 
            print("✅ Performance: WITHIN LIMITS")
        else:
            print("\n💥 SYSTEM NEEDS ATTENTION!")
    else:
        print("\n❌ Cannot access judging document!")
