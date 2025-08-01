#!/usr/bin/env python3
"""
Focused accuracy test with one document
"""
import requests
import json
import time

def test_focused_accuracy():
    """Test with focused questions on one document"""
    
    url = "https://llm-query-retrieval-system-production.up.railway.app/api/v1/hackrx/run"
    
    headers = {
        "Authorization": "Bearer 12776c804e23764323a141d7736af662e2e2d41a9deaf12e331188a32e1c299f",
        "Content-Type": "application/json"
    }
    
    # Test with BAJAJ document
    test_doc = "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/BAJHLIP23020V012223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D"
    
    # Focused test questions
    test_questions = [
        "What is the waiting period for pre-existing diseases?",
        "What is the grace period for premium payment?", 
        "Does this policy cover AYUSH treatments?",
        "What are the exclusions in this policy?",
        "What is the maximum sum insured available?"
    ]
    
    print("🎯 FOCUSED ACCURACY TEST")
    print("=" * 50)
    print(f"Document: BAJAJ Health Insurance Policy")
    print(f"Questions: {len(test_questions)}")
    
    payload = {
        "documents": test_doc,
        "questions": test_questions
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=40)
        end_time = time.time()
        
        processing_time = end_time - start_time
        print(f"\n⏱️  Response time: {processing_time:.2f} seconds")
        print(f"📊 Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            answers = data.get('answers', [])
            
            print(f"\n📝 DETAILED RESULTS:")
            print("=" * 50)
            
            # Analyze each answer in detail
            informative_count = 0
            
            for i, (question, answer) in enumerate(zip(test_questions, answers)):
                print(f"\n❓ QUESTION {i+1}: {question}")
                print(f"💬 ANSWER {i+1}: {answer}")
                
                # Detailed analysis
                answer_lower = answer.lower()
                is_informative = True
                
                if ("not available" in answer_lower or 
                    "error" in answer_lower or 
                    len(answer.strip()) < 30):
                    print(f"📊 ANALYSIS: ❌ Not informative")
                    is_informative = False
                else:
                    print(f"📊 ANALYSIS: ✅ Informative")
                    informative_count += 1
                
                # Check for specific content
                if "waiting period" in question.lower() and "waiting period" in answer_lower:
                    print(f"🎯 RELEVANCE: ✅ Contains relevant terms")
                elif "grace period" in question.lower() and "grace period" in answer_lower:
                    print(f"🎯 RELEVANCE: ✅ Contains relevant terms")
                elif "ayush" in question.lower() and "ayush" in answer_lower:
                    print(f"🎯 RELEVANCE: ✅ Contains relevant terms")
                elif "exclusion" in question.lower() and "exclusion" in answer_lower:
                    print(f"🎯 RELEVANCE: ✅ Contains relevant terms")
                elif "sum insured" in question.lower() and ("sum" in answer_lower or "insured" in answer_lower):
                    print(f"🎯 RELEVANCE: ✅ Contains relevant terms")
                elif is_informative:
                    print(f"🎯 RELEVANCE: ⚠️  Informative but may not be directly relevant")
                
                print("-" * 50)
            
            accuracy = (informative_count / len(test_questions)) * 100
            
            print(f"\n🏆 FINAL RESULTS:")
            print(f"✅ Informative answers: {informative_count}/{len(test_questions)}")
            print(f"🎯 Accuracy: {accuracy:.1f}%")
            print(f"⏱️  Response time: {processing_time:.2f}s")
            
            if accuracy >= 80:
                print("🎉 SUCCESS: Target accuracy ≥80% achieved!")
            elif accuracy >= 60:
                print("🔄 GOOD PROGRESS: Accuracy improving, needs fine-tuning")
            elif accuracy >= 40:
                print("📈 MODERATE PROGRESS: System working but needs improvement")
            else:
                print("⚠️  NEEDS MAJOR IMPROVEMENT: Low accuracy")
            
            return accuracy >= 80
            
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.Timeout:
        print("❌ Request timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_focused_accuracy()
    if success:
        print("\n🎉 SYSTEM READY FOR PRODUCTION!")
    else:
        print("\n🔧 SYSTEM NEEDS FURTHER OPTIMIZATION")
