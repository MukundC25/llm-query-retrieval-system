#!/usr/bin/env python3
"""
Test the improved system with sample policy documents
"""
import requests
import json
import time

def test_sample_documents():
    """Test with sample policy documents"""
    
    url = "https://llm-query-retrieval-system-production.up.railway.app/api/v1/hackrx/run"
    
    headers = {
        "Authorization": "Bearer 12776c804e23764323a141d7736af662e2e2d41a9deaf12e331188a32e1c299f",
        "Content-Type": "application/json"
    }
    
    # Sample policy documents
    sample_docs = [
        {
            "name": "BAJAJ Health Insurance",
            "url": "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/BAJHLIP23020V012223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D"
        },
        {
            "name": "HDFC Health Insurance", 
            "url": "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/HDFHLIP23024V072223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D"
        }
    ]
    
    # Test questions covering various policy aspects
    test_questions = [
        "What is the waiting period for pre-existing diseases?",
        "What is the grace period for premium payment?",
        "Does this policy cover maternity expenses?",
        "What is the maximum sum insured available?",
        "Are there any sub-limits on room rent?",
        "What is the waiting period for specific diseases like cataract?",
        "Does the policy cover AYUSH treatments?",
        "What are the exclusions in this policy?",
        "Is there coverage for organ donor expenses?",
        "What is the policy term and renewal age?"
    ]
    
    overall_results = []
    
    for doc in sample_docs:
        print(f"\n{'='*60}")
        print(f"🧪 Testing: {doc['name']}")
        print(f"{'='*60}")
        
        payload = {
            "documents": doc["url"],
            "questions": test_questions
        }
        
        start_time = time.time()
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=40)
            end_time = time.time()
            
            processing_time = end_time - start_time
            print(f"⏱️  Response time: {processing_time:.2f} seconds")
            print(f"📊 Status code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                answers = data.get('answers', [])
                
                # Analyze answer quality
                informative_answers = 0
                error_answers = 0
                
                for i, answer in enumerate(answers):
                    print(f"\n❓ Q{i+1}: {test_questions[i]}")
                    print(f"💬 A{i+1}: {answer}")
                    
                    if ("not available" in answer.lower() or 
                        "error" in answer.lower() or 
                        len(answer.strip()) < 20):
                        error_answers += 1
                    else:
                        informative_answers += 1
                
                accuracy = (informative_answers / len(test_questions)) * 100
                
                print(f"\n📊 Results for {doc['name']}:")
                print(f"✅ Informative answers: {informative_answers}/{len(test_questions)}")
                print(f"❌ Error/Not available: {error_answers}/{len(test_questions)}")
                print(f"🎯 Accuracy: {accuracy:.1f}%")
                print(f"⏱️  Performance: {processing_time:.2f}s")
                
                overall_results.append({
                    'document': doc['name'],
                    'accuracy': accuracy,
                    'response_time': processing_time,
                    'informative_answers': informative_answers,
                    'total_questions': len(test_questions)
                })
                
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"Response: {response.text}")
                
        except requests.Timeout:
            print("❌ Request timed out")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Overall summary
    if overall_results:
        print(f"\n{'='*60}")
        print("🏆 OVERALL SYSTEM PERFORMANCE")
        print(f"{'='*60}")
        
        avg_accuracy = sum(r['accuracy'] for r in overall_results) / len(overall_results)
        avg_response_time = sum(r['response_time'] for r in overall_results) / len(overall_results)
        
        print(f"📊 Average Accuracy: {avg_accuracy:.1f}%")
        print(f"⏱️  Average Response Time: {avg_response_time:.2f}s")
        
        if avg_accuracy >= 80:
            print("🎉 SUCCESS: Target accuracy ≥80% achieved!")
        else:
            print(f"⚠️  NEEDS IMPROVEMENT: Current accuracy {avg_accuracy:.1f}% < 80% target")
        
        if avg_response_time <= 30:
            print("✅ PERFORMANCE: Response time within limits")
        else:
            print("❌ PERFORMANCE: Response time exceeds 30s limit")

if __name__ == "__main__":
    test_sample_documents()
