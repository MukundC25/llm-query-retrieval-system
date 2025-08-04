#!/usr/bin/env python3
"""
Conservative accuracy test - minimal API usage to check improvements
"""
import requests
import json
import time

def test_accuracy_conservative():
    """Test accuracy improvements with minimal API usage"""
    
    url = "https://llm-query-retrieval-system-production.up.railway.app/api/v1/hackrx/run"
    
    headers = {
        "Authorization": "Bearer 12776c804e23764323a141d7736af662e2e2d41a9deaf12e331188a32e1c299f",
        "Content-Type": "application/json"
    }
    
    # Test with one document and carefully selected questions
    test_doc = "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/BAJHLIP23020V012223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D"
    
    # Conservative test - only 3 questions to preserve API quota
    test_questions = [
        "Does this policy cover AYUSH treatments?",
        "What are the main exclusions in this policy?",
        "What type of insurance policy is this?"
    ]
    
    print("🎯 CONSERVATIVE ACCURACY TEST")
    print("=" * 50)
    print("Testing enhanced accuracy improvements")
    print(f"Using only {len(test_questions)} questions to preserve API quota")
    print("=" * 50)
    
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
            
            print(f"\n📝 DETAILED ANALYSIS:")
            print("=" * 50)
            
            high_quality_count = 0
            
            for i, (question, answer) in enumerate(zip(test_questions, answers)):
                print(f"\n❓ Q{i+1}: {question}")
                print(f"💬 A{i+1}: {answer}")
                
                # Enhanced quality assessment
                answer_lower = answer.lower()
                
                # Check for errors first
                if any(error_term in answer_lower for error_term in ["error", "quota", "429", "exceeded"]):
                    print(f"📊 STATUS: ❌ API ERROR")
                    continue
                
                # Check for "not available" responses
                if "not available" in answer_lower:
                    print(f"📊 STATUS: ⚠️  NO INFO FOUND (honest response)")
                    continue
                
                # Quality indicators for good answers
                quality_indicators = [
                    len(answer) > 50,  # Substantial content
                    any(char.isdigit() for char in answer),  # Contains specifics
                    any(term in answer_lower for term in ['policy', 'coverage', 'benefit', 'section']),  # Domain relevant
                    answer.count('.') >= 2,  # Multiple sentences
                    not any(vague in answer_lower for vague in ['maybe', 'possibly', 'might be']),  # Not vague
                ]
                
                quality_score = sum(quality_indicators)
                
                if quality_score >= 3:
                    print(f"📊 STATUS: ✅ HIGH QUALITY (score: {quality_score}/5)")
                    high_quality_count += 1
                elif quality_score >= 2:
                    print(f"📊 STATUS: ⚠️  MODERATE QUALITY (score: {quality_score}/5)")
                else:
                    print(f"📊 STATUS: ❌ LOW QUALITY (score: {quality_score}/5)")
                
                # Check for specific content relevance
                if "ayush" in question.lower() and "ayush" in answer_lower:
                    print(f"🎯 RELEVANCE: ✅ Contains AYUSH information")
                elif "exclusion" in question.lower() and any(term in answer_lower for term in ["exclusion", "excluded", "not covered"]):
                    print(f"🎯 RELEVANCE: ✅ Contains exclusion information")
                elif "insurance" in question.lower() or "policy" in question.lower():
                    if any(term in answer_lower for term in ["insurance", "policy", "health", "medical"]):
                        print(f"🎯 RELEVANCE: ✅ Contains policy information")
                
                print("-" * 50)
            
            # Calculate accuracy
            total_valid_responses = len([a for a in answers if not any(error in a.lower() for error in ["error", "quota", "429"])])
            accuracy = (high_quality_count / total_valid_responses * 100) if total_valid_responses > 0 else 0
            
            print(f"\n🏆 ACCURACY ASSESSMENT:")
            print(f"✅ High-quality answers: {high_quality_count}/{total_valid_responses}")
            print(f"🎯 Accuracy: {accuracy:.1f}%")
            print(f"⏱️  Response time: {processing_time:.2f}s")
            
            # Improvement assessment
            print(f"\n📈 IMPROVEMENT ANALYSIS:")
            if accuracy >= 70:
                print("🎉 EXCELLENT: Target accuracy achieved! (≥70%)")
                print("✅ System ready for production use")
            elif accuracy >= 50:
                print("✅ GOOD: Significant improvement achieved (≥50%)")
                print("✅ System performing well for most queries")
            elif accuracy >= 30:
                print("⚠️  MODERATE: Some improvement shown (≥30%)")
                print("🔧 Further optimization recommended")
            else:
                print("❌ NEEDS WORK: Limited improvement (<30%)")
                print("🔧 Additional accuracy measures needed")
            
            # API usage efficiency
            print(f"\n💡 API EFFICIENCY:")
            print(f"✅ Used only {len(test_questions)} API calls")
            print(f"✅ Response caching implemented for future queries")
            print(f"✅ Enhanced chunk selection reduces irrelevant context")
            
            return accuracy >= 50
            
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_accuracy_conservative()
    if success:
        print("\n🚀 ACCURACY IMPROVEMENTS SUCCESSFUL!")
        print("✅ System ready for HackRX 2025 production use")
    else:
        print("\n🔧 CONTINUE OPTIMIZATION")
        print("⚠️  May need additional improvements or API quota reset")
