#!/usr/bin/env python3
"""
Diagnostic test to understand accuracy issues
"""
import requests
import json
import time

def test_accuracy_diagnostic():
    """Diagnose accuracy issues with minimal API usage"""
    
    url = "https://llm-query-retrieval-system-production.up.railway.app/api/v1/hackrx/run"
    
    headers = {
        "Authorization": "Bearer 12776c804e23764323a141d7736af662e2e2d41a9deaf12e331188a32e1c299f",
        "Content-Type": "application/json"
    }
    
    # Test with HDFC judging document
    test_doc = "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/HDFHLIP23024V072223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D"
    
    # Simple questions that should have answers
    test_questions = [
        "What company issued this insurance policy?",
        "What type of insurance is this?"
    ]
    
    print("🔍 ACCURACY DIAGNOSTIC TEST")
    print("=" * 50)
    print("Testing simplified system with basic questions")
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
            
            print(f"\n📝 DIAGNOSTIC RESULTS:")
            print("=" * 50)
            
            for i, (question, answer) in enumerate(zip(test_questions, answers)):
                print(f"\n❓ Q{i+1}: {question}")
                print(f"💬 A{i+1}: {answer}")
                
                # Detailed analysis
                answer_lower = answer.lower()
                
                # Check for different types of responses
                if "quota" in answer_lower or "429" in answer_lower:
                    print(f"📊 ISSUE: ❌ API QUOTA EXHAUSTED")
                    print(f"🔧 SOLUTION: Wait for quota reset or upgrade API plan")
                elif "error" in answer_lower:
                    print(f"📊 ISSUE: ❌ SYSTEM ERROR")
                    print(f"🔧 SOLUTION: Check system logs and fix technical issues")
                elif "not available" in answer_lower:
                    print(f"📊 ISSUE: ⚠️  INFORMATION NOT FOUND")
                    print(f"🔧 SOLUTION: Improve document processing or chunk selection")
                elif len(answer.strip()) < 20:
                    print(f"📊 ISSUE: ❌ VERY SHORT RESPONSE")
                    print(f"🔧 SOLUTION: Improve prompt engineering")
                else:
                    print(f"📊 STATUS: ✅ WORKING RESPONSE")
                    
                    # Check for quality
                    quality_indicators = [
                        len(answer) > 30,
                        "hdfc" in answer_lower or "ergo" in answer_lower,
                        "insurance" in answer_lower or "policy" in answer_lower,
                        not any(vague in answer_lower for vague in ['maybe', 'possibly'])
                    ]
                    
                    quality_score = sum(quality_indicators)
                    print(f"🎯 QUALITY: {quality_score}/4")
                    
                    if quality_score >= 3:
                        print(f"✅ HIGH QUALITY RESPONSE")
                    elif quality_score >= 2:
                        print(f"⚠️  MODERATE QUALITY")
                    else:
                        print(f"❌ LOW QUALITY")
                
                print("-" * 50)
            
            # Overall assessment
            working_responses = sum(1 for a in answers if not any(issue in a.lower() for issue in ["quota", "error", "429"]))
            informative_responses = sum(1 for a in answers if len(a.strip()) > 30 and "not available" not in a.lower())
            
            print(f"\n🏆 DIAGNOSTIC SUMMARY:")
            print(f"📊 Working responses: {working_responses}/{len(answers)}")
            print(f"📊 Informative responses: {informative_responses}/{len(answers)}")
            print(f"⏱️  Response time: {processing_time:.2f}s")
            
            # Root cause analysis
            print(f"\n🔍 ROOT CAUSE ANALYSIS:")
            if working_responses == 0:
                print("❌ CRITICAL: No working responses")
                print("🔧 PRIMARY ISSUE: API quota exhausted or system errors")
                print("🔧 IMMEDIATE ACTION: Wait for quota reset or check system logs")
            elif informative_responses == 0:
                print("⚠️  ISSUE: System working but not finding information")
                print("🔧 PRIMARY ISSUE: Document processing or chunk selection")
                print("🔧 IMMEDIATE ACTION: Improve text extraction and chunking")
            elif informative_responses < len(answers):
                print("⚠️  PARTIAL: Some responses working")
                print("🔧 PRIMARY ISSUE: Inconsistent information retrieval")
                print("🔧 IMMEDIATE ACTION: Optimize chunk selection and prompts")
            else:
                print("✅ GOOD: System finding information consistently")
                print("🔧 FOCUS: Improve response quality and detail")
            
            return informative_responses > 0
            
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_accuracy_diagnostic()
    if success:
        print("\n✅ SYSTEM HAS POTENTIAL - FOCUS ON OPTIMIZATION")
    else:
        print("\n❌ CRITICAL ISSUES NEED IMMEDIATE ATTENTION")
