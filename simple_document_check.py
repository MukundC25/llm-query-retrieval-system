#!/usr/bin/env python3
"""
Simple check to see what's in the document
"""
import requests
from pdfminer.high_level import extract_text
import io

def check_document_content():
    """Check what content is actually in the document"""
    
    # Test document
    test_url = "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/BAJHLIP23020V012223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D"
    
    print("🔍 Checking Document Content...")
    print("=" * 50)
    
    try:
        # Download document
        print("1. Downloading document...")
        response = requests.get(test_url, timeout=30)
        response.raise_for_status()
        content = response.content
        print(f"✅ Downloaded {len(content)} bytes")
        
        # Extract text
        print("\n2. Extracting text...")
        text = extract_text(io.BytesIO(content))
        print(f"✅ Extracted {len(text)} characters")
        
        # Search for key terms
        print("\n3. Searching for key information...")
        key_searches = [
            ("waiting period", "waiting period information"),
            ("pre-existing", "pre-existing disease information"),
            ("grace period", "grace period information"),
            ("premium", "premium information"),
            ("maternity", "maternity coverage"),
            ("sum insured", "sum insured information"),
            ("room rent", "room rent limits"),
            ("cataract", "cataract coverage"),
            ("AYUSH", "AYUSH treatment coverage"),
            ("exclusion", "exclusions"),
            ("organ donor", "organ donor coverage"),
            ("policy term", "policy term information")
        ]
        
        found_info = {}
        for search_term, description in key_searches:
            # Find all occurrences
            text_lower = text.lower()
            term_lower = search_term.lower()
            
            if term_lower in text_lower:
                # Find context around the term
                start_pos = text_lower.find(term_lower)
                context_start = max(0, start_pos - 200)
                context_end = min(len(text), start_pos + 300)
                context = text[context_start:context_end].strip()
                
                found_info[search_term] = context
                print(f"\n✅ Found '{search_term}':")
                print(f"   Context: ...{context}...")
            else:
                print(f"\n❌ Missing '{search_term}'")
        
        # Show document structure
        print(f"\n4. Document structure analysis...")
        lines = text.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        print(f"   Total lines: {len(lines)}")
        print(f"   Non-empty lines: {len(non_empty_lines)}")
        
        # Show first few lines
        print(f"\n5. First 10 non-empty lines:")
        for i, line in enumerate(non_empty_lines[:10]):
            print(f"   {i+1}: {line[:100]}...")
        
        # Check if this looks like a policy document
        policy_indicators = ["policy", "insurance", "coverage", "premium", "claim", "benefit"]
        found_indicators = [term for term in policy_indicators if term.lower() in text.lower()]
        print(f"\n6. Policy document indicators found: {found_indicators}")
        
        if len(found_indicators) >= 3:
            print("✅ This appears to be a valid policy document")
        else:
            print("⚠️  This might not be a complete policy document")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    check_document_content()
