#!/usr/bin/env python3
"""
Test document URL access
"""
import requests

def test_document_access():
    """Test document URL access"""
    
    # URL from the frontend with SAS token
    url_with_token = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    
    # URL without token
    url_without_token = "https://hackrx.blob.core.windows.net/assets/policy.pdf"
    
    print("Testing document URL access...")
    
    print("\n1. Testing URL without token:")
    try:
        response = requests.get(url_without_token, timeout=10)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print(f"✅ Success - Document size: {len(response.content)} bytes")
        else:
            print(f"❌ Failed: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n2. Testing URL with SAS token:")
    try:
        response = requests.get(url_with_token, timeout=10)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print(f"✅ Success - Document size: {len(response.content)} bytes")
            return url_with_token
        else:
            print(f"❌ Failed: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test with a public PDF
    print("\n3. Testing with a public PDF:")
    public_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    try:
        response = requests.get(public_url, timeout=10)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print(f"✅ Success - Document size: {len(response.content)} bytes")
            return public_url
        else:
            print(f"❌ Failed: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return None

if __name__ == "__main__":
    working_url = test_document_access()
    if working_url:
        print(f"\n✅ Working URL found: {working_url}")
    else:
        print("\n❌ No working URL found")
