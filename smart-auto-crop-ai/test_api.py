"""
Quick test script to verify the Smart Auto Crop AI API is working
"""
import requests
import sys

def test_api():
    """Test the API health endpoint"""
    try:
        print("Testing Smart Auto Crop AI API...")
        print("-" * 50)
        
        # Test health endpoint
        response = requests.get("http://localhost:8000/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print("[OK] API is healthy!")
            print(f"   Status: {data.get('status')}")
            print(f"   Service: {data.get('service')}")
            print()
            print("[SUCCESS] Server is running successfully!")
            print("[INFO] Open http://localhost:8000 in your browser to use the app")
            return True
        else:
            print(f"[ERROR] API returned status code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("[ERROR] Could not connect to the server")
        print("   Make sure the server is running with: python main.py")
        return False
    except Exception as e:
        print(f"[ERROR] {e}")
        return False

if __name__ == "__main__":
    success = test_api()
    sys.exit(0 if success else 1)
