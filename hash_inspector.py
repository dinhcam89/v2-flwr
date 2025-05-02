# direct_inspect.py
import sys
import requests
import json

def main():
    if len(sys.argv) < 2:
        print("Usage: python direct_inspect.py <ipfs_hash>")
        return
    
    ipfs_hash = sys.argv[1]
    api_url = "http://127.0.0.1:5001/api/v0"
    
    # Get content
    response = requests.post(
        f"{api_url}/cat",
        params={"arg": ipfs_hash}
    )
    
    if response.status_code != 200:
        print(f"Failed to retrieve content: {response.text}")
        return
    
    content = response.content
    
    # Try as JSON
    try:
        data = json.loads(content)
        print(json.dumps(data, indent=2))
        return
    except:
        pass
    
    # Try as text
    try:
        text = content.decode('utf-8')
        print(text)
        return
    except:
        pass
    
    # As hex
    print(content.hex()[:200] + "...")
    
if __name__ == "__main__":
    main()