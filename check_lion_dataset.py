"""
Helper script to find and configure LION-400M dataset URLs
"""

import requests
import json
from pathlib import Path

def check_lion_dataset_availability():
    """Check various sources for LION-400M dataset"""
    
    print("=== Checking LION-400M Dataset Availability ===\n")
    
    # Check Hugging Face datasets
    hf_repos = [
        "BAAI/LION-400M",  # Possible repository names
        "LION/LION-400M",
        "datasets/LION-400M",
        "JourneyDB/JourneyDB",  # Alternative large-scale image dataset
        "laion/laion400m",  # Check if it's actually LAION
        "laion/laion-400m",
    ]
    
    print("1. Checking Hugging Face repositories:")
    for repo in hf_repos:
        try:
            url = f"https://huggingface.co/api/datasets/{repo}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print(f"   ✓ Found: {repo}")
                data = response.json()
                print(f"     Description: {data.get('description', 'N/A')[:100]}...")
            else:
                print(f"   ✗ Not found: {repo}")
        except Exception as e:
            print(f"   ✗ Error checking {repo}: {e}")
    
    print("\n2. Checking for LAION dataset (similar to LION):")
    laion_urls = [
        "https://laion.ai/blog/laion-400-open-dataset/",
        "https://huggingface.co/datasets/laion/laion400m-met",
        "https://huggingface.co/datasets/laion/laion400m-ava",
    ]
    
    for url in laion_urls:
        try:
            response = requests.head(url, timeout=10)
            if response.status_code == 200:
                print(f"   ✓ Accessible: {url}")
            else:
                print(f"   ✗ Not accessible: {url}")
        except Exception as e:
            print(f"   ✗ Error checking {url}: {e}")
    
    print("\n3. Alternative large-scale image datasets:")
    alternatives = [
        ("Common Crawl Images", "https://commoncrawl.org/"),
        ("OpenImages", "https://storage.googleapis.com/openimages/web/index.html"),
        ("YFCC100M", "https://multimediacommons.wordpress.com/yfcc100m-core-dataset/"),
    ]
    
    for name, url in alternatives:
        try:
            response = requests.head(url, timeout=10)
            if response.status_code == 200:
                print(f"   ✓ {name}: {url}")
            else:
                print(f"   ✗ {name}: {url}")
        except Exception as e:
            print(f"   ✗ {name}: Error checking")

def create_sample_url_list():
    """Create a sample URL list for testing"""
    
    sample_urls = [
        # High-quality sample images
        "https://picsum.photos/2048/2048?random=1",
        "https://picsum.photos/2048/2048?random=2", 
        "https://picsum.photos/2048/2048?random=3",
        "https://picsum.photos/2048/2048?random=4",
        "https://picsum.photos/2048/2048?random=5",
        # Add more sample URLs as needed
    ]
    
    url_file = Path("sample_image_urls.txt")
    with open(url_file, 'w') as f:
        for url in sample_urls:
            f.write(f"{url}\n")
    
    print(f"\nCreated sample URL list: {url_file}")
    print("You can use this for testing with:")
    print(f"python3 download_lion_dataset.py --method urls --url_list {url_file}")

def print_usage_instructions():
    """Print usage instructions"""
    
    print("\n" + "="*60)
    print("=== LION-400M Dataset Download Instructions ===")
    print("="*60)
    
    print("\n1. Basic usage (Hugging Face method):")
    print("   python3 download_lion_dataset.py")
    
    print("\n2. With custom parameters:")
    print("   python3 download_lion_dataset.py \\")
    print("       --save_dir /hdd1/data/LION-400M \\")
    print("       --max_size_tb 1.0 \\")
    print("       --method huggingface")
    
    print("\n3. Using URL list:")
    print("   python3 download_lion_dataset.py \\")
    print("       --method urls \\")
    print("       --url_list your_urls.txt")
    
    print("\n4. Sample download for testing:")
    print("   python3 download_lion_dataset.py --method sample")
    
    print("\n5. Using the bash script:")
    print("   ./run_lion_download.sh")
    
    print("\nNote: If LION-400M is not directly available, the script will")
    print("try to download from alternative large-scale image datasets.")

if __name__ == "__main__":
    check_lion_dataset_availability()
    create_sample_url_list()
    print_usage_instructions()
