"""
Image crawling script for TissueMNIST classes
Crawls images for each class and validates with a trained model
"""

import os
import requests
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import time
import random
from io import BytesIO
import argparse
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, ElementClickInterceptedException

# MedMNIST-related imports (assuming they are installed)
# import medmnist
# from medmnist import INFO
# from torchvision.models import resnet18

# Mock imports for standalone execution if medmnist is not available
class MockInfo:
    def __getitem__(self, key):
        return {'label': {str(i): f'Class {i}' for i in range(8)}}

INFO = {'tissuemnist': MockInfo()['tissuemnist']}
from torchvision.models import resnet18


class ImageCrawler:
    def __init__(self, model_path, save_dir="/home/suyoung/Vscode/HAST/data/downloaded_images_5000"):
        """
        Initialize the image crawler
        Args:
            model_path: Path to the trained model for validation
            save_dir: Directory to save crawled images
        """
        self.save_dir = save_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Class information for TissueMNIST
        self.class_info = {
            0: "Collecting Duct, Connecting Tubule histology",
            1: "Distal Convoluted Tubule histology",
            2: "Glomerular endothelial cells histology",
            3: "Interstitial endothelial cells histology",
            4: "Leukocytes histology",
            5: "Podocytes histology",
            6: "Proximal Tubule Segments histology",
            7: "Thick Ascending Limb histology"
        }
        
        # Target counts per class
        self.target_counts = [625] * 8
        self.current_counts = [0] * 8
        
        # Create save directories
        os.makedirs(self.save_dir, exist_ok=True)
        for class_idx in range(8):
            class_dir = os.path.join(self.save_dir, f"class_{class_idx}")
            os.makedirs(class_dir, exist_ok=True)
        
        # Load the validation model
        self.model = self._load_model(model_path)
        
        # Image preprocessing
        self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
                ])

        # Setup selenium driver
        self.driver = self._setup_driver()
    
    def _setup_driver(self):
        """Setup Chrome driver for web scraping"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

        try:
            driver = webdriver.Chrome(options=chrome_options)
            return driver
        except Exception as e:
            print(f"Error setting up Chrome driver: {e}")
            print("Please install ChromeDriver and make sure it's in your PATH.")
            return None
    
    def _load_model(self, model_path):
        """Load the trained model for validation"""
        info = INFO['tissuemnist']
        n_classes = len(info['label'])
        
        model = resnet18(weights=None, num_classes=n_classes)
        # The original code used `pretrained=False` which is deprecated.
        # Use `weights=None` for an untrained model.
        
        if model_path and os.path.exists(model_path):
            try:
                # Assuming the model was saved with a 'net' key
                model.load_state_dict(torch.load(model_path, map_location=self.device)['net'], strict=True)
                print(f"Loaded model from {model_path}")
            except KeyError:
                 # If 'net' key doesn't exist, try loading the state dict directly
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Loaded model directly from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}. Using untrained model for validation.")
        else:
            print("No model path provided or file not found. Using untrained model.")
        
        model.to(self.device)
        model.eval()
        return model
    
    def _predict_class(self, image):
        """Predict the class of an image using the loaded model"""
        try:
            # Preprocess image
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply transforms
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                output = self.model(tensor)
                _, predicted = torch.max(output.data, 1)
                return predicted.item()
        except Exception as e:
            print(f"Error in prediction: {e}")
            return -1

    def _search_images_google(self, query, max_images=100):
        """
        Search for images using Google Images with a more robust method.
        This method clicks on thumbnails to get higher-quality image URLs.
        """
        if not self.driver:
            return []

        search_url = f"https://www.google.com/search?q={query}&tbm=isch&tbs=isz:l"
        self.driver.get(search_url)
        
        image_urls = set()
        
        # Scroll down to load images and find the 'Show more results' button
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        for _ in range(10): # Scroll a few times to load initial images
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                try:
                    # Try to click the "Show more results" button if it exists
                    more_results_button = self.driver.find_element(By.CSS_SELECTOR, "input.mye4qd")
                    if more_results_button:
                        more_results_button.click()
                        time.sleep(2)
                except:
                    # If button not found or other error, break the scroll loop
                    break
            last_height = new_height

        # Find all thumbnail elements
        # This selector targets the container for each image result. It might need updating if Google changes its layout.
        thumbnails = self.driver.find_elements(By.CSS_SELECTOR, "div.H8Rx8c")
        
        print(f"Found {len(thumbnails)} potential image thumbnails.")

        for thumb in thumbnails[:max_images]:
            if len(image_urls) >= max_images:
                break
            try:
                # Click the thumbnail to open the preview pane
                self.driver.execute_script("arguments[0].click();", thumb)
                time.sleep(1) # Wait a moment for the preview to start loading

                # Wait for the high-resolution image to be loaded in the preview pane
                # This selector targets the main image in the preview pane. It's the most likely to change.
                wait = WebDriverWait(self.driver, 10)
                large_image = wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "img.sFlh5c"))
                )
                
                img_url = large_image.get_attribute('src')

                # Add valid URLs to the set (avoids duplicates)
                if img_url and img_url.startswith(('http', 'https')):
                    image_urls.add(img_url)

            except (TimeoutException, ElementClickInterceptedException) as e:
                # print(f"Could not get URL for one thumbnail: {e}")
                continue
            except Exception as e:
                # print(f"An unexpected error occurred: {e}")
                continue

        return list(image_urls)
    
    def _download_image(self, url, timeout=10):
        """Download an image from URL"""
        try:
            # Some URLs might be base64 encoded, skip them
            if url.startswith('data:image'):
                return None

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=timeout, stream=True)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                return None
            
            image = Image.open(BytesIO(response.content))
            
            if image.size[0] < 224 or image.size[1] < 224:
                return None # Skip small images
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
            
        except Exception as e:
            # print(f"Error downloading image from {url}: {e}")
            return None
    
    def _save_image(self, image, class_idx, image_idx):
        """Save image to the appropriate class directory"""
        class_dir = os.path.join(self.save_dir, f"class_{class_idx}")
        filename = f"image_{image_idx:04d}.jpg"
        filepath = os.path.join(class_dir, filename)
        
        try:
            image.save(filepath, 'JPEG', quality=90)
            return True
        except Exception as e:
            print(f"Error saving image: {e}")
            return False
    
    def crawl_class(self, class_idx, max_attempts=10):
        """Crawl images for a specific class"""
        query = self.class_info[class_idx]
        target_count = self.target_counts[class_idx]
        
        print(f"\n=== Crawling Class {class_idx}: {query} ===")
        print(f"Target: {target_count} images | Current: {self.current_counts[class_idx]}")
        
        attempts = 0
        while self.current_counts[class_idx] < target_count and attempts < max_attempts:
            attempts += 1
            print(f"Search attempt {attempts}/{max_attempts}...")
            
            image_urls = self._search_images_google(query, max_images=target_count * 2) # Fetch more images than needed
            
            if not image_urls:
                print("No image URLs found in this attempt.")
                time.sleep(5)
                continue
            
            print(f"Found {len(image_urls)} unique image URLs. Starting download and validation...")
            
            for i, url in enumerate(image_urls):
                if self.current_counts[class_idx] >= target_count:
                    break
                
                # print(f"Processing URL {i+1}/{len(image_urls)}...")
                image = self._download_image(url)
                
                if image is None:
                    continue
                
                predicted_class = self._predict_class(image)
                
                if predicted_class == class_idx:
                    if self._save_image(image, class_idx, self.current_counts[class_idx]):
                        self.current_counts[class_idx] += 1
                        print(f"✓ Saved image {self.current_counts[class_idx]}/{target_count} for class {class_idx}")
                else:
                    # Optional: uncomment to see why images are rejected
                    # print(f"✗ Rejected. Predicted: {predicted_class}, Expected: {class_idx}")
                    pass
                
                time.sleep(random.uniform(0.5, 1.5)) # Shorter delay between downloads
            
            if self.current_counts[class_idx] < target_count:
                print(f"Attempt finished. Need {target_count - self.current_counts[class_idx]} more images for class {class_idx}.")
                time.sleep(5) # Delay before next search attempt
        
        print(f"--- Completed class {class_idx}: {self.current_counts[class_idx]}/{target_count} images gathered ---")

    def crawl_all_classes(self):
        """Crawl images for all classes"""
        print("Starting image crawling for TissueMNIST classes...")
        print(f"Target: {sum(self.target_counts)} total images")
        
        for class_idx in range(8):
            try:
                self.crawl_class(class_idx)
            except KeyboardInterrupt:
                print("\nCrawling interrupted by user.")
                break
            except Exception as e:
                print(f"An unexpected error occurred while crawling class {class_idx}: {e}")
                continue
        
        print("\n" + "="*25)
        print("=== Crawling Summary ===")
        print("="*25)
        total_downloaded = sum(self.current_counts)
        print(f"Total images downloaded: {total_downloaded}/{sum(self.target_counts)}")
        for i, count in enumerate(self.current_counts):
            print(f"Class {i} ({self.class_info[i].split(' ')[0]}): {count}/{self.target_counts[i]} images")
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'driver') and self.driver:
            self.driver.quit()

def main():
    parser = argparse.ArgumentParser(description='Crawl images for TissueMNIST classes')
    # Make sure the default paths are correct for your system
    parser.add_argument('--model_path', type=str, default="/home/suyoung/Vscode/HAST/models/checkpoints/weights_tissuemnist/resnet18_224_1.pth",
                        help='Path to trained model for validation')
    parser.add_argument('--save_dir', type=str, 
                        default='./downloaded_images_balanced',
                        help='Directory to save downloaded images')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Warning: Model path does not exist: {args.model_path}")
        print("The script will use an UNTRAINED model for validation, which might not be effective.")

    crawler = ImageCrawler(args.model_path, args.save_dir)
    crawler.crawl_all_classes()

if __name__ == "__main__":
    main()