"""
Download LION-400M dataset (images only, up to 1TB)
Downloads only image files from the LION-400M dataset to save storage space
"""

import os
import requests
import json
import tarfile
import zipfile
from pathlib import Path
import shutil
from tqdm import tqdm
import time
import hashlib
from urllib.parse import urlparse
import argparse

class LionDatasetDownloader:
    def __init__(self, save_dir="/hdd1/data/LION-400M", max_size_tb=1.0):
        """
        Initialize LION-400M dataset downloader
        Args:
            save_dir: Directory to save downloaded images
            max_size_tb: Maximum size to download in TB
        """
        self.save_dir = Path(save_dir)
        self.max_size_bytes = int(max_size_tb * 1024 * 1024 * 1024 * 1024)  # Convert TB to bytes
        self.current_size = 0
        self.downloaded_files = 0
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Image extensions to keep
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        
        # LION-400M dataset URLs (you may need to update these based on actual dataset structure)
        self.base_urls = [
            "https://huggingface.co/datasets/JourneyDB/JourneyDB/resolve/main/",
            # Add more URLs as needed for LION-400M dataset
        ]
        
        print(f"Initializing LION-400M downloader")
        print(f"Save directory: {self.save_dir}")
        print(f"Maximum download size: {max_size_tb} TB ({self.max_size_bytes:,} bytes)")
    
    def get_current_size(self):
        """Calculate current downloaded size"""
        total_size = 0
        for file_path in self.save_dir.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size
    
    def download_file(self, url, save_path, chunk_size=8192):
        """Download a single file with progress tracking"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            # Check if we have enough space
            if self.current_size + total_size > self.max_size_bytes:
                print(f"Skipping {url} - would exceed size limit")
                return False
            
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=save_path.name) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            self.current_size += total_size
            self.downloaded_files += 1
            return True
            
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            if save_path.exists():
                save_path.unlink()
            return False
    
    def extract_images_from_archive(self, archive_path):
        """Extract only image files from tar/zip archives"""
        extracted_images = 0
        
        try:
            if archive_path.suffix.lower() in ['.tar', '.tar.gz', '.tgz']:
                with tarfile.open(archive_path, 'r:*') as tar:
                    for member in tar.getmembers():
                        if member.isfile():
                            file_ext = Path(member.name).suffix.lower()
                            if file_ext in self.image_extensions:
                                # Extract to images directory
                                extract_path = self.save_dir / "images" / Path(member.name).name
                                
                                # Check size limit
                                if self.current_size + member.size > self.max_size_bytes:
                                    print(f"Size limit reached while extracting {member.name}")
                                    break
                                
                                extract_path.parent.mkdir(parents=True, exist_ok=True)
                                
                                with tar.extractfile(member) as source:
                                    with open(extract_path, 'wb') as target:
                                        shutil.copyfileobj(source, target)
                                
                                self.current_size += member.size
                                extracted_images += 1
                                
                                if extracted_images % 100 == 0:
                                    print(f"Extracted {extracted_images} images, "
                                          f"Size: {self.current_size / (1024**3):.2f} GB")
            
            elif archive_path.suffix.lower() == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_file:
                    for file_info in zip_file.filelist:
                        file_ext = Path(file_info.filename).suffix.lower()
                        if file_ext in self.image_extensions:
                            extract_path = self.save_dir / "images" / Path(file_info.filename).name
                            
                            # Check size limit
                            if self.current_size + file_info.file_size > self.max_size_bytes:
                                print(f"Size limit reached while extracting {file_info.filename}")
                                break
                            
                            extract_path.parent.mkdir(parents=True, exist_ok=True)
                            
                            with zip_file.open(file_info) as source:
                                with open(extract_path, 'wb') as target:
                                    shutil.copyfileobj(source, target)
                            
                            self.current_size += file_info.file_size
                            extracted_images += 1
                            
                            if extracted_images % 100 == 0:
                                print(f"Extracted {extracted_images} images, "
                                      f"Size: {self.current_size / (1024**3):.2f} GB")
        
        except Exception as e:
            print(f"Error extracting {archive_path}: {e}")
        
        return extracted_images
    
    def download_from_huggingface(self):
        """Download LION-400M/LAION dataset from Hugging Face (if available)"""
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
            
            # Try to access LAION or similar large-scale datasets
            repo_configs = [
                ("laion/laion400m", "LAION-400M dataset"),
                ("JourneyDB/JourneyDB", "JourneyDB dataset"),
                ("BAAI/LION-400M", "LION-400M dataset"),
            ]
            
            for repo_id, description in repo_configs:
                try:
                    print(f"Checking repository: {repo_id} ({description})")
                    files = list_repo_files(repo_id)
                    print(f"Found {len(files)} files in repository")
                    
                    # Filter for parquet files (LAION format) or image archives
                    data_files = [f for f in files if f.endswith(('.parquet', '.tar', '.tar.gz', '.zip', '.tgz'))]
                    print(f"Found {len(data_files)} data files")
                    
                    # Limit to first few files to respect size limit
                    max_files = min(10, len(data_files))  # Process up to 10 files
                    
                    for i, file_name in enumerate(data_files[:max_files]):
                        if self.current_size >= self.max_size_bytes:
                            print("Size limit reached!")
                            return
                        
                        print(f"Downloading {file_name} ({i+1}/{max_files}) from {repo_id}")
                        
                        try:
                            local_path = hf_hub_download(
                                repo_id=repo_id,
                                filename=file_name,
                                local_dir=self.save_dir / "temp",
                                local_dir_use_symlinks=False
                            )
                            
                            if file_name.endswith('.parquet'):
                                # Handle LAION parquet files
                                extracted = self.process_laion_parquet(Path(local_path))
                                print(f"Processed {extracted} images from {file_name}")
                            else:
                                # Handle archive files
                                extracted = self.extract_images_from_archive(Path(local_path))
                                print(f"Extracted {extracted} images from {file_name}")
                            
                            # Remove the file after processing to save space
                            Path(local_path).unlink()
                            
                        except Exception as e:
                            print(f"Error downloading {file_name}: {e}")
                            continue
                    
                    if data_files:  # Successfully found data files
                        break
                    
                except Exception as e:
                    print(f"Error accessing repository {repo_id}: {e}")
                    continue
        
        except ImportError:
            print("huggingface_hub not installed. Installing...")
            os.system("pip install huggingface_hub pandas")
            self.download_from_huggingface()
        except Exception as e:
            print(f"Error downloading from Hugging Face: {e}")
    
    def process_laion_parquet(self, parquet_path):
        """Process LAION parquet file and download images"""
        try:
            import pandas as pd
            
            df = pd.read_parquet(parquet_path)
            print(f"Loaded parquet with {len(df)} entries")
            
            downloaded_count = 0
            
            # Process each row in the parquet file
            for idx, row in df.iterrows():
                if self.current_size >= self.max_size_bytes:
                    break
                
                # Get image URL
                url = row.get('URL', row.get('url', ''))
                if not url or not url.startswith('http'):
                    continue
                
                # Generate filename
                filename = f"laion_{idx:08d}.jpg"
                save_path = self.save_dir / "images" / filename
                
                # Download image
                if self.download_file(url, save_path):
                    downloaded_count += 1
                    
                    if downloaded_count % 50 == 0:
                        print(f"Downloaded {downloaded_count} images from LAION")
                
                # Stop if we have enough images
                if downloaded_count >= 1000:  # Limit per parquet file
                    break
            
            return downloaded_count
            
        except ImportError:
            print("pandas not available for parquet processing")
            return 0
        except Exception as e:
            print(f"Error processing LAION parquet: {e}")
            return 0
    
    def download_from_urls(self, url_list_file=None):
        """Download images from a list of URLs"""
        if url_list_file and os.path.exists(url_list_file):
            with open(url_list_file, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]
        else:
            # Sample URLs for testing (replace with actual LION-400M URLs)
            urls = [
                # Add actual LION-400M dataset URLs here
            ]
        
        print(f"Found {len(urls)} URLs to download")
        
        for i, url in enumerate(urls):
            if self.current_size >= self.max_size_bytes:
                print("Size limit reached!")
                break
            
            # Generate filename from URL
            parsed_url = urlparse(url)
            filename = Path(parsed_url.path).name
            if not filename:
                filename = f"image_{i:06d}.jpg"
            
            save_path = self.save_dir / "images" / filename
            
            print(f"Downloading {i+1}/{len(urls)}: {filename}")
            success = self.download_file(url, save_path)
            
            if success:
                print(f"Progress: {self.current_size / (1024**3):.2f} GB / {self.max_size_bytes / (1024**3):.0f} GB")
            
            # Small delay to be respectful
            time.sleep(0.1)
    
    def download_sample_images(self):
        """Download sample images for testing (when actual LION-400M is not available)"""
        print("Downloading sample images for testing...")
        
        # Sample large image datasets
        sample_urls = [
            "https://picsum.photos/2048/2048?random=1",
            "https://picsum.photos/2048/2048?random=2",
            "https://picsum.photos/2048/2048?random=3",
            # Add more sample URLs if needed
        ]
        
        for i, url in enumerate(sample_urls):
            if self.current_size >= self.max_size_bytes:
                break
            
            save_path = self.save_dir / "images" / f"sample_{i:04d}.jpg"
            print(f"Downloading sample image {i+1}")
            self.download_file(url, save_path)
            time.sleep(1)  # Be respectful to the service
    
    def run_download(self, method="huggingface", url_list_file=None):
        """Run the download process"""
        print("Starting LION-400M dataset download...")
        print(f"Target size: {self.max_size_bytes / (1024**3):.1f} GB")
        
        # Update current size
        self.current_size = self.get_current_size()
        print(f"Current size: {self.current_size / (1024**3):.2f} GB")
        
        if method == "huggingface":
            self.download_from_huggingface()
        elif method == "urls":
            self.download_from_urls(url_list_file)
        elif method == "sample":
            self.download_sample_images()
        else:
            print(f"Unknown method: {method}")
            return
        
        # Final summary
        final_size = self.get_current_size()
        print(f"\n=== Download Summary ===")
        print(f"Total downloaded: {final_size / (1024**3):.2f} GB")
        print(f"Number of files: {self.downloaded_files}")
        print(f"Save directory: {self.save_dir}")
        
        # Count images
        image_count = len(list(self.save_dir.rglob("*.jpg"))) + \
                     len(list(self.save_dir.rglob("*.png"))) + \
                     len(list(self.save_dir.rglob("*.jpeg")))
        print(f"Total images: {image_count}")


def main():
    parser = argparse.ArgumentParser(description='Download LION-400M dataset (images only)')
    parser.add_argument('--save_dir', type=str, default="/hdd1/data/LION-400M",
                        help='Directory to save downloaded images')
    parser.add_argument('--max_size_tb', type=float, default=1.0,
                        help='Maximum size to download in TB')
    parser.add_argument('--method', type=str, default="huggingface",
                        choices=["huggingface", "urls", "sample"],
                        help='Download method')
    parser.add_argument('--url_list', type=str, default=None,
                        help='File containing list of URLs to download')
    
    args = parser.parse_args()
    
    # Check if save directory is accessible
    save_path = Path(args.save_dir)
    try:
        save_path.mkdir(parents=True, exist_ok=True)
        print(f"Save directory ready: {save_path}")
    except Exception as e:
        print(f"Error creating save directory {save_path}: {e}")
        return
    
    # Create downloader and start download
    downloader = LionDatasetDownloader(args.save_dir, args.max_size_tb)
    downloader.run_download(args.method, args.url_list)


if __name__ == "__main__":
    main()
