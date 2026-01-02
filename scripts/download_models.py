"""
Script to download pre-trained models.
"""

import os
import sys
import urllib.request
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Model URLs (these would be actual URLs in production)
MODEL_URLS = {
    'hand_landmarker': 'https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task',
    # Add more model URLs here when available
}


def download_file(url: str, destination: str):
    """Download a file from URL to destination"""
    try:
        logger.info(f"Downloading {url}")
        logger.info(f"Destination: {destination}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        # Download with progress
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, (downloaded / total_size) * 100)
            sys.stdout.write(f"\rProgress: {percent:.1f}%")
            sys.stdout.flush()
        
        urllib.request.urlretrieve(url, destination, reporthook=report_progress)
        print()  # New line after progress
        logger.info(f"Successfully downloaded to {destination}")
        
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        raise


def download_models():
    """Download all required models"""
    logger.info("Starting model download...")
    
    # Download hand landmarker if not exists
    hand_landmarker_path = 'hand_landmarker.task'
    if not os.path.exists(hand_landmarker_path):
        download_file(MODEL_URLS['hand_landmarker'], hand_landmarker_path)
    else:
        logger.info(f"{hand_landmarker_path} already exists, skipping")
    
    logger.info("\nModel download complete!")
    logger.info("\nNote: Advanced sequence models need to be trained separately.")
    logger.info("Run 'python scripts/train_model.py' after downloading datasets.")


def main():
    """Main function"""
    try:
        download_models()
    except Exception as e:
        logger.error(f"Failed to download models: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
