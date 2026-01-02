"""
Script to download datasets from Kaggle and other sources.

Note: Requires Kaggle API credentials to be set up.
See: https://github.com/Kaggle/kaggle-api#api-credentials
"""

import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_kaggle_api():
    """Check if Kaggle API is available"""
    try:
        import kaggle
        return True
    except ImportError:
        logger.error("Kaggle API not installed. Install with: pip install kaggle")
        return False
    except OSError as e:
        logger.error(f"Kaggle API credentials not found: {e}")
        logger.info("Set up credentials at: ~/.kaggle/kaggle.json")
        return False


def download_kaggle_dataset(dataset_name: str, destination: str):
    """
    Download a Kaggle dataset.
    
    Args:
        dataset_name: Kaggle dataset identifier (e.g., 'grassknoted/asl-alphabet')
        destination: Destination directory
    """
    try:
        import kaggle
        
        logger.info(f"Downloading Kaggle dataset: {dataset_name}")
        logger.info(f"Destination: {destination}")
        
        os.makedirs(destination, exist_ok=True)
        
        kaggle.api.dataset_download_files(
            dataset_name,
            path=destination,
            unzip=True
        )
        
        logger.info(f"Successfully downloaded {dataset_name}")
        
    except Exception as e:
        logger.error(f"Error downloading {dataset_name}: {e}")
        raise


def download_asl_datasets():
    """Download ASL datasets from Kaggle"""
    
    if not check_kaggle_api():
        return False
    
    datasets = {
        'asl_alphabet': 'grassknoted/asl-alphabet',
        'asl_mnist': 'datamunge/sign-language-mnist',
        # Add more datasets as needed
    }
    
    for name, kaggle_id in datasets.items():
        destination = f'datasets/kaggle/{name}'
        
        if os.path.exists(destination) and os.listdir(destination):
            logger.info(f"{name} already exists, skipping")
            continue
        
        try:
            download_kaggle_dataset(kaggle_id, destination)
        except Exception as e:
            logger.error(f"Failed to download {name}: {e}")
            continue
    
    return True


def download_how2sign_info():
    """
    Provide information about downloading How2Sign dataset.
    
    How2Sign is a research dataset and requires manual download
    and agreement to terms.
    """
    logger.info("\n" + "="*60)
    logger.info("How2Sign Dataset Information")
    logger.info("="*60)
    logger.info("\nThe How2Sign dataset is a research dataset for continuous ASL.")
    logger.info("It requires manual download and agreement to terms of use.")
    logger.info("\nSteps to download:")
    logger.info("1. Visit: https://how2sign.github.io/")
    logger.info("2. Read and agree to the terms of use")
    logger.info("3. Fill out the data request form")
    logger.info("4. Download the dataset files")
    logger.info("5. Extract to: datasets/how2sign/")
    logger.info("\nDataset structure:")
    logger.info("  datasets/how2sign/")
    logger.info("    ├── video_clips/")
    logger.info("    ├── annotations/")
    logger.info("    └── README.md")
    logger.info("="*60 + "\n")


def main():
    """Main function"""
    logger.info("ASL Dataset Downloader")
    logger.info("="*60 + "\n")
    
    # Download Kaggle datasets
    logger.info("Downloading Kaggle datasets...")
    success = download_asl_datasets()
    
    if success:
        logger.info("\nKaggle datasets downloaded successfully!")
    else:
        logger.warning("\nFailed to download some Kaggle datasets.")
        logger.info("Make sure you have set up Kaggle API credentials.")
    
    # Provide How2Sign information
    download_how2sign_info()
    
    logger.info("\nDataset setup instructions:")
    logger.info("1. Kaggle datasets should now be in datasets/kaggle/")
    logger.info("2. Follow the How2Sign instructions above to download that dataset")
    logger.info("3. Update config.yaml to enable the datasets you want to use")
    logger.info("4. Run 'python scripts/train_model.py' to train models")


if __name__ == "__main__":
    main()
