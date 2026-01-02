"""
Script to train ASL sequence recognition models.
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, setup_logging, ensure_directories
from src.dataset_loader import DatasetLoader
from src.sequence_model import ASLSequenceModel, create_callbacks
from src.preprocessing import extract_hand_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_training_data(config: dict):
    """
    Prepare training data from datasets.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        X_train, y_train, X_val, y_val
    """
    logger.info("Loading datasets...")
    
    loader = DatasetLoader(config.get('datasets', {}))
    
    # Load alphabet dataset for basic training
    images, labels, classes = loader.load_alphabet_dataset(source='kaggle')
    
    if len(images) == 0:
        logger.warning("No training data loaded. Please download datasets first.")
        logger.info("Run: python scripts/download_datasets.py")
        return None, None, None, None
    
    logger.info(f"Loaded {len(images)} images")
    logger.info(f"Classes: {len(classes)}")
    
    # Preprocess
    images = loader.preprocess_images(images)
    
    # Split into train/val
    from sklearn.model_selection import train_test_split
    
    validation_split = config.get('training', {}).get('validation_split', 0.2)
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels,
        test_size=validation_split,
        random_state=42,
        stratify=labels
    )
    
    # Convert labels to one-hot
    from tensorflow.keras.utils import to_categorical
    y_train = to_categorical(y_train, num_classes=len(classes))
    y_val = to_categorical(y_val, num_classes=len(classes))
    
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    
    return X_train, y_train, X_val, y_val, classes


def train_sequence_model(config: dict):
    """
    Train the sequence recognition model.
    
    Args:
        config: Configuration dictionary
    """
    logger.info("Starting model training...")
    
    # Ensure directories exist
    ensure_directories(config)
    
    # Prepare data
    X_train, y_train, X_val, y_val, classes = prepare_training_data(config)
    
    if X_train is None:
        logger.error("Failed to prepare training data")
        return
    
    # Get model parameters from config
    model_config = config.get('models', {})
    training_config = config.get('training', {})
    
    sequence_length = model_config.get('sequence_length', 30)
    feature_dim = 63  # 21 landmarks * 3 coordinates
    
    # For initial training, we'll use a simplified approach
    # In production, you would extract hand landmarks and create sequences
    logger.info("Note: For full sequence training, you need video sequences with landmarks")
    logger.info("This script provides the training framework.")
    
    # Create model
    logger.info("Creating model...")
    model = ASLSequenceModel(
        input_shape=(sequence_length, feature_dim),
        num_classes=len(classes),
        lstm_units=model_config.get('lstm_units', 128),
        dropout_rate=model_config.get('dropout_rate', 0.3),
        use_attention=True
    )
    
    # Compile model
    model.compile_model(
        learning_rate=training_config.get('learning_rate', 0.001)
    )
    
    model.summary()
    
    # Note: Actual training would require sequence data
    logger.info("\n" + "="*60)
    logger.info("Model architecture created successfully!")
    logger.info("="*60)
    logger.info("\nTo train the model, you need:")
    logger.info("1. Video sequences with hand landmarks extracted")
    logger.info("2. Corresponding labels for each sequence")
    logger.info("3. Sufficient training data (thousands of samples)")
    logger.info("\nModel will be saved to: models/sequence_model.h5")
    logger.info("="*60 + "\n")
    
    # Save model architecture (without training)
    model_path = model_config.get('sequence_model', 'models/sequence_model.h5')
    try:
        model.save(model_path)
        logger.info(f"Model architecture saved to {model_path}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")


def main():
    """Main training function"""
    logger.info("ASL Sequence Model Training")
    logger.info("="*60 + "\n")
    
    # Load configuration
    config = load_config('config.yaml')
    
    if not config:
        logger.warning("Config file not found, using defaults")
        config = {}
    
    setup_logging(config)
    
    # Train model
    train_sequence_model(config)
    
    logger.info("\nTraining script completed!")
    logger.info("Next steps:")
    logger.info("1. Prepare video sequence data with landmarks")
    logger.info("2. Run training with actual sequence data")
    logger.info("3. Evaluate model performance")
    logger.info("4. Integrate trained model into chat.py")


if __name__ == "__main__":
    main()
