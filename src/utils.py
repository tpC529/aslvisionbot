"""
Utility functions for ASL Vision Bot.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        logging.warning(f"Config file not found: {config_path}")
        return {}
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def setup_logging(config: Dict[str, Any]):
    """
    Setup logging based on configuration.
    
    Args:
        config: Configuration dictionary
    """
    log_config = config.get('logging', {})
    level = log_config.get('level', 'INFO')
    log_file = log_config.get('file', 'logs/aslvisionbot.log')
    console = log_config.get('console', True)
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Setup logging
    handlers = []
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    if console:
        handlers.append(logging.StreamHandler())
    
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def ensure_directories(config: Dict[str, Any]):
    """
    Ensure all required directories exist.
    
    Args:
        config: Configuration dictionary
    """
    directories = [
        'models',
        'datasets',
        'datasets/kaggle',
        'datasets/how2sign',
        'logs',
        'cache',
        'checkpoints',
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def get_model_path(model_name: str, config: Dict[str, Any]) -> str:
    """
    Get full path to a model file.
    
    Args:
        model_name: Name of the model
        config: Configuration dictionary
        
    Returns:
        Full path to model file
    """
    models_config = config.get('models', {})
    model_file = models_config.get(model_name, f"models/{model_name}.h5")
    
    return model_file


def get_dataset_path(dataset_name: str, config: Dict[str, Any]) -> str:
    """
    Get full path to a dataset directory.
    
    Args:
        dataset_name: Name of the dataset
        config: Configuration dictionary
        
    Returns:
        Full path to dataset directory
    """
    datasets_config = config.get('datasets', {})
    
    if dataset_name in datasets_config:
        return datasets_config[dataset_name].get('path', f"datasets/{dataset_name}/")
    
    return f"datasets/{dataset_name}/"


class PerformanceMonitor:
    """Monitor performance metrics"""
    
    def __init__(self):
        self.metrics = {
            'inference_times': [],
            'frame_rates': [],
            'accuracies': []
        }
    
    def add_inference_time(self, time_ms: float):
        """Add inference time measurement"""
        self.metrics['inference_times'].append(time_ms)
        
        # Keep only last 100 measurements
        if len(self.metrics['inference_times']) > 100:
            self.metrics['inference_times'] = self.metrics['inference_times'][-100:]
    
    def add_frame_rate(self, fps: float):
        """Add frame rate measurement"""
        self.metrics['frame_rates'].append(fps)
        
        if len(self.metrics['frame_rates']) > 100:
            self.metrics['frame_rates'] = self.metrics['frame_rates'][-100:]
    
    def get_average_inference_time(self) -> float:
        """Get average inference time"""
        if not self.metrics['inference_times']:
            return 0.0
        return sum(self.metrics['inference_times']) / len(self.metrics['inference_times'])
    
    def get_average_frame_rate(self) -> float:
        """Get average frame rate"""
        if not self.metrics['frame_rates']:
            return 0.0
        return sum(self.metrics['frame_rates']) / len(self.metrics['frame_rates'])
    
    def print_stats(self):
        """Print performance statistics"""
        print("\n=== Performance Statistics ===")
        print(f"Average Inference Time: {self.get_average_inference_time():.2f} ms")
        print(f"Average Frame Rate: {self.get_average_frame_rate():.2f} FPS")
        print("==============================\n")


if __name__ == "__main__":
    # Example usage
    config = load_config('config.yaml')
    setup_logging(config)
    ensure_directories(config)
    
    logger = logging.getLogger(__name__)
    logger.info("Utilities module initialized")
