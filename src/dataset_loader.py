"""
Dataset loader utilities for ASL Vision Bot.

This module provides utilities for loading and preprocessing datasets from
various sources including Kaggle and How2Sign.
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


class DatasetConfig:
    """Configuration for dataset loading"""
    def __init__(self, config_dict: dict):
        self.kaggle_path = config_dict.get('kaggle', {}).get('path', 'datasets/kaggle/')
        self.how2sign_path = config_dict.get('how2sign', {}).get('path', 'datasets/how2sign/')
        self.target_size = config_dict.get('preprocessing', {}).get('target_size', [224, 224])
        self.normalize = config_dict.get('preprocessing', {}).get('normalize', True)
        self.frame_sampling_rate = config_dict.get('preprocessing', {}).get('frame_sampling_rate', 3)


class KaggleDatasetLoader:
    """Loader for Kaggle ASL datasets"""
    
    def __init__(self, dataset_path: str, target_size: Tuple[int, int] = (224, 224)):
        self.dataset_path = Path(dataset_path)
        self.target_size = target_size
        
    def load_asl_alphabet(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load ASL Alphabet dataset.
        
        Returns:
            images: Array of images (N, H, W, C)
            labels: Array of label indices (N,)
            classes: List of class names
        """
        alphabet_path = self.dataset_path / "asl_alphabet"
        
        if not alphabet_path.exists():
            logger.warning(f"ASL Alphabet dataset not found at {alphabet_path}")
            return np.array([]), np.array([]), []
        
        images = []
        labels = []
        classes = []
        
        # ASL alphabet includes A-Z and special signs
        for class_idx, class_dir in enumerate(sorted(alphabet_path.iterdir())):
            if not class_dir.is_dir():
                continue
                
            class_name = class_dir.name
            classes.append(class_name)
            
            for img_file in class_dir.glob("*.jpg"):
                img = cv2.imread(str(img_file))
                if img is not None:
                    img = cv2.resize(img, self.target_size)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                    labels.append(class_idx)
        
        return np.array(images), np.array(labels), classes
    
    def load_asl_mnist(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load ASL MNIST dataset.
        
        Returns:
            images: Array of images (N, H, W, C)
            labels: Array of label indices (N,)
            classes: List of class names (A-Z, excluding J and Z which require motion)
        """
        mnist_path = self.dataset_path / "asl_mnist"
        
        if not mnist_path.exists():
            logger.warning(f"ASL MNIST dataset not found at {mnist_path}")
            return np.array([]), np.array([]), []
        
        # ASL MNIST typically has CSV files
        train_csv = mnist_path / "sign_mnist_train.csv"
        
        if not train_csv.exists():
            logger.warning(f"Training CSV not found at {train_csv}")
            return np.array([]), np.array([]), []
        
        import pandas as pd
        
        df = pd.read_csv(train_csv)
        labels = df.iloc[:, 0].values
        pixels = df.iloc[:, 1:].values
        
        # Reshape to images (28x28 for MNIST)
        images = pixels.reshape(-1, 28, 28, 1)
        
        # Resize to target size
        resized_images = []
        for img in images:
            resized = cv2.resize(img, self.target_size)
            if len(resized.shape) == 2:
                resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
            resized_images.append(resized)
        
        # Classes A-Z (excluding J=9 and Z=25 which require motion)
        classes = [chr(i) for i in range(ord('A'), ord('Z')+1) if chr(i) not in ['J', 'Z']]
        
        return np.array(resized_images), labels, classes


class How2SignDatasetLoader:
    """Loader for How2Sign continuous ASL dataset"""
    
    def __init__(self, dataset_path: str, target_size: Tuple[int, int] = (224, 224), 
                 frame_sampling_rate: int = 3):
        self.dataset_path = Path(dataset_path)
        self.target_size = target_size
        self.frame_sampling_rate = frame_sampling_rate
        
    def load_video_clip(self, video_path: str, max_frames: int = 30) -> np.ndarray:
        """
        Load and preprocess a video clip.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to load
            
        Returns:
            frames: Array of frames (T, H, W, C)
        """
        cap = cv2.VideoCapture(video_path)
        
        frames = []
        frame_count = 0
        
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample every Nth frame
            if frame_count % self.frame_sampling_rate == 0:
                frame = cv2.resize(frame, self.target_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            
            frame_count += 1
        
        cap.release()
        
        # Pad if necessary
        while len(frames) < max_frames:
            frames.append(np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8))
        
        return np.array(frames)
    
    def load_annotations(self, annotation_file: str) -> List[Dict]:
        """
        Load annotations from JSON file.
        
        Args:
            annotation_file: Path to annotation JSON file
            
        Returns:
            annotations: List of annotation dictionaries
        """
        annotation_path = self.dataset_path / annotation_file
        
        if not annotation_path.exists():
            logger.warning(f"Annotation file not found at {annotation_path}")
            return []
        
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)
        
        return annotations
    
    def load_sentence_dataset(self, split: str = 'train', max_samples: int = None) -> Tuple[List[np.ndarray], List[str]]:
        """
        Load sentence-level dataset.
        
        Args:
            split: Dataset split ('train', 'val', 'test')
            max_samples: Maximum number of samples to load
            
        Returns:
            video_sequences: List of video frame sequences
            sentences: List of corresponding sentence annotations
        """
        annotations_file = f"{split}_annotations.json"
        annotations = self.load_annotations(annotations_file)
        
        video_sequences = []
        sentences = []
        
        for idx, annotation in enumerate(annotations):
            if max_samples and idx >= max_samples:
                break
            
            video_id = annotation.get('video_id')
            sentence = annotation.get('sentence', '')
            
            video_path = self.dataset_path / 'video_clips' / f"{video_id}.mp4"
            
            if video_path.exists():
                frames = self.load_video_clip(str(video_path))
                video_sequences.append(frames)
                sentences.append(sentence)
        
        return video_sequences, sentences


class DatasetLoader:
    """Main dataset loader that coordinates all dataset sources"""
    
    def __init__(self, config: Union[dict, DatasetConfig]):
        if isinstance(config, dict):
            self.config = DatasetConfig(config)
        else:
            self.config = config
        
        self.kaggle_loader = KaggleDatasetLoader(
            self.config.kaggle_path,
            tuple(self.config.target_size)
        )
        
        self.how2sign_loader = How2SignDatasetLoader(
            self.config.how2sign_path,
            tuple(self.config.target_size),
            self.config.frame_sampling_rate
        )
    
    def load_alphabet_dataset(self, source: str = 'kaggle') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load alphabet dataset from specified source.
        
        Args:
            source: Dataset source ('kaggle')
            
        Returns:
            images, labels, classes
        """
        if source == 'kaggle':
            return self.kaggle_loader.load_asl_alphabet()
        else:
            raise ValueError(f"Unknown source: {source}")
    
    def load_sentence_dataset(self, source: str = 'how2sign', split: str = 'train') -> Tuple[List[np.ndarray], List[str]]:
        """
        Load sentence-level dataset from specified source.
        
        Args:
            source: Dataset source ('how2sign')
            split: Dataset split ('train', 'val', 'test')
            
        Returns:
            video_sequences, sentences
        """
        if source == 'how2sign':
            return self.how2sign_loader.load_sentence_dataset(split)
        else:
            raise ValueError(f"Unknown source: {source}")
    
    def preprocess_images(self, images: np.ndarray) -> np.ndarray:
        """
        Preprocess images (normalization, etc.)
        
        Args:
            images: Array of images (N, H, W, C)
            
        Returns:
            Preprocessed images
        """
        if self.config.normalize:
            # Normalize to [0, 1]
            images = images.astype(np.float32) / 255.0
        
        return images


def get_dataset_info(dataset_name: str) -> Dict:
    """
    Get information about a dataset.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary with dataset information
    """
    dataset_info = {
        'asl_alphabet': {
            'name': 'ASL Alphabet',
            'source': 'Kaggle',
            'num_classes': 29,  # A-Z + space + delete + nothing
            'type': 'image',
            'url': 'https://www.kaggle.com/datasets/grassknoted/asl-alphabet'
        },
        'asl_mnist': {
            'name': 'ASL MNIST',
            'source': 'Kaggle',
            'num_classes': 24,  # A-Z excluding J and Z
            'type': 'image',
            'url': 'https://www.kaggle.com/datasets/datamunge/sign-language-mnist'
        },
        'how2sign': {
            'name': 'How2Sign',
            'source': 'Research Dataset',
            'num_classes': 'variable',  # Continuous sentences
            'type': 'video',
            'url': 'https://how2sign.github.io/'
        }
    }
    
    return dataset_info.get(dataset_name, {})


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    config = {
        'kaggle': {'path': 'datasets/kaggle/'},
        'how2sign': {'path': 'datasets/how2sign/'},
        'preprocessing': {
            'target_size': [224, 224],
            'normalize': True,
            'frame_sampling_rate': 3
        }
    }
    
    loader = DatasetLoader(config)
    
    # Print dataset information
    for dataset in ['asl_alphabet', 'asl_mnist', 'how2sign']:
        info = get_dataset_info(dataset)
        print(f"\n{info.get('name', dataset)}:")
        print(f"  Source: {info.get('source')}")
        print(f"  Type: {info.get('type')}")
        print(f"  Classes: {info.get('num_classes')}")
