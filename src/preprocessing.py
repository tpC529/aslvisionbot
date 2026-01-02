"""
Preprocessing utilities for ASL datasets.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class VideoPreprocessor:
    """Preprocess video sequences for model input"""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        sequence_length: int = 30,
        normalize: bool = True
    ):
        self.target_size = target_size
        self.sequence_length = sequence_length
        self.normalize = normalize
    
    def preprocess_video(self, frames: np.ndarray) -> np.ndarray:
        """
        Preprocess video frames.
        
        Args:
            frames: Array of frames (T, H, W, C)
            
        Returns:
            Preprocessed frames
        """
        processed = []
        
        for frame in frames:
            # Resize
            if frame.shape[:2] != self.target_size:
                frame = cv2.resize(frame, self.target_size)
            
            # Normalize
            if self.normalize:
                frame = frame.astype(np.float32) / 255.0
            
            processed.append(frame)
        
        # Pad or truncate to sequence_length
        processed = self._adjust_sequence_length(processed)
        
        return np.array(processed)
    
    def _adjust_sequence_length(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Adjust sequence to target length"""
        # Handle empty frames list
        if not frames:
            # Create sequence of zero frames
            zero_frame = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.float32 if self.normalize else np.uint8)
            return [zero_frame] * self.sequence_length
        
        if len(frames) < self.sequence_length:
            # Pad with last frame
            last_frame = frames[-1]
            while len(frames) < self.sequence_length:
                frames.append(last_frame.copy())
        elif len(frames) > self.sequence_length:
            # Sample uniformly
            indices = np.linspace(0, len(frames) - 1, self.sequence_length, dtype=int)
            frames = [frames[i] for i in indices]
        
        return frames
    
    def extract_hand_region(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Extract hand region from frame based on landmarks.
        
        Args:
            frame: Input frame
            landmarks: Hand landmarks (21 x 3)
            
        Returns:
            Cropped hand region
        """
        # Get bounding box from landmarks
        x_coords = landmarks[:, 0]
        y_coords = landmarks[:, 1]
        
        x_min, x_max = int(x_coords.min()), int(x_coords.max())
        y_min, y_max = int(y_coords.min()), int(y_coords.max())
        
        # Add padding
        padding = 20
        h, w = frame.shape[:2]
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        # Crop
        cropped = frame[y_min:y_max, x_min:x_max]
        
        return cropped


class DataAugmentation:
    """Data augmentation for ASL video sequences"""
    
    def __init__(
        self,
        rotation_range: float = 10,
        zoom_range: float = 0.1,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        horizontal_flip: bool = False  # Usually False for ASL
    ):
        self.rotation_range = rotation_range
        self.zoom_range = zoom_range
        self.brightness_range = brightness_range
        self.horizontal_flip = horizontal_flip
    
    def augment_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply random augmentation to a frame"""
        # Random rotation
        if self.rotation_range > 0:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            frame = self._rotate(frame, angle)
        
        # Random zoom
        if self.zoom_range > 0:
            zoom_factor = np.random.uniform(1 - self.zoom_range, 1 + self.zoom_range)
            frame = self._zoom(frame, zoom_factor)
        
        # Random brightness
        if self.brightness_range:
            brightness = np.random.uniform(*self.brightness_range)
            frame = self._adjust_brightness(frame, brightness)
        
        # Horizontal flip (usually not used for ASL)
        if self.horizontal_flip and np.random.random() > 0.5:
            frame = cv2.flip(frame, 1)
        
        return frame
    
    def _rotate(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by angle"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h))
        return rotated
    
    def _zoom(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Zoom image by factor"""
        h, w = image.shape[:2]
        new_h, new_w = int(h * factor), int(w * factor)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        if factor > 1:
            # Crop center
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            zoomed = resized[start_h:start_h + h, start_w:start_w + w]
        else:
            # Pad
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            zoomed = cv2.copyMakeBorder(
                resized, pad_h, h - new_h - pad_h,
                pad_w, w - new_w - pad_w,
                cv2.BORDER_CONSTANT, value=0
            )
        
        return zoomed
    
    def _adjust_brightness(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust brightness by factor"""
        adjusted = np.clip(image * factor, 0, 255).astype(np.uint8)
        return adjusted


def extract_hand_features(landmarks: np.ndarray) -> np.ndarray:
    """
    Extract features from hand landmarks.
    
    Args:
        landmarks: Hand landmarks (21 x 3)
        
    Returns:
        Feature vector
    """
    features = []
    
    # Flatten landmarks
    features.extend(landmarks.flatten())
    
    # Add relative positions (to wrist)
    wrist = landmarks[0]
    relative_positions = landmarks - wrist
    features.extend(relative_positions.flatten())
    
    # Add distances between key points
    key_points = [0, 4, 8, 12, 16, 20]  # wrist, thumb tip, index tip, middle tip, ring tip, pinky tip
    for i in range(len(key_points)):
        for j in range(i + 1, len(key_points)):
            p1 = landmarks[key_points[i]]
            p2 = landmarks[key_points[j]]
            distance = np.linalg.norm(p1 - p2)
            features.append(distance)
    
    return np.array(features)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    preprocessor = VideoPreprocessor(
        target_size=(224, 224),
        sequence_length=30,
        normalize=True
    )
    
    print("Video Preprocessor initialized")
    print(f"Target size: {preprocessor.target_size}")
    print(f"Sequence length: {preprocessor.sequence_length}")
