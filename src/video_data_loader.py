import os
import json
import numpy as np
import cv2
from typing import Tuple, List, Dict, Optional
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import random

class WLASLVideoDataLoader:
    """
    Data loader for WLASL (Word-Level American Sign Language) dataset.
    Loads video frames and maps them to gloss labels.
    """
    
    def __init__(self, 
                 frames_dir: str = "WLASL/start_kit/frames",
                 split_file: str = "WLASL/start_kit/splits/WLASL100.json",
                 num_frames: int = 16,
                 frame_size: Tuple[int, int] = (224, 224),
                 batch_size: int = 32,
                 shuffle: bool = True):
        """
        Initialize the WLASL data loader.
        
        Args:
            frames_dir: Directory containing extracted video frames
            split_file: JSON file containing train/val/test splits
            num_frames: Number of frames to sample from each video
            frame_size: Target size for frame resizing (height, width)
            batch_size: Batch size for training
            shuffle: Whether to shuffle the data
        """
        self.frames_dir = frames_dir
        self.split_file = split_file
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Load split data
        self.splits = self._load_splits()
        
        # Create label mapping
        self.label_to_idx, self.idx_to_label = self._create_label_mapping()
        self.num_classes = len(self.label_to_idx)
        
        # Prepare data samples
        self.train_samples = self._prepare_samples('train')
        self.val_samples = self._prepare_samples('val')
        self.test_samples = self._prepare_samples('test')
        
        print(f"Dataset loaded:")
        print(f"  - Train samples: {len(self.train_samples)}")
        print(f"  - Val samples: {len(self.val_samples)}")
        print(f"  - Test samples: {len(self.test_samples)}")
        print(f"  - Number of classes: {self.num_classes}")
    
    def _load_splits(self) -> Dict:
        """Load the train/val/test splits from JSON file."""
        with open(self.split_file, 'r') as f:
            data = json.load(f)
        
        # Convert the gloss-based structure to split-based structure
        splits = {'train': [], 'val': [], 'test': []}
        
        for gloss_entry in data:
            gloss = gloss_entry['gloss']
            for instance in gloss_entry['instances']:
                sample = {
                    'video_id': instance['video_id'],
                    'gloss': gloss,
                    'split': instance['split']
                }
                splits[instance['split']].append(sample)
        
        return splits
    
    def _create_label_mapping(self) -> Tuple[Dict, Dict]:
        """Create mapping between gloss labels and indices."""
        all_labels = set()
        
        # Collect all unique labels from all splits
        for split_name in ['train', 'val', 'test']:
            if split_name in self.splits:
                for sample in self.splits[split_name]:
                    all_labels.add(sample['gloss'])
        
        # Create mappings
        label_to_idx = {label: idx for idx, label in enumerate(sorted(all_labels))}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        
        return label_to_idx, idx_to_label
    
    def _prepare_samples(self, split_name: str) -> List[Dict]:
        """Prepare samples for a specific split."""
        if split_name not in self.splits:
            return []
        
        samples = []
        for sample in self.splits[split_name]:
            video_id = str(sample['video_id'])
            gloss = sample['gloss']
            
            # Check if frames exist for this video
            video_frames_dir = os.path.join(self.frames_dir, video_id)
            if os.path.exists(video_frames_dir):
                samples.append({
                    'video_id': video_id,
                    'gloss': gloss,
                    'label_idx': self.label_to_idx[gloss],
                    'frames_dir': video_frames_dir
                })
        
        if self.shuffle:
            random.shuffle(samples)
        
        return samples
    
    def _load_video_frames(self, frames_dir: str) -> np.ndarray:
        """
        Load and preprocess frames for a video.
        
        Args:
            frames_dir: Directory containing frame images
            
        Returns:
            Array of shape (num_frames, height, width, channels)
        """
        # Get all frame files
        frame_files = sorted([f for f in os.listdir(frames_dir) 
                            if f.endswith('.jpg') or f.endswith('.png')])
        
        if not frame_files:
            raise ValueError(f"No frame files found in {frames_dir}")
        
        # Sample frames uniformly
        if len(frame_files) <= self.num_frames:
            # If we have fewer frames than needed, repeat the last frame
            selected_frames = frame_files + [frame_files[-1]] * (self.num_frames - len(frame_files))
        else:
            # Uniform sampling
            indices = np.linspace(0, len(frame_files) - 1, self.num_frames, dtype=int)
            selected_frames = [frame_files[i] for i in indices]
        
        # Load and preprocess frames
        frames = []
        for frame_file in selected_frames:
            frame_path = os.path.join(frames_dir, frame_file)
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.frame_size)
            frame = frame.astype(np.float32) / 255.0  # Normalize to [0, 1]
            frames.append(frame)
        
        return np.array(frames)
    
    def _data_generator(self, samples: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Generate batches of data."""
        while True:
            if self.shuffle:
                random.shuffle(samples)
            
            for i in range(0, len(samples), self.batch_size):
                batch_samples = samples[i:i + self.batch_size]
                
                batch_frames = []
                batch_labels = []
                
                for sample in batch_samples:
                    try:
                        frames = self._load_video_frames(sample['frames_dir'])
                        batch_frames.append(frames)
                        batch_labels.append(sample['label_idx'])
                    except Exception as e:
                        print(f"Error loading video {sample['video_id']}: {e}")
                        continue
                
                if batch_frames:
                    batch_frames = np.array(batch_frames)
                    batch_labels = to_categorical(batch_labels, num_classes=self.num_classes)
                    
                    yield batch_frames, batch_labels
    
    def get_train_generator(self):
        """Get training data generator."""
        return self._data_generator(self.train_samples)
    
    def get_val_generator(self):
        """Get validation data generator."""
        return self._data_generator(self.val_samples)
    
    def get_test_generator(self):
        """Get test data generator."""
        return self._data_generator(self.test_samples)
    
    def get_train_steps(self) -> int:
        """Get number of training steps per epoch."""
        return len(self.train_samples) // self.batch_size
    
    def get_val_steps(self) -> int:
        """Get number of validation steps per epoch."""
        return len(self.val_samples) // self.batch_size
    
    def get_test_steps(self) -> int:
        """Get number of test steps per epoch."""
        return len(self.test_samples) // self.batch_size
    
    def get_sample_data(self, num_samples: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Get a few sample batches for testing."""
        samples = self.train_samples[:num_samples]
        frames = []
        labels = []
        
        for sample in samples:
            try:
                video_frames = self._load_video_frames(sample['frames_dir'])
                frames.append(video_frames)
                labels.append(sample['label_idx'])
            except Exception as e:
                print(f"Error loading sample {sample['video_id']}: {e}")
                continue
        
        if frames:
            frames = np.array(frames)
            labels = to_categorical(labels, num_classes=self.num_classes)
            return frames, labels
        
        return np.array([]), np.array([])

# Example usage
if __name__ == "__main__":
    # Test the data loader
    loader = WLASLVideoDataLoader(
        frames_dir="WLASL/start_kit/frames",
        split_file="WLASL/start_kit/splits/WLASL100.json",
        num_frames=16,
        frame_size=(224, 224),
        batch_size=4
    )
    
    # Test with a few samples
    frames, labels = loader.get_sample_data(num_samples=3)
    print(f"Sample frames shape: {frames.shape}")
    print(f"Sample labels shape: {labels.shape}")
    
    # Test generator
    train_gen = loader.get_train_generator()
    batch_frames, batch_labels = next(train_gen)
    print(f"Generator batch frames shape: {batch_frames.shape}")
    print(f"Generator batch labels shape: {batch_labels.shape}") 