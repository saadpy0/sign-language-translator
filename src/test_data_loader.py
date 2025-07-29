#!/usr/bin/env python3
"""
Simplified test version of the WLASL data loader
Tests basic functionality without TensorFlow dependencies.
"""

import os
import json
import numpy as np
import cv2
from typing import Tuple, List, Dict
import random

class SimpleWLASLDataLoader:
    """
    Simplified data loader for WLASL dataset testing.
    """
    
    def __init__(self, 
                 frames_dir: str = "WLASL/start_kit/frames",
                 split_file: str = "WLASL/start_kit/splits/WLASL100.json",
                 num_frames: int = 16,
                 frame_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the simplified WLASL data loader.
        """
        self.frames_dir = frames_dir
        self.split_file = split_file
        self.num_frames = num_frames
        self.frame_size = frame_size
        
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
        
        return samples
    
    def _load_video_frames(self, frames_dir: str) -> np.ndarray:
        """
        Load and preprocess frames for a video.
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
            # Create one-hot encoded labels manually
            labels_one_hot = np.zeros((len(labels), self.num_classes))
            for i, label_idx in enumerate(labels):
                labels_one_hot[i, label_idx] = 1
            return frames, labels_one_hot
        
        return np.array([]), np.array([])

def test_data_loader():
    """Test the data loader functionality."""
    print("🧪 Testing WLASL Data Loader")
    print("=" * 40)
    
    try:
        # Initialize data loader
        loader = SimpleWLASLDataLoader(
            frames_dir="../WLASL/start_kit/frames",
            split_file="../WLASL/start_kit/splits/WLASL100.json",
            num_frames=16,
            frame_size=(224, 224)
        )
        
        print("✅ Data loader initialized successfully!")
        
        # Test with a few samples
        print("\n📊 Testing data loading...")
        frames, labels = loader.get_sample_data(num_samples=3)
        
        if len(frames) > 0:
            print(f"✅ Data loading test successful!")
            print(f"   - Sample frames shape: {frames.shape}")
            print(f"   - Sample labels shape: {labels.shape}")
            print(f"   - Frame value range: [{frames.min():.3f}, {frames.max():.3f}]")
            print(f"   - Number of classes: {loader.num_classes}")
            
            # Show some sample labels
            print(f"\n📝 Sample labels:")
            for i, label_idx in enumerate(np.argmax(labels, axis=1)):
                gloss = loader.idx_to_label[label_idx]
                print(f"   - Sample {i+1}: {gloss} (index: {label_idx})")
                
        else:
            print("❌ No test data loaded")
            return False
            
    except Exception as e:
        print(f"❌ Error testing data loader: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 All tests passed!")
    return True

if __name__ == "__main__":
    test_data_loader() 