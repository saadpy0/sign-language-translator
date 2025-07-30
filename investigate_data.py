#!/usr/bin/env python3
"""
Data Investigation Script for WLASL Dataset
Checks frame quality, label distribution, and data integrity.
"""

import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import Counter
import random

def check_data_structure():
    """Check the overall data structure and file organization."""
    print("🔍 Checking Data Structure")
    print("=" * 50)
    
    # Check main directories
    frames_dir = "/content/drive/MyDrive/asl/WLASL/start_kit/frames"
    split_file = "/content/drive/MyDrive/asl/WLASL/start_kit/splits/WLASL100.json"
    
    print(f"Frames directory exists: {os.path.exists(frames_dir)}")
    print(f"Split file exists: {os.path.exists(split_file)}")
    
    if os.path.exists(frames_dir):
        video_dirs = os.listdir(frames_dir)
        print(f"Number of video directories: {len(video_dirs)}")
        print(f"Sample video directories: {video_dirs[:5]}")
        
        # Check a few video directories
        for i, video_dir in enumerate(video_dirs[:3]):
            video_path = os.path.join(frames_dir, video_dir)
            frame_files = os.listdir(video_path)
            print(f"  {video_dir}: {len(frame_files)} frames")
            if frame_files:
                print(f"    Sample frames: {frame_files[:3]}")

def load_and_check_splits():
    """Load and analyze the split file."""
    print("\n📊 Analyzing Split File")
    print("=" * 50)
    
    split_file = "/content/drive/MyDrive/asl/WLASL/start_kit/splits/WLASL100.json"
    
    if not os.path.exists(split_file):
        print("❌ Split file not found!")
        return None
    
    with open(split_file, 'r') as f:
        data = json.load(f)
    
    print(f"Number of glosses: {len(data)}")
    
    # Analyze gloss distribution
    gloss_counts = {}
    total_instances = 0
    
    for gloss_entry in data:
        gloss = gloss_entry['gloss']
        instances = gloss_entry['instances']
        gloss_counts[gloss] = len(instances)
        total_instances += len(instances)
    
    print(f"Total instances: {total_instances}")
    print(f"Average instances per gloss: {total_instances / len(data):.2f}")
    
    # Show gloss distribution
    sorted_glosses = sorted(gloss_counts.items(), key=lambda x: x[1], reverse=True)
    print(f"\nTop 10 glosses by instance count:")
    for i, (gloss, count) in enumerate(sorted_glosses[:10]):
        print(f"  {i+1}. {gloss}: {count} instances")
    
    # Check split distribution
    split_counts = {'train': 0, 'val': 0, 'test': 0}
    for gloss_entry in data:
        for instance in gloss_entry['instances']:
            split_counts[instance['split']] += 1
    
    print(f"\nSplit distribution:")
    for split, count in split_counts.items():
        print(f"  {split}: {count} instances ({count/total_instances*100:.1f}%)")
    
    return data

def check_frame_quality():
    """Check the quality and consistency of extracted frames."""
    print("\n🖼️ Checking Frame Quality")
    print("=" * 50)
    
    frames_dir = "/content/drive/MyDrive/asl/WLASL/start_kit/frames"
    
    if not os.path.exists(frames_dir):
        print("❌ Frames directory not found!")
        return
    
    video_dirs = os.listdir(frames_dir)
    sample_videos = random.sample(video_dirs, min(5, len(video_dirs)))
    
    frame_stats = []
    
    for video_dir in sample_videos:
        video_path = os.path.join(frames_dir, video_dir)
        frame_files = sorted(os.listdir(video_path))
        
        if not frame_files:
            print(f"❌ No frames found in {video_dir}")
            continue
        
        # Check first few frames
        for i, frame_file in enumerate(frame_files[:3]):
            frame_path = os.path.join(video_path, frame_file)
            img = cv2.imread(frame_path)
            
            if img is None:
                print(f"❌ Could not read frame: {frame_path}")
                continue
            
            height, width, channels = img.shape
            frame_stats.append({
                'video': video_dir,
                'frame': frame_file,
                'shape': (height, width, channels),
                'mean_pixel': np.mean(img),
                'std_pixel': np.std(img)
            })
            
            print(f"✅ {video_dir}/{frame_file}: {height}x{width}x{channels}, "
                  f"mean={np.mean(img):.1f}, std={np.std(img):.1f}")
    
    # Analyze frame statistics
    if frame_stats:
        shapes = [stat['shape'] for stat in frame_stats]
        unique_shapes = set(shapes)
        print(f"\nFrame shape consistency: {len(unique_shapes)} unique shapes")
        for shape in unique_shapes:
            count = shapes.count(shape)
            print(f"  {shape}: {count} frames")

def visualize_sample_frames():
    """Visualize sample frames from different videos."""
    print("\n👁️ Visualizing Sample Frames")
    print("=" * 50)
    
    frames_dir = "/content/drive/MyDrive/asl/WLASL/start_kit/frames"
    
    if not os.path.exists(frames_dir):
        print("❌ Frames directory not found!")
        return
    
    video_dirs = os.listdir(frames_dir)
    sample_videos = random.sample(video_dirs, min(3, len(video_dirs)))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, video_dir in enumerate(sample_videos):
        video_path = os.path.join(frames_dir, video_dir)
        frame_files = sorted(os.listdir(video_path))
        
        if frame_files:
            # Load first frame
            frame_path = os.path.join(video_path, frame_files[0])
            img = cv2.imread(frame_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            axes[i].imshow(img_rgb)
            axes[i].set_title(f"{video_dir}\n{img_rgb.shape}")
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('/content/drive/MyDrive/asl/sample_frames.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Sample frames saved to /content/drive/MyDrive/asl/sample_frames.png")

def check_label_mapping():
    """Check if video IDs in frames match video IDs in splits."""
    print("\n🏷️ Checking Label Mapping")
    print("=" * 50)
    
    frames_dir = "/content/drive/MyDrive/asl/WLASL/start_kit/frames"
    split_file = "/content/drive/MyDrive/asl/WLASL/start_kit/splits/WLASL100.json"
    
    if not os.path.exists(frames_dir) or not os.path.exists(split_file):
        print("❌ Required files not found!")
        return
    
    # Get video IDs from frames directory
    frame_video_ids = set(os.listdir(frames_dir))
    print(f"Video IDs in frames directory: {len(frame_video_ids)}")
    
    # Get video IDs from split file
    with open(split_file, 'r') as f:
        data = json.load(f)
    
    split_video_ids = set()
    for gloss_entry in data:
        for instance in gloss_entry['instances']:
            split_video_ids.add(instance['video_id'])
    
    print(f"Video IDs in split file: {len(split_video_ids)}")
    
    # Check overlap
    overlap = frame_video_ids.intersection(split_video_ids)
    missing_in_frames = split_video_ids - frame_video_ids
    extra_in_frames = frame_video_ids - split_video_ids
    
    print(f"Overlap: {len(overlap)} videos")
    print(f"Missing in frames: {len(missing_in_frames)} videos")
    print(f"Extra in frames: {len(extra_in_frames)} videos")
    
    if missing_in_frames:
        print(f"Sample missing videos: {list(missing_in_frames)[:5]}")
    
    if extra_in_frames:
        print(f"Sample extra videos: {list(extra_in_frames)[:5]}")
    
    coverage = len(overlap) / len(split_video_ids) * 100
    print(f"Coverage: {coverage:.1f}%")

def check_frame_sequence_quality():
    """Check the quality of frame sequences for a few videos."""
    print("\n🎬 Checking Frame Sequence Quality")
    print("=" * 50)
    
    frames_dir = "/content/drive/MyDrive/asl/WLASL/start_kit/frames"
    
    if not os.path.exists(frames_dir):
        print("❌ Frames directory not found!")
        return
    
    video_dirs = os.listdir(frames_dir)
    sample_videos = random.sample(video_dirs, min(3, len(video_dirs)))
    
    for video_dir in sample_videos:
        video_path = os.path.join(frames_dir, video_dir)
        frame_files = sorted(os.listdir(video_path))
        
        if len(frame_files) < 2:
            print(f"⚠️ {video_dir}: Only {len(frame_files)} frames")
            continue
        
        print(f"\n📹 {video_dir}: {len(frame_files)} frames")
        
        # Check frame sequence
        frame_numbers = []
        for frame_file in frame_files:
            if frame_file.startswith('frame_') and frame_file.endswith('.jpg'):
                try:
                    frame_num = int(frame_file[6:-4])  # Extract number from 'frame_XXXX.jpg'
                    frame_numbers.append(frame_num)
                except ValueError:
                    print(f"  ⚠️ Invalid frame filename: {frame_file}")
        
        if frame_numbers:
            frame_numbers.sort()
            print(f"  Frame range: {frame_numbers[0]} to {frame_numbers[-1]}")
            print(f"  Missing frames: {len(range(frame_numbers[0], frame_numbers[-1]+1)) - len(frame_numbers)}")
            
            # Check for gaps
            gaps = []
            for i in range(len(frame_numbers)-1):
                if frame_numbers[i+1] - frame_numbers[i] > 1:
                    gaps.append((frame_numbers[i], frame_numbers[i+1]))
            
            if gaps:
                print(f"  Gaps found: {gaps[:3]}")  # Show first 3 gaps
            else:
                print(f"  ✅ No gaps in frame sequence")

def main():
    """Main investigation function."""
    print("🔬 WLASL Dataset Investigation")
    print("=" * 60)
    
    # Run all checks
    check_data_structure()
    split_data = load_and_check_splits()
    check_frame_quality()
    visualize_sample_frames()
    check_label_mapping()
    check_frame_sequence_quality()
    
    print("\n" + "=" * 60)
    print("✅ Investigation complete!")
    print("\n📋 Summary of findings will be displayed above.")

if __name__ == "__main__":
    main() 