import cv2
import os
import numpy as np

VIDEO_PATH = '/Users/saad/Documents/slcnn/WLASL/start_kit/raw_videos/17712.mp4'
NPY_PATH = '/Users/saad/Documents/slcnn/WLASL/start_kit/frame_data/07068.npy'

if not os.path.exists(VIDEO_PATH):
    print('Video file does not exist.')
    exit(1)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print('Failed to open video with OpenCV.')
    exit(1)

frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()

print(f'Total frames read: {len(frames)}')
if frames:
    print(f'First frame shape: {frames[0].shape}')
else:
    print('No frames extracted.')

# Test loading the .npy file
if os.path.exists(NPY_PATH):
    data = np.load(NPY_PATH, allow_pickle=True).item()
    print(f"Loaded {NPY_PATH}")
    print(f"Keys: {list(data.keys())}")
    print(f"Frames shape: {data['frames'].shape}")
    print(f"Label: {data['label']}")
else:
    print(f"{NPY_PATH} does not exist.") 