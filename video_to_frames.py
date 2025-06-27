import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

VIDEO_DIR = '/Users/saad/Documents/slcnn/WLASL/start_kit/raw_videos'
LABELS_FILE = '/Users/saad/Documents/slcnn/WLASL/start_kit/labels.csv'
OUTPUT_DIR = '/Users/saad/Documents/slcnn/WLASL/start_kit/frame_data'

IMG_SIZE = 64
FRAMES_PER_VIDEO = 16

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

df = pd.read_csv(LABELS_FILE)
df['filename'] = df['filename'].str.replace('.mp4', '', regex=False)  # remove .mp4

labels = sorted(df['gloss'].unique())
label_map = {label: idx for idx, label in enumerate(labels)}

for _, row in tqdm(df.iterrows(), total=len(df)):
    vid_id = row['filename']
    label = row['gloss']
    label_idx = label_map[label]

    video_path = os.path.join(VIDEO_DIR, f'{vid_id}.mp4')
    if not os.path.exists(video_path):
        continue

    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frames.append(frame)
    cap.release()

    if len(frames) >= FRAMES_PER_VIDEO:
        sampled = np.linspace(0, len(frames) - 1, FRAMES_PER_VIDEO, dtype=int)
        frames = [frames[i] for i in sampled]
        arr = np.stack(frames)
        np.save(os.path.join(OUTPUT_DIR, f'{vid_id}.npy'), {'frames': arr, 'label': label_idx})
