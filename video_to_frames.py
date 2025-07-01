import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

VIDEO_DIR = '/Users/saad/Documents/slcnn/WLASL/start_kit/raw_videos'
LABELS_FILE = '/Users/saad/Documents/slcnn/WLASL/start_kit/labels.csv'
OUTPUT_DIR = '/Users/saad/Documents/slcnn/WLASL/start_kit/frame_data'
SKIPPED_LOG = '/Users/saad/Documents/slcnn/WLASL/start_kit/skipped_videos.txt'

IMG_SIZE = 64
FRAMES_PER_VIDEO = 16
TIMEOUT = 30  # seconds per video
N_WORKERS = 4

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

df = pd.read_csv(LABELS_FILE)
df['filename'] = df['filename'].str.replace('.mp4', '', regex=False)
labels = sorted(df['gloss'].unique())
label_map = {label: idx for idx, label in enumerate(labels)}

def process_video(args):
    vid_id, label, label_idx = args
    video_path = os.path.join(VIDEO_DIR, f'{vid_id}.mp4')
    output_path = os.path.join(OUTPUT_DIR, f'{vid_id}.npy')
    if not os.path.exists(video_path):
        return (vid_id, 'not_found')
    if os.path.exists(output_path):
        return (vid_id, 'already_processed')
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return (vid_id, 'not_openable')
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frames.append(frame)
        cap.release()
        if len(frames) < FRAMES_PER_VIDEO:
            return (vid_id, f'not_enough_frames:{len(frames)}')
        sampled = np.linspace(0, len(frames) - 1, FRAMES_PER_VIDEO, dtype=int)
        frames = [frames[i] for i in sampled]
        arr = np.stack(frames)
        np.save(output_path, {'frames': arr, 'label': label_idx})
        return (vid_id, 'success')
    except Exception as e:
        return (vid_id, f'error:{e}')

if __name__ == "__main__":
    # Limit to first 10 videos for testing
    test_df = df.head(10)
    tasks = [(row['filename'], row['gloss'], label_map[row['gloss']]) for _, row in test_df.iterrows()]
    skipped = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(process_video, task): task[0] for task in tasks}
        for future in tqdm(as_completed(futures), total=len(futures)):
            vid_id = futures[future]
            try:
                result = future.result(timeout=TIMEOUT)
            except Exception as e:
                skipped.append((vid_id, f'timeout_or_crash:{e}'))
                continue
            if result[1] != 'success' and result[1] != 'already_processed':
                skipped.append(result)
    if skipped:
        with open(SKIPPED_LOG, 'a') as f:
            for vid_id, reason in skipped:
                f.write(f"{vid_id},{reason}\n")
