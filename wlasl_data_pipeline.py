import os
import subprocess
import sys
import random
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

def run_cmd(cmd):
    print(f"\n[Running] {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"[Error] Command failed: {cmd}")
        sys.exit(1)

# 1. Clone WLASL repo if not already present
if not os.path.exists('WLASL'):
    run_cmd('git clone https://github.com/dxli94/WLASL.git')

os.chdir('WLASL/start_kit')

# 2. Install dependencies
run_cmd(f'{sys.executable} -m pip install yt-dlp opencv-python tqdm matplotlib')

# 3. Download raw videos
run_cmd(f'{sys.executable} video_downloader.py')

# 4. Extract trimmed video clips
run_cmd(f'{sys.executable} preprocess.py')

# 5. Extract frames from trimmed videos
video_dir = 'videos'
frames_dir = 'frames'
os.makedirs(frames_dir, exist_ok=True)

for video_file in tqdm(os.listdir(video_dir)):
    if video_file.endswith('.mp4'):
        video_path = os.path.join(video_dir, video_file)
        vidcap = cv2.VideoCapture(video_path)
        success, image = vidcap.read()
        count = 0
        video_name = os.path.splitext(video_file)[0]
        video_frames_dir = os.path.join(frames_dir, video_name)
        os.makedirs(video_frames_dir, exist_ok=True)
        while success:
            cv2.imwrite(os.path.join(video_frames_dir, f"frame_{count:04d}.jpg"), image)
            success, image = vidcap.read()
            count += 1

print("Frame extraction complete.")

# 6. Visualize a few samples
sample_videos = random.sample(os.listdir(frames_dir), min(3, len(os.listdir(frames_dir))))
for vid in sample_videos:
    frame_files = sorted(os.listdir(os.path.join(frames_dir, vid)))
    if frame_files:
        img = cv2.imread(os.path.join(frames_dir, vid, frame_files[0]))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"Sample from {vid}")
        plt.show() 