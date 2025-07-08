🧠 Project Overview for Cursor
🎯 Goal: Full Sign Language Recognition System
The goal of this project is to build an end-to-end Sign Language Recognition system that can accurately translate video-based American Sign Language (ASL) into text or speech, using deep learning and computer vision.

🧱 Project Phases & Current Progress
✅ 1. Static ASL Alphabet Classification
Dataset: ASL Alphabet images.

Built and trained a VGG-style CNN model.

Achieved 99%+ accuracy on validation/test sets.

Integrated with webcam: live letter prediction.

➡️ Completed.

✅ 2. Word-Level Sign Recognition (Current Phase)
Dataset: WLASL (Word-Level ASL) — videos of isolated glosses (ASL words).

Steps completed so far:

✅ Downloaded and partially filtered WLASL videos (over 3,900 processed).

✅ Generated a labels.csv file mapping video filenames to gloss labels.

✅ Identified 825 unique gloss labels.

🛠 Working on: video_to_frames.py — script to convert each .mp4 video into a sequence of uniformly sampled and resized frames, saved as .npy.

➡️ This script has not successfully run yet — needs bugfixes due to corrupted videos and timeout issues.

⏳ 3. Temporal Modeling with LSTM / GRU
Use the CNN frame encoder on videos.

Extract frame-wise embeddings.

Train an LSTM/GRU/Transformer model for word classification based on temporal input.

Optionally use CTC loss for variable-length glosses.

⏳ 4. Real-Time Word Prediction Pipeline
Webcam feed processed frame-by-frame.

Frame buffer passed to temporal model.

Continuous prediction of signs (real-time gloss recognition).

⏳ 5. UI / Deployment
Build a minimal interface showing:

Webcam feed.

Live predicted gloss/letter.

Optionally generate sentence using language model.

Optionally output text-to-speech.

Export to TFLite / ONNX for lightweight deployment.