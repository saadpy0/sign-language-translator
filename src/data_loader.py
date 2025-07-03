import tensorflow as tf
import numpy as np
import os
from glob import glob
import cv2

def load_asl_dataset(data_dir, img_size=(64, 64), batch_size=32, validation_split=0.2):
    # Load raw datasets
    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory=data_dir,
        validation_split=validation_split,
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        directory=data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )

    # Normalize pixel values to [0, 1]
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # Optimize data pipeline
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds


def load_wlasl_sequence_dataset(frame_data_dir, batch_size=32, validation_split=0.2, seed=123, img_size=(224, 224), allowed_labels=None):
    """
    Loads .npy frame sequence data for temporal modeling.
    Returns train and validation tf.data.Dataset objects.
    If allowed_labels is provided, only samples with those labels are loaded.
    """
    npy_files = sorted(glob(os.path.join(frame_data_dir, '*.npy')))
    np.random.seed(seed)
    np.random.shuffle(npy_files)
    
    # Filter files by allowed_labels if provided
    if allowed_labels is not None:
        filtered_files = []
        for path in npy_files:
            data = np.load(path, allow_pickle=True).item()
            label = int(data['label'])
            if label in allowed_labels:
                filtered_files.append(path)
        npy_files = filtered_files

    n_total = len(npy_files)
    n_val = int(n_total * validation_split)
    val_files = npy_files[:n_val]
    train_files = npy_files[n_val:]

    def load_npy(path):
        path = path.numpy().decode('utf-8')
        data = np.load(path, allow_pickle=True).item()
        frames = data['frames'].astype(np.float32)
        # Resize to img_size and normalize to [-1, 1]
        frames = np.stack([cv2.resize(f, img_size) for f in frames])
        frames = (frames / 127.5) - 1.0
        label = np.int32(data['label'])
        # Debug: Print info for the first few files
        if not hasattr(load_npy, 'counter'):
            load_npy.counter = 0
        if load_npy.counter < 5:
            print(f"[DEBUG] File: {path}")
            print(f"[DEBUG] Frames shape: {frames.shape}, dtype: {frames.dtype}")
            print(f"[DEBUG] Label: {label}, type: {type(label)}")
        load_npy.counter += 1
        # Check label validity
        if np.isnan(label) or label < 0 or label >= 825:
            print(f"[ERROR] Invalid label {label} in file {path}")
        return frames, label

    def make_dataset(file_list):
        ds = tf.data.Dataset.from_tensor_slices(file_list)
        ds = ds.shuffle(buffer_size=len(file_list), seed=seed)
        def _load_and_set_shape(x):
            frames, label = tf.py_function(load_npy, [x], [tf.float32, tf.int32])
            frames.set_shape((16, img_size[0], img_size[1], 3))
            label.set_shape(())
            return frames, label
        ds = ds.map(_load_and_set_shape, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = make_dataset(train_files)
    val_ds = make_dataset(val_files)
    return train_ds, val_ds
