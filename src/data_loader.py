import tensorflow as tf
import numpy as np
import os
from glob import glob

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


def load_wlasl_sequence_dataset(frame_data_dir, batch_size=32, validation_split=0.2, seed=123):
    """
    Loads .npy frame sequence data for temporal modeling.
    Returns train and validation tf.data.Dataset objects.
    """
    npy_files = sorted(glob(os.path.join(frame_data_dir, '*.npy')))
    np.random.seed(seed)
    np.random.shuffle(npy_files)
    n_total = len(npy_files)
    n_val = int(n_total * validation_split)
    val_files = npy_files[:n_val]
    train_files = npy_files[n_val:]

    def load_npy(path):
        path = path.numpy().decode('utf-8')
        data = np.load(path, allow_pickle=True).item()
        frames = data['frames'].astype(np.float32) / 255.0  # normalize
        label = np.int32(data['label'])
        return frames, label

    def make_dataset(file_list):
        ds = tf.data.Dataset.from_tensor_slices(file_list)
        ds = ds.shuffle(buffer_size=len(file_list), seed=seed)
        def _load_and_set_shape(x):
            frames, label = tf.py_function(load_npy, [x], [tf.float32, tf.int32])
            frames.set_shape((16, 64, 64, 3))
            label.set_shape(())
            return frames, label
        ds = ds.map(_load_and_set_shape, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    def augment_frames(frames, label):
        def augment(frame):
            frame = tf.squeeze(frame)  # ensure shape (64, 64, 3)
            frame = tf.image.random_flip_left_right(frame)
            frame = tf.image.random_brightness(frame, max_delta=0.2)
            # Random zoom: resize to a random scale then back to 64x64
            scale = tf.random.uniform([], 0.9, 1.1)
            new_size = tf.cast(64 * scale, tf.int32)
            frame = tf.image.resize(frame, [new_size, new_size])
            frame = tf.image.resize_with_crop_or_pad(frame, 64, 64)
            return frame
        frames = tf.map_fn(augment, frames)
        return frames, label

    train_ds = make_dataset(train_files)
    # train_ds = train_ds.map(augment_frames, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = make_dataset(val_files)
    return train_ds, val_ds
