import os
import tensorflow as tf
from src.data_loader import load_wlasl_sequence_dataset
from src.i3d_model import build_i3d_model

# Config
FRAME_DATA_DIR = '/content/drive/My Drive/asl_project/data'
BATCH_SIZE = 16  # 3D CNNs are memory intensive
EPOCHS = 50
MODEL_SAVE_PATH = 'models/lstm/wlasl_i3d_best.keras'
PRETRAINED_WEIGHTS = 'models/rgb_inception_i3d_kinetics_only_tf_dim_ordering_tf_kernels_no_top.h5'
NUM_FRAMES = 16  # Adjust if your .npy files have a different number of frames
IMG_SIZE = (224, 224)

# Data
print('Loading data...')
train_ds, val_ds = load_wlasl_sequence_dataset(
    frame_data_dir=FRAME_DATA_DIR,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    seed=123,
    img_size=IMG_SIZE
)

# Set the correct number of classes based on the dataset
num_classes = 825
print(f'Number of classes: {num_classes}')

# Model
print('Building I3D model...')
model = build_i3d_model(
    input_shape=(NUM_FRAMES, IMG_SIZE[0], IMG_SIZE[1], 3),
    num_classes=num_classes,
    dropout_rate=0.5,
    weights_path=PRETRAINED_WEIGHTS
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# Callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy', mode='max'),
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
]

# Train
print('Training...')
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

print(f'Training complete. Best model saved to {MODEL_SAVE_PATH}') 