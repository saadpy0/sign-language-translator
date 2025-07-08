import os
import tensorflow as tf
from src.data_loader import load_wlasl_sequence_dataset
from src.cnn_lstm_model import build_cnn_lstm_model

# Config
FRAME_DATA_DIR = 'WLASL/start_kit/frame_data'
BATCH_SIZE = 32
EPOCHS = 50
MODEL_SAVE_PATH = 'models/lstm/wlasl_cnn_lstm_best.keras'

# Data
print('Loading data...')
train_ds, val_ds = load_wlasl_sequence_dataset(
    frame_data_dir=FRAME_DATA_DIR,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    seed=123
)

# Model
print('Building model...')
model = build_cnn_lstm_model()
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