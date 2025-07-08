import os
import tensorflow as tf
from src.data_loader import load_wlasl_sequence_dataset
from src.video_sign_bilstm_model import build_video_sign_bilstm_model

# --- Config ---
FRAME_DATA_DIR = '/content/drive/MyDrive/asl_project/WLASL/start_kit/frame_data/'
BATCH_SIZE = 8
IMG_SIZE = (224, 224)
FRAMES = 16
NUM_CLASSES = 825
EPOCHS = 30
LSTM_UNITS = 128
DROPOUT_RATE = 0.5
CNN_TRAINABLE = True  # Unfreeze CNN for fine-tuning
MODEL_SAVE_PATH = 'models/video_sign_bilstm_best.h5'

# --- Data Loading ---
print('Loading data...')
train_ds, val_ds = load_wlasl_sequence_dataset(
    frame_data_dir=FRAME_DATA_DIR,
    batch_size=BATCH_SIZE,
    img_size=IMG_SIZE
)

# --- Model Instantiation ---
print('Building model...')
model = build_video_sign_bilstm_model(
    input_shape=(FRAMES, IMG_SIZE[0], IMG_SIZE[1], 3),
    num_classes=NUM_CLASSES,
    lstm_units=LSTM_UNITS,
    cnn_trainable=CNN_TRAINABLE,
    dropout_rate=DROPOUT_RATE
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Lower learning rate
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- Callbacks ---
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
    tf.keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_loss'),
    tf.keras.callbacks.CSVLogger('training_log.csv', append=True)
]

# --- Training ---
print('Starting training...')
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

print(f'Training complete. Best model saved to {MODEL_SAVE_PATH}') 