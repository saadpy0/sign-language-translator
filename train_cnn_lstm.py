import os
import tensorflow as tf
from src.data_loader import load_wlasl_sequence_dataset
from src.cnn_lstm_model import build_cnn_lstm_model

# Config
FRAME_DATA_DIR = '/content/drive/MyDrive/asl_project/WLASL/start_kit/frame_data'
BATCH_SIZE = 8  # Reduce if OOM
EPOCHS = 50
MODEL_SAVE_PATH = '/content/drive/MyDrive/asl_project/models/lstm/wlasl_cnn_lstm_best.keras'
NUM_FRAMES = 16
IMG_SIZE = (224, 224)
LSTM_UNITS = 128
NUM_CLASSES = 825

# Ensure model save directory exists
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# Data
print('Loading data...')
train_ds, val_ds = load_wlasl_sequence_dataset(
    frame_data_dir=FRAME_DATA_DIR,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    seed=123,
    img_size=IMG_SIZE
)

# Debug: Print a batch of data and labels
for batch in train_ds.take(1):
    frames, labels = batch
    print("[DEBUG] Frames shape:", frames.shape)
    print("[DEBUG] Labels:", labels.numpy())
    print("[DEBUG] Labels min/max:", labels.numpy().min(), labels.numpy().max())
    print("[DEBUG] Unique labels in batch:", set(labels.numpy()))

# Overfit test: Use a single batch
OVERFIT_TEST = True
if OVERFIT_TEST:
    train_ds = train_ds.take(1)
    val_ds = val_ds.take(1)

# Model
print('Building CNN+LSTM model...')
model = build_cnn_lstm_model(
    input_shape=(NUM_FRAMES, IMG_SIZE[0], IMG_SIZE[1], 3),
    num_classes=NUM_CLASSES,
    lstm_units=LSTM_UNITS,
    trainable_cnn=False
)
model.compile(
    optimizer='adam',
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

# Overfit test: Print predictions after training
if OVERFIT_TEST:
    for batch in train_ds.take(1):
        frames, labels = batch
        preds_after = model.predict(frames)
        print("[OVERFIT TEST] Predictions after training:", preds_after.argmax(axis=1))
        print("[OVERFIT TEST] True labels:", labels.numpy()) 