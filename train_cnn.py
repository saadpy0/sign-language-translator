from src.data_loader import load_asl_dataset
from src.cnn_model import build_cnn_model
import tensorflow as tf

# Load dataset
train_ds, val_ds = load_asl_dataset("data/raw/archive/asl_alphabet_train/asl_alphabet_train", img_size=(64, 64), batch_size=32)

# Build model
model = build_cnn_model(input_shape=(64, 64, 3), num_classes=29)

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
history = model.fit(train_ds, validation_data=val_ds, epochs=10)

# Save model
model.save("models/cnn/asl_cnn_model.h5")
