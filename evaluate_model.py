from src.data_loader import load_asl_dataset
from tensorflow.keras.models import load_model
import numpy as np

# Load validation set
_, val_ds = load_asl_dataset("data/raw/archive/asl_alphabet_train/asl_alphabet_train", img_size=(64, 64), batch_size=32)

# Load trained model
model = load_model("models/cnn/asl_cnn_model.h5")

# Evaluate
loss, accuracy = model.evaluate(val_ds)
print(f"\nValidation Accuracy: {accuracy:.4f}")
print(f"Validation Loss: {loss:.4f}")
