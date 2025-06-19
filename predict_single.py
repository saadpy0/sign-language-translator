import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os
import matplotlib.pyplot as plt
import string

# Label map: 0 → A, 1 → B, ..., 28 → space/del/nothing
label_map = dict(enumerate(list(string.ascii_uppercase) + ["del", "nothing", "space"]))

# Load model
model = load_model("models/cnn/asl_cnn_model.h5")

# Path to test images
test_dir = "data/test_images"

# Loop through images
for fname in os.listdir(test_dir):
    if fname.lower().endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(test_dir, fname)

        # Load and preprocess image
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(64, 64))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        pred = model.predict(img_array)
        label = label_map[np.argmax(pred)]

        # Show result
        plt.imshow(img)
        plt.title(f"Predicted: {label}")
        plt.axis("off")
        plt.show()
