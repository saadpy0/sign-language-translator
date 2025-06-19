import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import string

# Load model
model = load_model("models/cnn/asl_cnn_model.h5")

# Label map: 0–25 → A–Z, 26–28 → del, nothing, space
label_map = dict(enumerate(list(string.ascii_uppercase) + ["del", "nothing", "space"]))

# Open webcam
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    print("Frame grabbed:", ret)
    if not ret:
        break

    # Define region of interest (ROI)
    roi = frame[50:350, 50:350]
    roi_resized = cv2.resize(roi, (64, 64))
    roi_normalized = roi_resized.astype("float32") / 255.0
    roi_input = np.expand_dims(roi_normalized, axis=0)

    # Predict
    preds = model.predict(roi_input)
    label = label_map[np.argmax(preds)]

    # Display prediction
    cv2.putText(frame, f"Prediction: {label}", (100, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.rectangle(frame, (50, 50), (350, 350), (255, 0, 0), 2)

    # Show frame
    cv2.imshow("ASL Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
