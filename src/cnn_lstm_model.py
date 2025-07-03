import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn_lstm_model(
    input_shape=(16, 224, 224, 3),
    num_classes=825,
    lstm_units=128,
    trainable_cnn=False
):
    """
    Builds a 2D CNN + LSTM model for video classification.
    Args:
        input_shape: (frames, height, width, channels)
        num_classes: number of output classes
        lstm_units: number of units in the LSTM layer
        trainable_cnn: whether to fine-tune the CNN backbone
    Returns:
        Keras Model
    """
    frames, height, width, channels = input_shape
    cnn_base = tf.keras.applications.MobileNetV2(
        input_shape=(height, width, channels),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    cnn_base.trainable = trainable_cnn

    # Input: (batch, frames, height, width, channels)
    video_input = layers.Input(shape=input_shape)
    # TimeDistributed applies the CNN to each frame
    x = layers.TimeDistributed(cnn_base)(video_input)
    # x shape: (batch, frames, cnn_features)
    x = layers.LSTM(lstm_units, return_sequences=False)(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=video_input, outputs=output)
    return model

# Example usage:
# model = build_cnn_lstm_model()
# model.summary() 